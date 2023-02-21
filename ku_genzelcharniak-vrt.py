"""
Maria Kunilovskaya
8 Feb: replaced "\r", which could not be avoided in Perl, with "_" for ngram tokenisation (output is tested)
31 January 2023:
from genzelcharniak-vrt_v0.2.1.pl which accepts TreeTagger output
Annotate a vrt file with bits per word (surprisal) adding to the input new fields with values for:
(1) The cross entropy H(Doc|Rest)
(2) The entropy H(Doc)
where Doc is the language model for the document at hand, Rest for the rest of the corpus
Requires sentence tags <s>, </s> to calculate the entropy per sentence

TESTRUN:
-- python3 ku_genzelcharniak-vrt.py input/brown_exerpt_A-1.vrt output/py_BROWN-SPR.vrt

NB! ku_test_output.py compares the output of ku_genzelcharniak-vrt.py and genzelcharniak-vrt_v0.2.1.pl
"""


import argparse
from collections import defaultdict
from collections import OrderedDict
from operator import itemgetter
import math


def collect_terms(file):
    terms = defaultdict(int)
    tokens = []
    try:
        with open(file, 'r') as f:
            for line in f:
                if line[0] == "<":
                    pass
                else:
                    # avoid stripping \r
                    fields = line.strip().split("\t")
                    term = fields[0].strip().lower()  # token
                    pos = fields[1]
                    if pos == "SENT":
                        term = "SENT"
                    terms[term] += 1
                    tokens.append(term)
    except FileNotFoundError:
        print(f"Can't open {file}")
    return terms, tokens


def collect_ngrams(file, terms=None, maxngram=None):
    with open(file, "r") as f:
        prevterms = []
        ngrams = defaultdict(int)
        ngramtypes = defaultdict(int)
        # insert a sentence marker at the beginning of a document
        for line in f:
            if line.startswith("<text"):  # this will not work for RusLTC pro and stu
                prevterms = []
                ngrams, ngramtypes, prevterms = addup("SENT", ngrams, ngramtypes, prevterms)
                prevterms = ["SENT"]
            elif line.startswith("</text"):
                # subtract last sentence marker to compensate for the extra sentence marker at the beginning of text.
                ngrams["SENT"] -= 1
                ngrams["_"] -= 1
            elif line.startswith("<"):
                pass
            else:
                fields = line.strip().split("\t")
                term = fields[0].strip().lower()
                pos = fields[1]
                if pos == "SENT":  # UD has no SENT markers
                    term = "SENT"
                elif term in terms and terms[term] == 1:
                    term = "UNK"
                if len(prevterms) > maxngram:
                    prevterms.pop(0)
                ngrams, ngramtypes, prevterms = addup(term, ngrams, ngramtypes, prevterms)
                prevterms.append(term)
                if pos == "SENT":
                    prevterms = ["SENT"]

    return ngrams, ngramtypes, prevterms


def addup(term, ngrams=None, ngramtypes=None, prevterms=None):
    ngram = term

    # Unigram probability of $term
    if ngram in ngrams:
        ngrams[ngram] += 1  # it is a default dict, it does not throw an error if the key is not in
    else:
        ngrams[ngram] = 1
        # counter for unigram types (counted only once)
        if "_" in ngramtypes:
            ngramtypes["_"] += 1
        else:
            ngramtypes["_"] = 1

    # counter for unigram tokens
    if "_" in ngrams:
        ngrams["_"] += 1
    else:
        ngrams["_"] = 1

    # same counters for all n-grams (tokens and types of orders 1,2,3)
    # sliding window in sentence boundaries: max context 3 + term
    for i in range(len(prevterms), 0, -1):
        ngram = prevterms[i - 1] + "_" + ngram
        # splitting once starting from the right and getting first element, effectively deleting the last word
        context = ngram.rsplit("_", 1)[0]
        # adding bigrams and trigrams to the unigrams obtained in collect_terms
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
            if context in ngramtypes:
                ngramtypes[context] += 1
            else:
                ngramtypes[context] = 1

    return ngrams, ngramtypes, prevterms


def addupdoc(term, ngrams=None, ngramtypes=None, prevterms=None, docngrams=None, docngramtypes=None,
             restngramtypes=None):
    ngram = term
    if ngram in docngrams:
        docngrams[ngram] += 1
    else:
        docngrams[ngram] = 1
        # counting unigram types
        if "_" in docngramtypes:
            docngramtypes["_"] += 1
        else:
            docngramtypes["_"] = 1

    if "_" not in restngramtypes:
        restngramtypes["_"] = ngramtypes["_"]

    # ngram only occurs in this document
    if ngrams[ngram] - docngrams[ngram] == 0:
        restngramtypes["_"] -= 1
    if "_" in docngrams:
        docngrams["_"] += 1
    else:
        docngrams["_"] = 1
    for i in range(len(prevterms), 0, -1):  # iterate from max 3 to 1
        ngram = prevterms[i - 1] + "_" + ngram

        context = ngram.rsplit("_", 1)[0]
        if ngram in docngrams:
            docngrams[ngram] += 1
        else:
            docngrams[ngram] = 1
            if context in docngramtypes:
                docngramtypes[context] += 1
            else:
                docngramtypes[context] = 1

        if context not in restngramtypes:
            restngramtypes[context] = ngramtypes[context]

        if ngrams[ngram] - docngrams[ngram] == 0:
            restngramtypes[context] -= 1

    return ngrams, prevterms, docngrams, docngramtypes, restngramtypes


def get_wb_ent(ngram, count=None, docngramtypes=None, docngrams=None):
    if count > 8:
        raise Exception("what? " + ngram)
    if ngram == '':  # exausted rest, i.e. right-hand side context
        return 1.0 / docngramtypes["_"]

    context = ngram
    rest = ngram
    if "_" in context:
        # perl: $context =~ s/\r[^\r]+$//; = remove all characters from the first \r character to the end of the string
        context = context.rsplit("_", 1)[0]  # get first word, lose \r
        # split once at the first _ and get the contents on the right side of the split, in effect losing the first word
        rest = rest.split("_", 1)[1]
    else:
        context = "_"
        rest = ''

    typecount = docngramtypes[context]
    if docngrams[ngram] == 0:
        return get_wb_ent(rest, count=count, docngramtypes=docngramtypes, docngrams=docngrams)

    mle = docngrams[ngram] / docngrams[context]
    lambda1 = docngrams[context] / (docngrams[context] + typecount)

    return lambda1 * mle + (1 - lambda1) * get_wb_ent(rest, count + 1, docngramtypes=docngramtypes, docngrams=docngrams)


def get_ent_bits(term, prevterms=None, docngramtypes=None, docngrams=None):
    ngram = term
    for i in range(len(prevterms), 0, -1):
        ngram = prevterms[i - 1] + "_" + ngram
    return -math.log2(get_wb_ent(ngram, count=0, docngramtypes=docngramtypes, docngrams=docngrams))


def get_wb_cross(ngram, count=None, ngrams=None, docngrams=None, restngramtypes=None):
    if count > 8:
        raise Exception("what? {}".format(ngram))
    if ngram == '':
        return 1.0 / restngramtypes["_"]

    context = ngram
    rest = ngram
    if "_" in context:
        context = context.rsplit("_", 1)[0]  # extract the portion of the string before the last \r
        rest = rest.split("_", 1)[1]  # extract the portion of the string after the first \r
    else:
        context = "_"
        rest = ''

    typecount = restngramtypes[context]
    # if freq in corpus == freq in the current doc
    if ngrams[ngram] - docngrams[ngram] == 0:
        return get_wb_cross(rest, count=count, ngrams=ngrams, docngrams=docngrams, restngramtypes=restngramtypes)
    # maximum likelihood estimation = hits (excluding this doc) normalised to size of corpus (excluding this doc)
    mle = (ngrams[ngram] - docngrams[ngram]) / (ngrams[context] - docngrams[context])
    lambda1 = (ngrams[context] - docngrams[context]) / (ngrams[context] - docngrams[context] + typecount)

    return lambda1 * mle + (1 - lambda1) * get_wb_cross(rest, count + 1, ngrams=ngrams, docngrams=docngrams,
                                                        restngramtypes=restngramtypes)


# this function is supposed to produce additional probability from the sum of (i) relative position of ngram in a doc
# and (ii) ngram relative frequency weighted by inverted gamma
def get_cross_bits(term, prevterms=None, ngrams=None, docngrams=None, restngramtypes=None,
                   cache_tokens=None, cache_terms=None, cache_count=None, last_occ=None,
                   tau=None, gamma=None, lmbda=None):
    ngram = term
    for i in range(len(prevterms), 0, -1):
        ngram = prevterms[i - 1] + "_" + ngram

    # own probability of the term:
    prob = get_wb_cross(ngram, count=0, ngrams=ngrams, docngrams=docngrams, restngramtypes=restngramtypes)

    if cache_tokens > 0:  # skipping the first term (i.e. added SENT in the doc beginning)
        # the model exludes the current document:
        # (1-gamma) is multiplied by the quotient of (counts of item less counts in the current doc) and (size of corpus less current doc)
        try:
        # here, new terms are routinely absent from last_occ and cache_terms dicts,
        # they are added below the call of get_cross_bits function in compute_bits
        # there is no update loop for these dicts
            cacheprob = (gamma * (tau ** (cache_count - last_occ[term])) * cache_terms[term] / cache_tokens) + \
                        (1 - gamma) * ((ngrams[term] - docngrams[term]) / (ngrams["_"] - docngrams["_"]))
        # in Perl, if a key does not exist in the hash, it will return undef instead of throwing an error
        # If used in the expression, the expression will evaluate to 0
        except KeyError:
            cacheprob = (gamma * (tau ** (cache_count - 0)) * 0 / cache_tokens) + (
                    1 - gamma) * ((ngrams[term] - docngrams[term]) / (ngrams["_"] - docngrams["_"]))
        prob = lmbda * prob + (1 - lmbda) * cacheprob

    if prob > 1:
        context = ngram
        context = context.rsplit("_", 1)[0]
        raise Exception(
            f"{ngram},{prob},{docngrams[ngram]},{restngramtypes[context]},{docngrams[term]}")  # {restngramtypes['\r ']}

    return -math.log2(prob)


def compute_bits(lines=None, terms=None, ngrams=None, docngrams=None, docngramtypes=None, restngramtypes=None,
                 outf=None, opt_nocross=None, opt_noent=None, cross_name=None, ent_name=None, maxngram=None,
                 tau=None, gamma=None, lmbda=None):
    prevterms = ["SENT"]
    c_sent = 0
    cross_sum = 0
    ent_sum = 0
    # cache_... seem to be doc-level counts: they are advanced inside lines, refreshed after each text id=
    cache_tokens = 0  # with tau==1, cache_tokens has the same value as cache_count, i.e position of the item in the doc
    cache_count = 0
    last_occ = {}  # hash for last occurrence of a term in document (used for decay of cache)
    cache_terms = {}

    out_buffer = ""
    sent_buffer = ""
    sentence_atts = ""
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("<text"):
            out_buffer += line  # don't strip lines for a more readable xml
        elif line.startswith("</text"):
            out_buffer += line

            # write to file command
            no_empty_lines = [line for line in out_buffer.splitlines() if line]  # delete empty lines at doc end
            out_buffer = "\n".join(no_empty_lines)
            print(out_buffer, file=outf)

            # end of this text
            break
        elif line.startswith("<s"):
            sentence_atts = line[3:].strip()[:-1]  # get contents of <s> tag, losing the tag, which adding a newline?
            out_buffer += line
        elif line.startswith("</s"):
            cross_avg = 0
            ent_avg = 0
            if c_sent > 0:
                cross_avg = cross_sum / c_sent
                ent_avg = ent_sum / c_sent
            if opt_nocross:
                out_buffer += f"<s{sentence_atts} {ent_name}=\"{ent_avg:.2f}\">\n{sent_buffer}"
            elif opt_noent:
                out_buffer += f"<s{sentence_atts} {cross_name}=\"{cross_avg:.2f}\">\n{sent_buffer}"
            else:
                out_buffer += f"<s{sentence_atts} {cross_name}=\"{cross_avg:.2f}\" {ent_name}=\"{ent_avg:.2f}\">\n{sent_buffer}"
            cross_sum = 0
            ent_sum = 0
            c_sent = 0
            sent_buffer = ""
            out_buffer += line
        else:
            # processing lines containing tokens
            fields = line.strip().split("\t")
            term = fields[0]
            pos = fields[1]
            term1 = term.strip().lower()
            if term1 in terms and terms[term1] == 1:
                term1 = "UNK"
            elif pos == "SENT":
                term1 = "SENT"
            if len(prevterms) > maxngram:
                prevterms.pop(0)

            # ===== calculate spr H(Doc|Rest) and srplocal (self-surprisal, H(Doc))====
            cross_bits = get_cross_bits(term1, prevterms=prevterms, ngrams=ngrams, docngrams=docngrams,
                                        restngramtypes=restngramtypes, cache_tokens=cache_tokens,
                                        cache_terms=cache_terms, cache_count=cache_count, last_occ=last_occ,
                                        tau=tau, gamma=gamma, lmbda=lmbda)

            ent_bits = get_ent_bits(term1, prevterms=prevterms, docngramtypes=docngramtypes, docngrams=docngrams)

            # formating vrt word-lines appending the srp and/or srplocal values
            if opt_nocross:
                sent_buffer += f"{line.strip()}\t{ent_bits:.2f}\n"
            elif opt_noent:
                sent_buffer += f"{line.strip()}\t{cross_bits:.2f}\n"
            else:
                sent_buffer += f"{line.strip()}\t{cross_bits:.2f}\t{ent_bits:.2f}\n"

            c_sent += 1
            cross_sum += cross_bits
            ent_sum += ent_bits

            prevterms.append(term1)
            if term1 == "SENT":
                prevterms = ["SENT"]

            cache_tokens = tau * cache_tokens + 1
            if term1 in cache_terms:
                cache_terms[term1] = (tau ** (cache_count - last_occ[term1])) * cache_terms[term1] + 1
            else:
                cache_terms[term1] = 1

            last_occ[term1] = cache_count
            cache_count += 1


def compute_entropy(file, outfile, terms=None, ngrams=None, ngramtypes=None, opt_nocross=None, opt_noent=None,
                    cross_name=None, ent_name=None, maxngram=None, tau=None, gamma=None, lmbda=None):
    with open(file, 'r') as inf:
        with open(outfile, 'w') as outf:
            # current document is cached in @lines for second pass in computebits
            lines = []
            for line in inf:
                if line.startswith('<text id='):
                    print(line.split('"')[1])
                    lines = []
                    lines.append(line)
                    docngrams = {}  # freq dict of all ngram tokens for this doc
                    docngramtypes = {}  # counts of ngramtypes in this document, stored in lines variable
                    restngramtypes = {}  # counts of ngramtypes in other documents

                    # insert a sentence marker at the beginning of a document
                    prevterms = []
                    ngrams, prevterms, docngrams, docngramtypes, restngramtypes = addupdoc("SENT", ngrams=ngrams,
                                                                                           ngramtypes=ngramtypes,
                                                                                           prevterms=prevterms,
                                                                                           docngrams=docngrams,
                                                                                           docngramtypes=docngramtypes,
                                                                                           restngramtypes=restngramtypes)
                    prevterms = ["SENT"]
                elif line.startswith('</text'):
                    lines.append(line)

                    docngrams["SENT"] -= 1
                    docngrams["_"] -= 1

                    # compute and write to file cross-entropy and self-surprisal values
                    compute_bits(lines=lines, terms=terms, ngrams=ngrams, docngrams=docngrams,
                                 docngramtypes=docngramtypes, restngramtypes=restngramtypes,
                                 outf=outf, opt_nocross=opt_nocross, opt_noent=opt_noent,
                                 cross_name=cross_name, ent_name=ent_name,
                                 maxngram=maxngram, tau=tau, gamma=gamma, lmbda=lmbda)

                elif line.startswith('<'):
                    lines.append(line)
                else:
                    lines.append(line)
                    fields = line.strip().split("\t")
                    term = fields[0]
                    pos = fields[1]
                    term = term.strip().lower()
                    if pos == "SENT":
                        term = "SENT"
                    elif term in terms and terms[term] == 1:
                        term = "UNK"
                    if len(prevterms) > maxngram:
                        prevterms.pop(0)
                    ngrams, prevterms, docngrams, docngramtypes, restngramtypes = addupdoc(term, ngrams=ngrams,
                                                                                           ngramtypes=ngramtypes,
                                                                                           prevterms=prevterms,
                                                                                           docngrams=docngrams,
                                                                                           docngramtypes=docngramtypes,
                                                                                           restngramtypes=restngramtypes)
                    prevterms.append(term)
                    if pos == "SENT":
                        prevterms = ["SENT"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Annotate a vrt file with bits per word (surprisal) from H(Doc|Rest) and H(Doc)')
    parser.add_argument("infile", help="vrt input file")
    parser.add_argument("outfile", help="vrt output file")
    parser.add_argument("--lmbda", type=float, help="mixing parameter for ngram-prob vs. cache-prob", default=1.0)
    parser.add_argument("--gamma", type=float, help="mixing parameter for Jelinek-Mercer smoothing of cache-prob",
                        default=0.9)
    parser.add_argument("--tau", type=float, help="decay parameter for cache-prob (only 1 or 0.99 usually makes sense)",
                        default=1.0)
    parser.add_argument("--maxngram", type=int, help="context length for ngrams", default=3)
    parser.add_argument("--cross_name", "--cn", type=str, help="name for cross-entropy H(Doc|Rest) in sentence tag",
                        default="srp")
    parser.add_argument("--ent_name", "--en", type=str, help="name for self-entropy H(Doc) in sentence tag",
                        default="srplocal")
    # if none of the flags below are passed, the script outputs both
    parser.add_argument("--nocross", help="pass this flag to annotate cross-entropy rate only", action='store_true')
    parser.add_argument("--noent", help="pass this flag to annotate entropy rate only", action='store_true')

    args = parser.parse_args()

    print("Run settings:", ' '.join(f'--{k} {v}' for k, v in vars(args).items()))

    if args.nocross and args.noent:
        raise Exception("Options --nocross and --noent are mutually exclusive!")

    # first pass: compute term frequencies, to transform terms with only 1 occurrence to unknown.
    my_terms, my_tokens = collect_terms(args.infile)
    # for k,v in my_terms.items():
    #     print(k, v)
    print(f'Tokens: {len(my_tokens)}')
    print(f'Types (terms-hash): {len(my_terms)}')

    # second pass: compute ngram frequencies
    my_ngrams, my_ngramtypes, my_prevterms = collect_ngrams(args.infile, terms=my_terms, maxngram=args.maxngram)
    print(f'Collected ngrams (all orders + _): {len(my_ngrams)}')
    # ngrams_sort = OrderedDict(sorted(my_ngrams.items(), key=itemgetter(1), reverse=True))
    # for (k, v) in list(ngrams_sort.items()):  # [:1000]:  # for bottom 10 [-10:]
    #     # if '\r ' in k:
    #     # if k == '\r ':
    #     print(k, v)
    print(f'Collected uni- and bigram contexts aka ngramtypes: {len(my_ngramtypes)}')
    # ngramtypes_sort = OrderedDict(sorted(my_ngramtypes.items(), key=itemgetter(1), reverse=True))
    # for (k, v) in list(ngramtypes_sort.items()):  # [:-50]
    #     # if '\r ' in k:
    #     # if k == '\r ':
    #     print(k, v)
    with open('temp/cr_ku_ngrams.txt', 'w') as outdict:
        for (k, v) in list(my_ngrams.items()):  # [:-50]
            # if '\r ' in k:
            # if k == '\r ':
            # if "_" in k:
                # print(k.replace("_", "_"))
                # k = k.replace("_", "\n")
            outstring = f'{k}\t{v}'
            print(outstring, file=outdict)

    with open('temp/cr_ku_ngramtypes.txt', 'w') as outdict:
        for (k, v) in list(my_ngramtypes.items()):  # [:-50]
            # if '\r ' in k:
            # if k == '\r ':
            # if "_" in k:
                # print(k.replace("_", "#"))
                # k = k.replace("_", "\n")
            outstring = f'{k}\t{v}'
            print(outstring, file=outdict)

    # # third pass: compute cache frequencies and bits, etc., two passes per document
    # # two passes per document: First compute local ngram frequencies, then compute and output bits.
    compute_entropy(args.infile, args.outfile, terms=my_terms, ngrams=my_ngrams, ngramtypes=my_ngramtypes,
                    opt_nocross=args.nocross, opt_noent=args.noent, cross_name=args.cross_name, ent_name=args.ent_name,
                    maxngram=args.maxngram, tau=args.tau, gamma=args.gamma, lmbda=args.lmbda)
