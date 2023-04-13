## Aim
Generate per-word surprisal values for (Stanza)-parsed corpora (mainly EPIC-UdS)
* in python, by re-producing genzelcharniak-vrt_v0.2.1.pl
* based on lemmas, rather than tokens
* from other types of LM, including (pretrained) multilingual

## Backround
LMs are models that assign probabilities to words and their sequencies.
Probability can be calculated with regard probability distribution in the given doc or with regard to a corpus (excluding current document).
Surprisal is -log2(prob), i.e. inverse probability measured in bits of information.

## Implementation in genzelcharniak-vrt_v0.2.1.pl
* calculates log2 probability of tokens based on 3 preceding words (according to Juravsky, this is a 4gram LM)
* no ngrams across sentence boundaries are generated
* hapax are replaced with UNK
* constant parameters (weights) are 1.0 (they don't affect the calculations), except gamma=0.99
* PROBABILITY is a sum of (1) products between freq_of_4gram_over_freq_of_3gram_left_context(aka MLE) and 
  relative freq of its 3gram preceding context (lambda1, "how unusual is the context in the corpus") + (2)  
  and inverse-lambda1_weighted sum of probabilities for component ngrams in the corpus (excluding current doc)
* CROSS- and SELF-: probability is estimated with regard to the whole corpus (with the exception of the current document) and based on the current document only)
* there is a function that estimates "cache decay" (position of ngram in the doc), employing gamma, but it seems to have no affect on the main calculations, given default parameters
* LM backs off to lower-order ngrams to estimate probability of ngrams missing from current doc or the rest of the corpus
* I don't see calculations of entropy measures, Shannon or relative (aka Kullback-Leibler divergence or cross-entropy), 
  if they are defined as H = -sum(pk * log(pk)) or CE = -sum(pk * log(qk))

10 Feb 2023