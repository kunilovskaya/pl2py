"""
5 Feb 2023
kunilovskaya
compare surprisal values from ku_genzelcharniak-vrt.py and from genzelcharniak-vrt_v0.2.1.pl

Run instructions:
python3 ku_test_output.py

"""

from scipy.stats import pearsonr, spearmanr


def get_vals(file, val):
    lines = open(file, 'r').readlines()
    # skip lines starting with <
    all_res = []
    for line in lines:
        if line.startswith("<"):
            continue
        else:
            fields = line.strip().split("\t")
            if val == 'cross':
                col = 3
            else:
                col = 4
            res = fields[col].strip()
            all_res.append(float(res))

    return all_res


if __name__ == "__main__":
    pl = 'output/true_BROWN-SPR.vrt'
    py = 'output/ku_BROWN-SPR.vrt'

    # collect values in 4th (cross) and 5th (self) tab-separated columns
    true_cross_ent = get_vals(pl, "cross")
    test_cross_ent = get_vals(pl, "cross")

    pearson, p_val = pearsonr(true_cross_ent, test_cross_ent)
    print('%.3f, p < %.3f' % (pearson, p_val))

    true_self_ent = get_vals(pl, "self")
    test_self_ent = get_vals(pl, "self")

    pearson, p_val = pearsonr(true_self_ent, test_self_ent)
    print('%.3f, p < %.3f' % (pearson, p_val))





