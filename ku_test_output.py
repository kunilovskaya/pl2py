"""
5 Feb 2023
kunilovskaya
compare surprisal values from ku_genzelcharniak-vrt.py and from genzelcharniak-vrt_v0.2.1.pl

Run instructions:
python3 ku_test_output.py

"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_vals(file, val):
    lines = open(file, 'r').readlines()
    # skip lines starting with <
    all_res = []
    for line in lines:
        if line.startswith("<"):
            continue
        else:
            fields = line.strip().split("\t")
            try:
                if val == 'cross':
                    col = 3
                else:
                    col = 4
                res = fields[col].strip()
                all_res.append(float(res))
            except IndexError:
                print(file)
                print(fields)

    return all_res, lines


def my_rmse(trues, predictions):
    return mean_squared_error(trues, predictions, squared=False)


if __name__ == "__main__":
    pl = 'output/truest_BROWN-SPR.vrt'
    py = 'output/ku_BROWN-SPR.vrt'

    # collect values in 4th (cross) and 5th (self) tab-separated columns
    true_cross_ent, pl_lines = get_vals(pl, "cross")
    test_cross_ent, py_lines = get_vals(py, "cross")

    # *** 1 ***
    print(len(true_cross_ent), len(test_cross_ent))

    # *** 2 ***
    pearson, p_val = pearsonr(true_cross_ent, test_cross_ent)
    print('%.3f, p < %.3f' % (pearson, p_val))

    true_self_ent = get_vals(pl, "self")
    test_self_ent = get_vals(py, "self")

    try:
        pearson, p_val = pearsonr(np.asarray(true_self_ent), np.asarray(test_self_ent))
        print('%.3f, p < %.3f' % (pearson, p_val))
    except AttributeError:
        print('I am not sure what happened to the format')

    # *** 3 ***
    error = my_rmse(true_cross_ent, test_cross_ent)
    print('%.3f' % (error))

    # *** 4 ***
    my_mismatch = set(pl_lines).difference(set(py_lines))
    counter = 0
    for i in my_mismatch:
        print(i.strip().replace("\t", " "))
        counter += 1
    print(f"Number of lines with mismatching content: {counter} ({counter/len(true_cross_ent):.2f}%)")





