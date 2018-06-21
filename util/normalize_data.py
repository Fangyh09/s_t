import argparse
import os

import numpy as np


def normalize_score(scores):
    scores = np.array(scores)
    scores *= 1.0
    scores /= np.max(scores)
    print(scores)


def write_to_file(sent, fout):
    for word in sent:
        fout.write(word + "\n")
    fout.write("\n")


def merge_file(f1_name, f2_name, fout_Name):
    article1, scores1 = parse_file(f1_name)
    article2, scores2 = parse_file(f2_name)

    scores1 = normalize_score(scores1)
    scores2 = normalize_score(scores2)

    len = len(article1)
    fout = open(fout_Name, "w")
    assert len(article1) == len(article2)
    for i in xrange(len):
        if scores1 > scores2:
            write_to_file(article1[i])
        else:
            write_to_file(article2[i])
    fout.close()


def parse_file(fname):
    # fname = "../dev.eval"
    if not os.path.exists(fname):
        raise ValueError(fname + "not exist")
    # fout_name = fname + ".reverse"
    # fout = open(fout_name, "w+")

    with open(fname) as f:
        arr = []
        scores = []
        article = []
        state = 0
        for line in f:
            line = line.strip()
            if state == 0:
                if len(line) == 0 or line[:9] == "Prob is :":
                    return False
                else:
                    arr.append(line)
                    state = 1
            elif state == 1:
                if len(line) == 0 or line[:9] == "Prob is :":
                    # print(line)
                    # arr_len = len(arr)
                    # for item in reversed(arr):
                    #     fout.write(item + "\n")
                    # fout.write("\n")
                    article.append(arr)
                    str_tmp = line[9:]
                    scores.append(float(str_tmp))
                    arr = []
                else:
                    arr.append(line)
        # (normalize_score(scores))
        return article, scores
        # else:
        #     if len(line) == 0:
        #         return False
        #     else:
        #         arr_len = len(arr)
        #         for item in reversed(arr):
        #             fout.write(item + "\n")
        #         fout.write("\n")
        #         arr = []
        #         state = 1
        # fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reverser the file')
    parser.add_argument('--p1', type=str, default="")
    parser.add_argument('--p2', type=str, default="")
    out = "score-merge.txt"
    args = parser.parse_args()
    merge_file(args.p1, args.p2, out)
