import argparse
import os

import numpy as np

# only support bieo

# PROBLEM = ["E-problem", "B-problem", "I-problem"]
# TREATMENT = ["E-treatment", "B-treatment", "I-treatment"]
# TEST = ["I-test", "B-test", "E-test"]

PROBLEM = ["B-problem"]
TREATMENT = ["B-treatment"]
TEST = ["B-test"]

PROBLEM_RATIO = 0.45
TREATMENT_RATIO = 0.33
TEST_RATIO = 0.33


def cross_entropy(p_prob, p_treat, p_test):
    # return -PROBLEM_RATIO * np.log(1e-5 + p_prob) -TREATMENT_RATIO * np.log(
    #     1e-5 + p_treat) -TEST_RATIO * np.log(1e-5 + p_test)
    return abs(p_prob - 0.428) + abs(p_test - 0.28) + abs(0.296 - p_treat)


def analyse_paragraph(part_article):
    state = 0
    num_sentence = 0
    num_problem = 0
    num_test = 0
    num_treatment = 0

    has_prob = 0
    has_test = 0
    has_treat = 0
    for paragraph in part_article:
        for line in paragraph:
            line = line.strip()

            ls = line.split(' ')
            word, tag = ls[0], ls[-1]
            if tag in PROBLEM:
                num_problem += 1
            elif tag in TEST:
                num_test += 1
            elif tag in TREATMENT:
                num_treatment += 1
        num_sentence += 1

    # print("#sentences is %d,\n #problem is %d, #test is %d, #treament is %d"
    #       % (num_sentence, num_problem, num_test, num_treatment))
    return num_sentence, num_problem, num_test, num_treatment


def main(fname):
    # fname = "./ids.save"
    if not os.path.exists(fname):
        raise ValueError(fname + "not exist")
    with open(fname) as f:
        article = []
        paragraph = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if len(paragraph) > 0:
                    article.append(paragraph)
                    paragraph = []
                else:
                    pass
            else:
                paragraph.append(line)
        print("#paragraph %d" % len(article))
        num_paragraph = len(article)
        num_p1 = 1600
        min_entropy = 1000000000
        best_choose = []
        article = np.array(article)
        # np.random.seed(np.random.randint(19235645))

        for idx in xrange(1000):
            choose_id = np.random.choice(num_paragraph, num_p1)
            part1 = article[choose_id]
            num_sentence, num_problem, num_test, num_treatment = \
                analyse_paragraph(part1)
            p_prob = num_problem * 1.0 / num_sentence
            p_test = num_test * 1.0 / num_sentence
            p_treat = num_treatment * 1.0 / num_sentence
            entropy = cross_entropy(p_prob, p_treat, p_test)
            if entropy < min_entropy:
                min_entropy = entropy
                best_choose = choose_id
                print("===> new entropy %f" % entropy)
                print("#problem is %f, #test is %f, #treament is %f"
                      % (p_prob, p_test, p_treat))

        choose_id = np.array(choose_id)
        np.savez("ids.save", choose_id)


def write_part_to_file(article, name):
    with open(name, "w") as f:
        for para in article:
            for line in para:
                f.write(line + "\n")
            f.write("\n")


def split(fname, npz_path):
    choose_ids = np.load(npz_path)["arr_0"]
    if not os.path.exists(fname):
        raise ValueError(fname + "not exist")
    with open(fname) as f:
        article = []
        paragraph = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if len(paragraph) > 0:
                    article.append(paragraph)
                    paragraph = []
                else:
                    pass
            else:
                paragraph.append(line)
        print("#paragraph %d" % len(article))
        idx = 0
        part1 = []
        part2 = []
        for para in article:
            if idx in choose_ids:
                part1.append(para)
            else:
                part2.append(para)
            idx += 1
    write_part_to_file(part1, "part1.eval")
    write_part_to_file(part2, "part2.eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reverser the file')
    parser.add_argument('--path', type=str, default="")
    args = parser.parse_args()
    # path = "./dev.eval.bieo"
    # analyse_paragraph(args.path)

    # main(args.path)
    # split("./data/train_dev.txt", "./data/0.428_0.281_0.295ids.save.npz")
    # main(path)
