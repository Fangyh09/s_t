import argparse
import os

# only support bieo

# PROBLEM = ["E-problem", "B-problem", "I-problem"]
# TREATMENT = ["E-treatment", "B-treatment", "I-treatment"]
# TEST = ["I-test", "B-test", "E-test"]

PROBLEM = ["B-problem"]
TREATMENT = ["B-treatment"]
TEST = ["B-test"]


def main(fname):
    # fname = "../dev.eval"
    if not os.path.exists(fname):
        raise ValueError(fname + "not exist")

    with open(fname) as f:
        state = 0
        num_sentence = 0
        num_problem = 0
        num_test = 0
        num_treatment = 0

        has_prob = 0
        has_test = 0
        has_treat = 0
        for line in f:
            line = line.strip()

            if state == 0:
                if len(line) == 0:
                    return False
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]
                    if tag in PROBLEM:
                        has_prob += 1
                    elif tag in TEST:
                        has_test += 1
                    elif tag in TREATMENT:
                        has_treat += 1

                    state = 1
            elif state == 1:
                if len(line) == 0:
                    num_sentence += 1
                    num_problem += has_prob
                    num_test += has_test
                    num_treatment += has_treat

                    has_prob = 0
                    has_test = 0
                    has_treat = 0
                    state = 2
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]
                    if tag in PROBLEM:
                        has_prob += 1
                    elif tag in TEST:
                        has_test += 1
                    elif tag in TREATMENT:
                        has_treat += 1
            else:
                if len(line) == 0:
                    return False
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]
                    if tag in PROBLEM:
                        has_prob += 1
                    elif tag in TEST:
                        has_test += 1
                    elif tag in TREATMENT:
                        has_treat += 1
                    state = 1
        print("#sentences is %d,\n #problem is %d, #test is %d, #treament is %d"
              % (num_sentence, num_problem, num_test, num_treatment))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reverser the file')
    parser.add_argument('--path', type=str, default="")

    args = parser.parse_args()
    # path = "./dev.eval.bieo"
    main(args.path)
    # main(path)
