import argparse
import os

BIO_MODE = "bio"
BIEO_MODE = "bieo"
BIO_TAGS = ["I-test", "I-problem", "I-treatment", "O", "B-treatment",
            "B-problem",
        "B-test"]

BIEO_TAGS= ["E-problem", "E-treatment", "I-test", "B-treatment", "B-problem",
              "I-treatment", "B-test", "O", "I-problem", "E-test"]


def check_str(line, mode):
    """
    :param line: after remove
    :return:
    """
    line = line.strip()
    ls = line.split(' ')
    word, tag = ls[0], ls[-1]
    if mode == BIO_MODE:
        return tag in BIO_TAGS
    elif mode == BIEO_MODE:
        return tag in BIEO_TAGS
    else:
        raise ValueError(mode + "is illegal")


def main(fname, mode):
    # fname = "../dev.eval"
    if not os.path.exists(fname):
        raise ValueError(fname + "not exist")

    with open(fname) as f:
        state = 0
        for line in f:
            line = line.strip('\n')

            if state == 0:
                if len(line) == 0:
                    return False
                else:
                    if not check_str(line, mode):
                        return False
                    else:
                        state = 1
            elif state == 1:
                if len(line) == 0:
                    state = 2
                else:
                    if not check_str(line, mode):
                        return False
                    else:
                        pass
            else:
                if len(line) == 0:
                    return False
                else:
                    if not check_str(line, mode):
                        return False
                    else:
                        state = 1

        return state == 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reverser the file')
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--mode', type=str, default="") #bio, bieo

    args = parser.parse_args()
    path = "./dev.eval.bieo"
    res = main(args.path, args.mode)
    # res = main(path, "bieo")
    if res:
        print("Con! check passed")
    else:
        print("Sorry, check failed...")
    # a = "B-problem"
    # print(a[1:])

