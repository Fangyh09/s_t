import argparse
import os

BIO_MODE = "bio"
BIEO_MODE = "bieo"
BIO_TAGS = ["I-test", "I-problem", "I-treatment", "O", "B-treatment",
            "B-problem",
        "B-test"]

BIEO_TAGS= ["E-problem", "E-treatment", "I-test", "B-treatment", "B-problem",
              "I-treatment", "B-test", "O", "I-problem", "E-test"]


# def check_str(line, mode):
#     """
#     :param line: after remove
#     :return:
#     """
#     ls = line.split(' ')
#     word, tag = ls[0], ls[-1]
#     if mode == BIO_MODE:
#         return tag in BIO_TAGS
#     elif mode == BIEO_MODE:
#         return tag in BIEO_TAGS
#     else:
#         raise ValueError(mode + "is illegal")


def main(fname):
    # fname = "../dev.eval"
    if not os.path.exists(fname):
        raise ValueError(fname + "not exist")
    fout_name = fname + ".reverse"
    fout = open(fout_name, "w+")

    with open(fname) as f:
        arr = []
        state = 0
        for line in f:
            line = line.strip()
            if state == 0:
                if len(line) == 0:
                    return False
                else:
                    arr.append(line)
                    state = 1
            elif state == 1:
                if len(line) == 0:
                    arr_len = len(arr)
                    for item in reversed(arr):
                        fout.write(item + "\n")
                    fout.write("\n")
                    arr = []
                    state = 1
                else:
                    arr.append(line)
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
        fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reverser the file')
    parser.add_argument('--path', type=str, default="")

    args = parser.parse_args()
    path = "./dev.eval.bieo"
    main(args.path)
    # main(path)
    # res = main(path, "bieo")

    # a = "B-problem"
    # print(a[1:])

