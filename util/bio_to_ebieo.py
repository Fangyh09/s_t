import argparse
import os

DEBUG_MODE = False


class Writer:
    def __init__(self, fout):
        self.fout = fout

    def write(self, str):
        if not DEBUG_MODE:
            self.fout.write(str)
        else:
            print(str)


def main(fname):
    if DEBUG_MODE:
        fname = "../dev.eval"
    fout_name = fname + ".ebieo"
    if not os.path.exists(fname):
        raise ValueError(fname + "not exist")

    with open(fname) as f:
        content = f.readlines()
    # content = [x.strip() for x in content]

    fout = open(fout_name, "w+")
    writer = Writer(fout)

    words, tag = [], []
    pre = []
    next = []
    size = len(content)
    # print("length" + str(length))
    idx = 0
    for line in content:

        line = line.strip()
        if (len(line) == 0 or line.startswith("-DOCSTART-")):
            writer.write(line + "\n")
            pre = []
            if len(words) != 0:
                pass
        else:
            # pass
            ls = line.split(' ')
            word, tag = ls[0], ls[-1]


            # print("pre", pre)
            if tag == "O" and len(pre) > 1 and pre[1] == '-':
                newtag = "E" + pre[1:]
                writer.write(word + " " + newtag + "\n")
            elif tag == "O" and idx + 1 < size and len(content[idx + 1].strip()) > 0 \
                and content[idx + 1].strip().split(' ')[1][0] == 'B':
                newtag = "E" + content[idx + 1].strip().split(' ')[1][1:]
                writer.write(word + " " + newtag + "\n")
            else:
                writer.write(word + " " + tag + "\n")

            pre = tag
            # print(word, tag)
        idx += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BIO to BIEO')
    parser.add_argument('--path', type=str, default="")

    args = parser.parse_args()
    main(args.path)
    # a = "B-problem"
    # print(a[1:])

