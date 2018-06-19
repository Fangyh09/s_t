import argparse
import os


def main(fname):
    # fname = "../dev.eval"
    if not os.path.exists(fname):
        raise ValueError(fname + "not exist")

    fout_name = fname + ".reverse"
    fout = open(fout_name, "w+")
    with open(fname) as f:
        words, tags = [], []
        pre = []
        for line in f:
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                fout.write(line + "\n")
                # print(line)
                pre = []
                if len(words) != 0:
                    pass
            else:
                # pass
                ls = line.split(' ')
                word, tag = ls[0], ls[-1]
                # print("pre", pre)
                if len(pre) > 1 and pre[1] == '-' and tag == "O":
                    newtag = "E" + pre[1:]
                    fout.write(word + " " + newtag + "\n")
                    # print(word + " " + newtag)
                else:
                    fout.write(word + " " + tag + "\n")
                    # print(word + " " + tag)
                pre = tag
                # print(word, tag)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reverser the file')
    parser.add_argument('--path', type=str, default="")

    args = parser.parse_args()
    main(args.path)
    # a = "B-problem"
    # print(a[1:])

