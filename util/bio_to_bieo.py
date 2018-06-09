import argparse
import os

def main(fname):
    # fname = "../dev.eval"
    if not os.path.exists(fname):
        raise ValueError(fname + "not exist")

    fout_name = fname + ".bieo"
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

    # with open(fname) as f:
    #     content = f.readlines()
    # content = [x.strip() for x in content]
    #
    # fout = open(fout_name, "w+")
    # pre = []
    # for x in content:
    #     # rm '/n'
    #     ls = x.split(' ')
    #     word = x.strip().split(' ')[0]
    #     if len(ls) > 0:
    #         print(ls)
    #     print(ls)
    #     if len(ls) > 0 and ls[1] == "O":
    #         print("ls")
    #         print(ls)
    #         if len(pre) == 2 and len(pre[1]) > 1 and pre[1][1] == '-':
    #             newtag = "E" + pre[1][1:]
    #             # fout.write(ls[0] + " " + newtag)
    #             print(ls[0] + " " + newtag)
    #     else:
    #         # pass
    #         # fout.write(ls[0] + " " + ls[1] + "\n")
    #         print(ls[0] + " " + ls[1] + "\n")
    #     pre = ls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BIO to BIEO')
    parser.add_argument('--path', type=str, default="")

    args = parser.parse_args()
    main(args.path)
    # a = "B-problem"
    # print(a[1:])

