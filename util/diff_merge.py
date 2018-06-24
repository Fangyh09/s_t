
def get_patch():
    # diff_name = "./diff.txt"
    # # third_party = "./result_elmo87_14.submit"
    # third_party = "/Users/yinghong/fsdownload/config9-test.predict" \
    #               ".rmprob.2col.submit"
    # targe_name = "./config-ensem-1-test.predict.rmprob.2col.submit"
    # fout_name = "google-ensem-1.submit"

    diff_name = "./diff-dev.txt"
    # third_party = "./result_elmo87_14.submit"
    third_party = "./config9-dev.predict.rmprob"
    targe_name = "./config-ensem-1-dev.predict.rmprob"
    fout_name = "google-ensem-dev-1"
    fout = open(fout_name, "w")
    with open(diff_name) as f:
        diff_content = f.readlines()
    diff_content = [x.strip() for x in diff_content]

    with open(third_party) as f:
        pred_content = f.readlines()
    pred_content = [x.strip() for x in pred_content]

    with open(targe_name) as f:
        targe_content = f.readlines()
    targe_content = [x.strip() for x in targe_content]

    replace_map = {}

    diff_len = len(diff_content)

    idx = 0
    while idx < diff_len:
        line = diff_content[idx]
        idx += 1
        com_pos = line.find(",")
        if com_pos > 0:
            c_pos = line.find("c")
            line_beg = int(line[: com_pos]) - 1
            line_end = int(line[com_pos + 1: c_pos]) - 1
            num_line = line_end - line_beg + 1
            c1 = 0
            c2 = 0
            # count for c1
            offset = 0
            while offset < num_line:
                cur_line = diff_content[idx][2:]
                idx += 1
                if cur_line == pred_content[line_beg + offset]:
                    c1 += 1
                else:
                    pass
                offset += 1

            # count for c2
            idx += 1
            offset = 0
            while offset < num_line:
                cur_line = diff_content[idx][2:]
                idx += 1
                if cur_line == pred_content[line_beg + offset]:
                    c2 += 1
                else:
                    pass
                offset += 1
            if c2 > c1:
                prev_idx = idx - num_line
                arr = []
                for i in range(0, num_line):
                    arr.append(diff_content[prev_idx + i][2:])
                    targe_content[line_beg + i] = diff_content[prev_idx + i][2:]
                replace_map[line] = arr

        else:
            c_pos = line.find("c")
            line_no = int(line[:c_pos]) - 1
            poss_a = diff_content[idx]
            # dummpy to remove ---
            idx += 1
            idx += 1
            poss_b = diff_content[idx][2:]
            idx += 1
            poss_pred = pred_content[line_no][2:]
            if poss_pred == poss_a or poss_pred == poss_b:
                replace_map[line] = [poss_pred]
                # update
                targe_content[line_no] = poss_pred
            else:
                pass
    for item in targe_content:
        fout.write(item)
        # if len(item) > 0:
        fout.write("\n")
    return replace_map


if __name__ == "__main__":
    a = get_patch()
    print(a)



