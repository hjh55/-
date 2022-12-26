train_list_path = "/root/test1110/train_data/rec_data/train"
label_path = "/root/test1110/train_data/rec_data/rec_gt_train.txt"
with open(train_list_path, 'r') as fr:
    with open(label_path, 'w', encoding="utf-8") as fw:
        lines = fr.readlines()
        for line in lines:
            line = line.split("\t")
            fw.writelines(line[-2] + "\t" + line[-1])
