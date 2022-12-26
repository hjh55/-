f3 = open("test1.txt","r",encoding="utf-8") # 未去重的文件
f4 = open("word_dict_name.txt","a",encoding="utf-8")  # 去重的文件
data1 = set()
for a in [a.strip('\n') for a in list(f3)]:
    if a not in data1:
        f4.write(a+'\n')
        data1.add(a)
f3.close()
f4.close()