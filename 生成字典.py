# 字符字典
with open('excel2txt.txt','r',encoding='utf-8') as f:
    data= f.read()
for i in data:
    with open("test.txt","a",encoding='utf-8') as fi:
        fi.write(i+"\n")
# 去空行
f1 = open("test.txt","r",encoding='utf-8')  # 未去空行的文件
f2 = open("test1.txt","w",encoding='utf-8') # 储存去空行的文件
for line in f1.readlines():
    line1 = line.replace("\n","")
    if line1 == "":
        continue
    else:
        f2.write(line)
f1.close()
f2.close()
# 去重
f3 = open("test1.txt","r",encoding="utf-8") # 未去重的文件
f4 = open("{word_dict_name}.txt","a",encoding="utf-8")  # 去重的文件
data1 = set()
for a in [a.strip('\n') for a in list(f3)]:
    if a not in data1:
        f4.write(a+'\n')
        data1.add(a)
f3.close()
f4.close()