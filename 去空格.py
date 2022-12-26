f1 = open("list5.txt","r",encoding='utf-8')  # 未去空行的文件
f2 = open("test2.txt","w",encoding='utf-8') # 储存去空行的文件
for line in f1.readlines():
    line1 = line.replace("\n","")
    if line1 == "":
        continue
    else:
        f2.write(line)
f1.close()
f2.close()