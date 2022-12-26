f1 = open("word_dict_name.txt",'r',encoding='utf-8')
ftext1 = f1.read()
len1 = len(ftext1)
f2 = open("ppocr_keys_v1.txt","r",encoding='utf-8')
ftext2 = f2.read()
len2 = len(ftext2)
f3 = open('list5.txt','w',encoding='utf-8')
str_temp=[]
for i in range (len1):
    if ftext1[i] in ftext2:
        str_temp.append(ftext1[i])
        f3.write(ftext1[i]+'\n')
print(len(str_temp))