import cv2
from PIL import ImageFont,ImageDraw,Image
import numpy as np
import uuid
import os
import random
import math
#生僻字库
with open("list3.txt",encoding='utf-8') as f:
    word = f.readlines()

word_dict_length = len(word)#字库字数
#fontPath = "PingFang Heavy.ttf"#黑体
fontPath = '/test4/test1124/PingFang Heavy.ttf'
'''
#save_dir = "/test4/test1124/train_data_newfortest/train" #生成训练集
save_dir = "/test4/test1124/train_data_newfortest/test"#生成验证集
'''
#save_dir = "/test4/test1124/train_data_5000/train"
save_dir = "/test4/test1124/train_data_5000/test"
#标签字符限制
min_words_length = 2
max_words_length = 25
img_count = 1250 #图像数量
#绘制图像 
def draw_to_image(size,fontPath,font_size,text,save_dir,file_name):
    background = np.ones(size)*255
    #读取字体，并确定字体大小
    font = ImageFont.truetype(fontPath,font_size)
    #将数组转化为图像
    img = Image.fromarray(background)
    #创建绘制的对象
    draw = ImageDraw.Draw(img)
    #绘制文本，指定文本绘制的起点(左上角，(1,1))，文本内容，字体格式，字体颜色
    draw.text((1,1),text,font=font,fill='black')
    img = np.array(img)
    #图像保存路径
    out_path = os.path.join(save_dir,file_name)
    #保存绘制的图像
    cv2.imwrite(out_path,img)
    
    
def generate_words(word,count,length_range_min,length_range_max):
    line_words_list = []
    line_words = ""
    globe_word_dict_index = 0
    ##开始生成图像标签
    for k in range(count+1):#生成前1000张数据
        #随机生成标签字符的长度
        bit_length = np.random.randint(length_range_min,length_range_max)
        for i in range(bit_length+1):
            try:
                line_words = line_words + word[globe_word_dict_index].split("\n")[0]  #消除换行符'\n'
            #列表溢出时
            except:
                globe_word_dict_index = 0
                line_words = line_words + word[globe_word_dict_index].split("\n")[0]
            globe_word_dict_index+=1
        #依次储存图像标签
        line_words_list.append(line_words)
        line_words = ""
    # print(line_words_list)

    return line_words_list
line_words_list = generate_words(word,img_count,min_words_length,max_words_length)

# 对图像做数据增强
# 图像水平翻转
def horizon_flip(img):
    '''
    图像水平翻转
    :param img:
    :return:水平翻转后的图像
    '''
    return img[:, ::-1]
# 图像垂直翻转
def vertical_flip(img):
    '''
    图像垂直翻转
    :param img:
    :return:
    '''
    return img[::-1]
# 旋转图像
def rotate(img, limit_up=10, limit_down=-10):
    '''
    在一定角度范围内，图像随机旋转
    :param img:
    :param limit_up:旋转角度上限
    :param limit_down: 旋转角度下限
    :return: 旋转后的图像
    '''
    # 旋转矩阵
    rows, cols = img.shape[:2]
    center_coordinate = (int(cols / 2), int(rows / 2))
    angle = random.uniform(limit_down, limit_up)
    M = cv2.getRotationMatrix2D(center_coordinate, angle, 1)
    # 仿射变换
    out_size = (cols, rows)
    rotate_img = cv2.warpAffine(img, M, out_size, borderMode=cv2.BORDER_REPLICATE)
    return rotate_img

# 平移图像
def shift(img, distance_down, distance_up):
    '''
    利用仿射变换实现图像平移，平移距离∈[down, up]
    :param img: 原图
    :param distance_down:移动距离下限
    :param distance_up: 移动距离上限
    :return: 平移后的图像
    '''
    rows, cols = img.shape[:2]
    y_shift = random.uniform(distance_down, distance_up)
    x_shift = random.uniform(distance_down, distance_up)
    # 生成平移矩阵
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    # 平移
    img_shift = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    return img_shift

# 裁剪图像
def crop(img, crop_x, crop_y):
    '''
    读取部分图像，进行裁剪
    :param img:
    :param crop_x:裁剪x尺寸
    :param crop_y:裁剪y尺寸
    :return:
    '''
    rows, cols = img.shape[:2]
    # 偏移像素点
    x_offset = random.randint(0, cols - crop_x)
    y_offset = random.randint(0, rows - crop_y)
    # 读取部分图像
    img_part = img[y_offset:(y_offset+crop_y), x_offset:(x_offset+crop_x)]
    return img_part

# 对比度、亮度调整
def lighting_adjust(img, k_down, k_up, b_down, b_up):
    '''
    图像亮度、对比度调整
    :param img:
    :param k_down:对比度系数下限
    :param k_up:对比度系数上限
    :param b_down:亮度增值上限
    :param b_up:亮度增值下限
    :return:调整后的图像
    '''
    # 对比度调整系数
    slope = random.uniform(k_down, k_up)
    # 亮度调整系数
    bias = random.uniform(b_down, b_up)
    # 图像亮度和对比度调整
    img = img * slope + bias
    # 灰度值截断，防止超出255
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# 给图片加高斯噪声
def Gaussian_noise(img, mean=0, std=1):
    '''
    图像加高斯噪声
    :param img: 原图
    :param mean: 均值
    :param std: 标准差
    :return:
    '''
    # 高斯噪声图像
    gauss = np.random.normal(loc=mean, scale=std, size=img.shape)
    img_gauss = img + gauss
    # 裁剪
    out = np.clip(img_gauss, 0, 255)
    return out

# 归一化
def normalization(img, mean, std):
    '''
    图像归一化,图像像素点从(0,255)->(0,1)
    :param img:
    :param mean:所有样本图像均值
    :param std: 所有样本图像标准差
    :return:
    '''
    img -= mean
    img /= std
    return img
    
for image_name , line in enumerate(line_words_list):
    #length = len(line)
    length = 25
    #随即设置书写字符的字体大小
    #font_size = 33 if np.random.randint(0,2)==0 else 20
    font_size = 20
    file_name = str((image_name+1))+".jpg"
    size = (32,math.ceil(length*11))
    draw_to_image(size,fontPath,font_size,line,save_dir,file_name)
    # if(font_size==33):
    #     size = (42,math.ceil(length*34.4))
    #     draw_to_image(size,fontPath,font_size,line,save_dir,file_name)
    # else:
    #     size = (29,math.ceil(length*20.8))
    #     draw_to_image(size,fontPath,font_size,line,save_dir,file_name)

   #train_data_newfortest
    '''
    with open("/test4/test1124/train_data_newfortest/rec_gt_train.txt","a") as f:
         f.write("/test4/test1124/train_data_newfortest/train/"+file_name+"\t"+line+"\n")
    with open("/test4/test1124/train_data_newfortest/rec_gt_test.txt","a") as f:
        f.write("/test4/test1124/train_data_newfortest/test/"+file_name+"\t"+line+"\n")

    #train_data(5000)

    with open("/test4/test1124/train_data_5000/train.txt","a") as f:
         f.write("/test4/test1124/train_data_5000/train/"+file_name+"\t"+line+"\n")
    '''    
    #生成rec_gt_test.txt
    with open("/test4/test1124/train_data_5000/test.txt","a") as f:
        f.write("/test4/test1124/train_data_5000/test/"+file_name+"\t"+line+"\n")