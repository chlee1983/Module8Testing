# import os
# import cv2
# import glob
# import matplotlib as plt
# import shutil
#
# # Setting of Directories for Image Transferring
# img_dir = 'C:/Users/user/Downloads/Segment2/Segment2/00030762442170551684.jpg'
# final_dir = 'C:/Users/user/Downloads/Segment2/label/2'
# os.mkdir(final_dir, mode = 0o777)
# img = cv2.imread(img_dir)
# os.chdir(final_dir)
# filename = '0.jpg'
# shutil.move(img_dir,final_dir)
# cv2.imshow('image',img)
#
# # Prompt for chinese character
#
# char_input = input("Please enter a Chinese Character: ")
# # if character input matches with a character in the dictionary, move that image file to the designated folder
# # else if character input does not match with any character in the dictionary, add new character to dictionary, create a new directory, then move that image file to the designated folder
#
# # Creation of Dictionary for Chinese Characters
# Dict_Chinese = {}
#
# str = '我'
# print(str)
# Dict_Chinese[str] = 0
#
# str2 = '与'
# print(str2)
# Dict_Chinese[str2] = 1
# print(Dict_Chinese)
#
# list_all = Dict_Chinese.keys()
# print(list_all)

# print('发' in list_all)

"""LOOP VERSION"""

# #import the library opencv
# import cv2
# #globbing utility.
# import glob
# import matplotlib as plt
# import matplotlib.image as mpimg
# #select the path
# path = "C:/Users/user/Downloads/Segment2/Segment2/*.jpg"
# for file in glob.glob(path):
#     print(file)
#     a = cv2.imread(file)
#     print(a)
#     # %%%%%%%%%%%%%%%%%%%%%
#     #conversion numpy array into rgb image to show
#     c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
#     cv2.imshow('Color image', c)
#     #wait for 1 second
#     cv2.waitKey(5000)
#     #destroy the window
#     cv2.destroyAllWindows()

#import the library opencv
import cv2
#globbing utility.
import glob
#matplolib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#utils
import os
import shutil
import random
import json
# Chinese Character Dictionary
# d = {'我': "0", '与': "1", '相': "2", '祖': "3", '田': "4", '九': "5", '十': "6", '二': "7", '团': "8", '.': "9",
#      '不': "13", '是': "14", '一': "15", '问': "16", '丰': "17", '手': "18",
#      '读': "21", '五': '22', '方': '23', '午': '24', '周': '25', '围': '26', '砌': '27', '有': '28', '故': '29', '反': '30',
#      '事': '31',
#      '了': '33', ',': '34', '饭': '35',
#      '吃': '37', '西': '39', '下': '40', '闹': '42', '怕': '43', '芸': '44', '累': '45',
#      'blank': '46', '从': '48', '讨': '52', '百': '53', '己': '54', '挤': '55', '时': '56', '间': '57',
#      '过': '58', '夜': '59', '外': '60', '卖': '61', '耍': '63', '饮': '64', '们': '65',
#      '至': '66', '男': '67', '父': '69', '这': '70', '伯': '71', '法': '72', '但': '76',
#      '见': '77', '访': '78', '妈': '79', '分': '81', '互': '82', '区': '83', '防': '85',
#      '饿': '87', '书': '88', '装': '89', '亇': '91', '白': '93', '茫': '94',
#      '亮': '95', '又': '96', '力': '97', '荤': '98', '干': '100', '造': '101',
#      '好': '102'}
#9, s, s, s, 13, ...，17, 18, s, s,
#21, 22, 23, 24,s,...
#35, s, 37,s,39,40,s, ..., 47, 48, s
#s, s, 52..., 61, s, 62...,67, s,68...,72, s,s,s, 76...79,s
#80...83,s,85,s,87,88,89,s,91,s,93...,98,s,100

#Chinese Dictionary
with open('chinese_dictionary.json') as json_file:
    d = json.load(json_file)

character_list = d.keys()

#select the path
path = "C:/Users/user/Downloads/Segment2/Segment2/45767165028323741714.jpg"

for file in glob.glob(path):
    print(file)
    a = mpimg.imread(file)
    print(a)
    # %%%%%%%%%%%%%%%%%%%%%
    #conversion numpy array into rgb image to show
    c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.imshow(c)
    plt.show()
    #Chinese Character Input + Feedback
    char_input = input("Please enter a Chinese Character: ")
    if char_input in character_list:
        print('Yes')
        print('file',file)
        final_dir = 'C:/Users/user/Downloads/Segment2/label/'+str(d[char_input])+'/'
        # filename = str(random.randint(0,9))+ str(random.randint(0,9))+ str(random.randint(0,9))+ str(random.randint(0,9))+ str(random.randint(0,9))+'.jpg'
        # final_dir = final_dir+filename
        shutil.move(path,final_dir)
    else:
        print('No')
        key_input = input("Please enter the key number： ")
        final_dir = 'C:/Users/user/Downloads/Segment2/label/'+str(key_input)+'/'
        # filename = str(random.randint(0,9))+ str(random.randint(0,9))+ str(random.randint(0,9))+ str(random.randint(0,9))+ str(random.randint(0,9))+'.jpg'
        os.mkdir(final_dir, mode=0o777)
        os.chdir(final_dir)
        shutil.move(path,final_dir)
        d[char_input] = [key_input]
    print(d)
with open("C:/Users/user/PycharmProjects/ChineseCharacterClassification/chinese_dictionary.json", "w") as outfile:
    json.dump(d, outfile)
json_object = json.dumps(d, indent=4)

# Dict_Chinese = {}
# str = '我'
# # print(str)
# Dict_Chinese[str] = 0
# str2 = '与'
# # print(str2)
# Dict_Chinese[str2] = 1
# print(Dict_Chinese)
