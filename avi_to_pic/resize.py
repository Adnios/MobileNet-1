import os
from PIL import Image

file_dir = '/mnt/d/temp/MobileNet/avi_to_pic/pic/'
file_dir2 = '/mnt/d/temp/MobileNet/avi_to_pic/pic2/'
file_dir_list = os.listdir(file_dir)
for sub_dir in file_dir_list:
    sub_file = file_dir + sub_dir
    sub_file2 = file_dir2 + sub_dir
    print(sub_file)
    sub_dir_list = os.listdir(sub_file)
    for item in sub_dir_list:
        pic = sub_file + '/' + item
        pic2 = sub_file2 + '/' + item
        print(pic)
        im = Image.open(pic)
        im = im.resize((32,32), Image.ANTIALIAS)
        im.save(pic2)
