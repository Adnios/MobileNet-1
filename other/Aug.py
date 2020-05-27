'''
reference : 
https://zhuanlan.zhihu.com/p/30197320
https://blog.csdn.net/qq_41084756/article/details/84454192
数据增强
'''

import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=44,
    rescale=1. / 255,
    width_shift_range=0.4,
    height_shift_range=0.8,
    shear_range=0.7,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')
for file_name in os.listdir('/media/scrutiny/Data/temp/Canny_pic/test/test/Paper/'):
    img = load_img('/media/scrutiny/Data/temp/Canny_pic/test/test/Paper/' + file_name)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 1
    print(file_name)
    for batch in datagen.flow(x,
                              batch_size=32,
                              save_to_dir='/media/scrutiny/Data/temp/Canny_pic/Aug/Scissor/',
                              save_prefix='test',
                              save_format='.jpg'):
        i += 1
        print(i)
        if i > 11:
            break
