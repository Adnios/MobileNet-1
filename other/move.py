from PIL import Image
import os
import shutil

img_h = 128
class BatchRename():
    def __init__(self):
        self.path = ''  # 存放图片的文件夹路径

    def rename(self):
        index = 1
        filelist = os.listdir(self.path)
        for item in filelist:
            if item.endswith('.jpg'):  # 图片格式为jpg
                if index % 4 == 0:
                    file = self.path + '/' + item
                    new_file = self.path + '/test/' + item
                    shutil.move(file, new_file)
                    print(file)
            index = index + 1


if __name__ == '__main__':

    demo = BatchRename()
    file = '/media/scrutiny/Data/temp/final/Scissor'
    demo.path =  file
    demo.rename()
    demo = BatchRename()
    file = '/media/scrutiny/Data/temp/final/Rock'
    demo.path = file
    demo.rename()
    demo = BatchRename()
    file = '/media/scrutiny/Data/temp/final/Paper'
    demo.path = file
    demo.rename()
