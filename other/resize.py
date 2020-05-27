from PIL import Image
import os
'''
修改目录名即可
'''
img_h = 128
class BatchRename():
    def __init__(self):
        self.path = ''  # 存放图片的文件夹路径

    def rename(self):
        filelist = os.listdir(self.path)
        for item in filelist:
            if item.endswith('.jpg'):  # 图片格式为jpg
                file = self.path + '/' + item
                img = Image.open(file)
                img = img.resize([img_h, img_h])
                img.save(file)
                print(file)


if __name__ == '__main__':

    demo = BatchRename()
    file = '/media/scrutiny/Data/temp/heatmap/Scissor'
    demo.path =  file
    demo.rename()
    demo = BatchRename()
    file = '/media/scrutiny/Data/temp/heatmap/Rock'
    demo.path = file
    demo.rename()
    demo = BatchRename()
    file = '/media/scrutiny/Data/temp/heatmap/Paper'
    demo.path = file
    demo.rename()
