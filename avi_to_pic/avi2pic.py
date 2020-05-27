# @Author  : Jeff

"""
avi file to jpg file
"""
from cv2 import cv2 as cv
import argparse
import os
import pdb


def Img(avi_folder, save_folder, frame_interval):
    for avi in os.listdir(avi_folder):
        print(avi)
        save_dir = save_folder + os.path.splitext(avi)[0] + "/"
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)

        video = cv.VideoCapture(avi_folder+avi)
        rval, frame = video.read()
        if rval == True:
            fps = video.get(cv.CAP_PROP_FPS)
            frame_all = video.get(cv.CAP_PROP_FRAME_COUNT)
            print("[INFO] 视频FPS: {}".format(fps))
            print("[INFO] 视频总帧数: {}".format(frame_all))
            print("[INFO] 视频时长: {}s".format(frame_all/fps))

        frame_count = 1
        count = 0
        while rval == True:
            rval, frame = video.read()
            try:
                if frame_count % frame_interval == 0:
                    count += 1
                    filename = os.path.sep.join(
                        [save_dir, "{}.png".format(count)])
                    # print(filename)
                    # print(frame)
                    cv.imwrite(filename, frame)
                    print("保存图片:{}".format(filename))
                frame_count += 1
            except:
                break
        # 关闭视频文件
        video.release()
        print("[INFO] 总共保存：{}张图片".format(count))


if __name__ == "__main__":
    avi_dir = "avi_to_pic/avi/"
    save_dir = "avi_to_pic/pic/"
    Img(avi_dir, save_dir, 1)
