import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
'''
load tobi数据集
'''

class DataLoader:
    """Data Loader class. As a simple case, the model is tried on TinyImageNet. For larger datasets,
    you may need to adapt this class to use the Tensorflow Dataset API"""

    def __init__(self, batch_size, shuffle=False):
        self.X_train = None
        self.y_train = None
        self.img_mean = None
        self.train_data_len = 0

        self.X_val = None
        self.y_val = None
        self.val_data_len = 0

        self.X_test = None
        self.y_test = None
        self.test_data_len = 0

        self.shuffle = shuffle
        self.batch_size = batch_size

    def load_data(self):
        '''
        因为数据集可能不同，所以还是在这儿每次手动改比较好
        '''
        gesture = []          # 单通道图片需要再手动reshape一下
        label_gesture = []
        file_dir = 'avi_to_pic/pic/'
        file_dir_list = os.listdir(file_dir)
        dir_sum = 0
        pic_sum = 0
        for sub_dir in file_dir_list:
            dir_sum = dir_sum + 1
            sub_dir_path = file_dir + sub_dir + '/'
            print(sub_dir_path)
            if sub_dir.startswith('scissors'):
                sub_dir_list = os.listdir(sub_dir_path)
                for item in sub_dir_list:
                    pic_sum = pic_sum + 1
                    gesture.append(plt.imread(sub_dir_path + item).reshape([64, 64, 3]))
                    label_gesture.append(0)
            elif sub_dir.startswith('rock'):
                sub_dir_list = os.listdir(sub_dir_path)
                for item in sub_dir_list:
                    pic_sum = pic_sum + 1
                    gesture.append(plt.imread(sub_dir_path + item).reshape([64, 64, 3]))
                    label_gesture.append(1)
            elif sub_dir.startswith('paper'):
                sub_dir_list = os.listdir(sub_dir_path)
                for item in sub_dir_list:
                    pic_sum = pic_sum + 1
                    gesture.append(plt.imread(sub_dir_path + item).reshape([64, 64, 3]))
                    label_gesture.append(2)
            elif sub_dir.startswith('background'):
                sub_dir_list = os.listdir(sub_dir_path)
                for item in sub_dir_list:
                    pic_sum = pic_sum + 1
                    gesture.append(plt.imread(sub_dir_path + item).reshape([64, 64, 3]))
                    label_gesture.append(3)
            else :
                print("Erro", sub_dir)

        print("dir_sum=",dir_sum,"pic_sum=",pic_sum)

        train_data = np.array(gesture)
        val_data = np.array(gesture)
        self.X_train = train_data
        self.y_train = np.array(label_gesture)
        self.X_val = val_data
        self.y_val = np.array(label_gesture)
        self.train_data_len = self.X_train.shape[0]
        self.val_data_len = self.X_val.shape[0]
        img_height = 64
        img_width = 64
        print(val_data.shape)
        num_channels = 3
        return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

    def generate_batch(self, type='train'):
        """Generate batch from X_train/X_test and y_train/y_test using a python DataGenerator"""
        if type == 'train':
            # Training time!
            new_epoch = True
            start_idx = 0
            mask = None
            while True:
                if new_epoch:
                    start_idx = 0
                    if self.shuffle:
                        mask = np.random.choice(self.train_data_len, self.train_data_len, replace=False)
                    else:
                        mask = np.arange(self.train_data_len)
                    new_epoch = False

                # Batch mask selection
                X_batch = self.X_train[mask[start_idx:start_idx + self.batch_size]]
                y_batch = self.y_train[mask[start_idx:start_idx + self.batch_size]]
                start_idx += self.batch_size

                # Reset everything after the end of an epoch
                if start_idx >= self.train_data_len:
                    new_epoch = True
                    mask = None
                yield X_batch, y_batch
        elif type == 'test':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_test[start_idx:start_idx + self.batch_size]
                y_batch = self.y_test[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.test_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        elif type == 'val':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_val[start_idx:start_idx + self.batch_size]
                y_batch = self.y_val[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.val_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        else:
            raise ValueError("Please select a type from \'train\', \'val\', or \'test\'")
