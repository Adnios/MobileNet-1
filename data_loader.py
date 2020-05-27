import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


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
        # Please make sure to change this function to load your train/validation/test data.
        # 数据集换成我们的数据集
        # train_data = np.array([plt.imread('./data/test_images/0.jpg'), plt.imread('./data/test_images/1.jpg'),
        #               plt.imread('./data/test_images/2.jpg'), plt.imread('./data/test_images/3.jpg')])
        # self.X_train = train_data
        # self.y_train = np.array([284, 264, 682, 2])

        # val_data = np.array([plt.imread('./data/test_images/0.jpg'), plt.imread('./data/test_images/1.jpg'),
        #             plt.imread('./data/test_images/2.jpg'), plt.imread('./data/test_images/3.jpg')])

        # self.X_val = val_data
        # self.y_val = np.array([284, 264, 682, 2])
        '''
        因为数据集可能不同，所以还是在这儿每次手动改比较好
        '''
        gesture_1 = []          # 单通道图片需要再手动reshape一下
        label_gesture_1 = []
        # file_dir = '/media/scrutiny/Data/temp/final/'
        # file_Scissor = file_dir + 'Scissor/'
        # file_Rock = file_dir + 'Rock/'
        # file_Paper = file_dir + 'Paper/'
        # file_Background = file_dir + 'Background/'
        # for file in os.listdir(file_Scissor):
        #     gesture_1.append(plt.imread(file_Scissor+file).reshape([128,128,1]))
        #     label_gesture_1.append(0)
        #
        # for file in os.listdir(file_Rock):
        #     gesture_1.append(plt.imread(file_Rock+file).reshape([128,128,1]))
        #     label_gesture_1.append(1)
        #
        # for file in os.listdir(file_Paper):
        #     gesture_1.append(plt.imread(file_Paper+file).reshape([128, 128, 1]))
        #     label_gesture_1.append(2)
        #
        # for file in os.listdir(file_Background):
        #     gesture_1.append(plt.imread(file_Background+file).reshape([128, 128, 1]))
        #     label_gesture_1.append(3)

        gesture_test = []  # 单通道图片需要再手动reshape一下
        label_gesture_test = []
        file_dir_test = '/mnt/d/temp/final/test/'
        file_Scissor_test = file_dir_test + 'Scissor/'
        file_Rock_test = file_dir_test + 'Rock/'
        file_Paper_test = file_dir_test + 'Paper/'
        file_Background = file_dir_test + 'Background/'
        for file in os.listdir(file_Scissor_test):
            gesture_test.append(plt.imread(file_Scissor_test + file).reshape([128, 128, 1]))
            label_gesture_test.append(0)

        for file in os.listdir(file_Rock_test):
            gesture_test.append(plt.imread(file_Rock_test + file).reshape([128, 128, 1]))
            label_gesture_test.append(1)

        for file in os.listdir(file_Paper_test):
            gesture_test.append(plt.imread(file_Paper_test + file).reshape([128, 128, 1]))
            label_gesture_test.append(2)

        for file in os.listdir(file_Background):
            gesture_1.append(plt.imread(file_Background+file).reshape([128, 128, 1]))
            label_gesture_1.append(3)

        train_data = np.array(gesture_1)
        val_data = np.array(gesture_test)
        self.X_train = train_data
        self.y_train = np.array(label_gesture_1)
        self.X_val = val_data
        self.y_val = np.array(label_gesture_test)
        self.train_data_len = self.X_train.shape[0]
        self.val_data_len = self.X_val.shape[0]
        img_height = 128
        img_width = 128
        print(val_data.shape)
        num_channels = 1
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
