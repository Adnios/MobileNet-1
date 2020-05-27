from utils import parse_args, create_experiment_dirs, calculate_flops
from model_6 import MobileNet
# from model import MobileNet
from train import Train
from data_loader2 import DataLoader
from summarizer import Summarizer
import tensorflow as tf
from time import *
from easydict import EasyDict as edict
import json
'''
只测试一张图片,后来没有使用
img为待识别图片地址
test2.json为各种参数，模型地址对应里面的experiment_dir
mian函数最后返回的是一个链表，可能需要修改！！！
'''
def main(img):

    config = 'config/test2.json'
    with open(config, 'r') as config_file:
        config_args_dict = json.load(config_file)


    config_args = edict(config_args_dict)

    print(config_args)
    print("\n")

    # Create the experiment directories
    _, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(config_args.experiment_dir)

    # Reset the default Tensorflow graph
    tf.reset_default_graph()

    # Tensorflow specific configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Data loading
    data = DataLoader(config_args.batch_size, config_args.shuffle, img)
    print("Loading Data...")
    config_args.img_height, config_args.img_width, config_args.num_channels, \
    config_args.train_data_size, config_args.test_data_size = data.load_data()
    print("Data loaded\n\n")

    # Model creation
    print("Building the model...")
    model = MobileNet(config_args)
    print("Model is built successfully\n\n")

    # Summarizer creation
    summarizer = Summarizer(sess, config_args.summary_dir)
    # Train class
    trainer = Train(sess, model, data, summarizer)

    print("Final test!")
    ans_list = trainer.test('val')
    # ans_list = trainer.test('train')
    # print(len(ans_list))
    print(ans_list)
    print("Testing Finished\n\n")


if __name__ == '__main__':
    b = time()
    img = '/mnt/d/temp/final/Scissor/1.jpg'
    main(img)
    e = time()
    print("耗时：", e - b)
