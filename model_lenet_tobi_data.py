import tensorflow as tf
from layers import depthwise_separable_conv2d, conv2d, avg_pool_2d, dense, flatten, dropout
import os
from utils import load_obj, save_obj
import numpy as np
'''
tobi数据集,调整参数
'''

class MobileNet:
    """
    MobileNet Class
    """

    def __init__(self,
                 args):

        # init parameters and input
        self.X = None
        self.y = None
        self.logits = None
        self.is_training = None
        self.loss = None
        self.regularization_loss = None
        self.cross_entropy_loss = None
        self.train_op = None
        self.accuracy = None
        self.y_out_argmax = None
        self.summaries_merged = None
        self.args = args
        self.mean_img = None
        self.nodes = dict()

        self.pretrained_path = os.path.realpath(self.args.pretrained_path)

        self.__build()

    def __init_input(self):
        with tf.variable_scope('input'):
            # Input images
            self.X = tf.placeholder(tf.float32,
                                    [self.args.batch_size, self.args.img_height, self.args.img_width,
                                     self.args.num_channels])
            # Classification supervision, it's an argmax. Feel free to change it to one-hot,
            # but don't forget to change the loss from sparse as well
            self.y = tf.placeholder(tf.int32, [self.args.batch_size])
            # is_training is for batch normalization and dropout, if they exist
            self.is_training = tf.placeholder(tf.bool)

    def __init_mean(self):
        # 图片均值化，改为单通道,应该可以再优化，这个值取多少比较合适稍后再研究
        # Preparing the mean image.
        img_mean = np.ones((1, 32, 32, 1))
        img_mean[:, :, :, 0] *= 103.939
        # img_mean[:, :, :, 1] *= 116.779
        # img_mean[:, :, :, 2] *= 123.68
        self.mean_img = tf.constant(img_mean, dtype=tf.float32)

    def __build(self):
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_mean()
        self.__init_input()
        self.__init_network()
        self.__init_output()

    def __init_network(self):
        with tf.variable_scope('mobilenet_encoder'):
            # Preprocessing as done in the paper
            with tf.name_scope('pre_processing'):
                preprocessed_input = (self.X - self.mean_img) / 255.0

            # Model is here!
            # Conv0 5*5 6
            with tf.variable_scope('conv0') as scope:
                # 建立weights和biases的共享变量
                # conv1, shape = [kernel size, kernel size, channels, kernel numbers]
                weights = tf.get_variable('weights',
                                          shape=[5, 5, 3, 6],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                                      dtype=tf.float32))  # stddev标准差
                biases = tf.get_variable('biases',
                                         shape=[6],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))
                # 卷积层 strides = [1, x_movement, y_movement, 1], padding填充周围有valid和same可选择
                conv = tf.nn.conv2d(preprocessed_input, weights, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)  # 加入偏差
                conv0 = tf.nn.relu(pre_activation, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

            # Pool0 2*2 2
            with tf.variable_scope('pooling0_lrn') as scope:
                pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='pooling0')


            # Conv1 3*3 1
            with tf.variable_scope('conv1') as scope:
                # 建立weights和biases的共享变量
                # conv1, shape = [kernel size, kernel size, channels, kernel numbers]
                weights = tf.get_variable('weights',
                                          shape=[5, 5, 6, 16],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                                      dtype=tf.float32))  # stddev标准差
                biases = tf.get_variable('biases',
                                         shape=[16],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))
                # 卷积层 strides = [1, x_movement, y_movement, 1], padding填充周围有valid和same可选择
                conv = tf.nn.conv2d(conv0, weights, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)  # 加入偏差
                conv1 = tf.nn.relu(pre_activation, name=scope.name)  # 加上激活函数非线性化处理，且是在conv1的命名空间

            # Pool1 2*2 2
            with tf.variable_scope('pooling1_lrn') as scope:
                pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='pooling1')


            # print("Conv1 = ", end_time - begin_time)

            # Conv2 5*5 1

            with tf.variable_scope('conv2') as scope:
                weights = tf.get_variable('weights',
                                          shape=[5, 5, 16, 120],  # 这里只有第三位数字需要等于上一层的tensor维度
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
                biases = tf.get_variable('biases',
                                         shape=[120],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)
                conv2 = tf.nn.relu(pre_activation, name='conv2')


            with tf.variable_scope('conv3') as scope:
                weights = tf.get_variable('weights',
                                          shape=[1, 1, 120, 84],  # 这里只有第三位数字需要等于上一层的tensor维度
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
                biases = tf.get_variable('biases',
                                         shape=[84],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)
                conv3 = tf.nn.relu(pre_activation, name='conv3')


            ############################################################################################
            self.logits = flatten(conv2d('fc', conv3, kernel_size=(1, 1), num_filters=self.args.num_classes,
                                         l2_strength=self.args.l2_strength,
                                         bias=self.args.bias))
            self.__add_to_nodes([self.logits])


    def __init_output(self):
        with tf.variable_scope('output'):
            self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            # softmax
            self.cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='loss'))
            self.loss = self.regularization_loss + self.cross_entropy_loss

            # Important for Batch Normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)
            self.y_out_argmax = tf.cast(tf.argmax(tf.nn.softmax(self.logits), axis=-1), tf.int32)
            #self.y_out_argmax = tf.argmax(tf.nn.softmax(self.logits), axis=-1, output_type=tf.int32)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))

        # Summaries needed for TensorBoard
        with tf.name_scope('train-summary-per-iteration'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('acc', self.accuracy)
            self.summaries_merged = tf.summary.merge_all()

    def __restore(self, file_name, sess):
        try:
            print("Loading ImageNet pretrained weights...")
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mobilenet_encoder')
            dict = load_obj(file_name)
            run_list = []
            for variable in variables:
                for key, value in dict.items():
                    if key in variable.name:
                        run_list.append(tf.assign(variable, value))
            sess.run(run_list)
            print("ImageNet Pretrained Weights Loaded Initially\n\n")
        except:
            print("No pretrained ImageNet weights exist. Skipping...\n\n")

    def load_pretrained_weights(self, sess):
        self.__restore(self.pretrained_path, sess)

    def __add_to_nodes(self, nodes):
        for node in nodes:
            self.nodes[node.name] = node

    def __init_global_epoch(self):
        """
        Create a global epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

    def __init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

