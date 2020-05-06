from FLAGS import *
import os
import tensorlayer as tl
import numpy as np
from tensorflow.contrib import layers
"""
This file contains some basic blocks that are widely used in CNNs
"""


def weight_variable(shape, name, wd=True):
    """
    A function to create weight variables
    :param shape: The shape of weight
    :param name: The name of the weight
    :param wd: Whether or not this variable should be weight decade
    :return: A weight-variableds
    """
    initializer = tf.glorot_uniform_initializer()#tf.contrib.layers.xavier_initializer()  # tf.truncated_normal_initializer(stddev=0.1)
    if wd:
        weight = tf.get_variable(name='weight'+name, shape=shape,
                                 initializer=initializer,
                                 collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    else:
        weight = tf.get_variable(name='weight' + name, shape=shape,
                                 initializer=initializer,
                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES])

    return weight


def bias_variable(shape, name):
    """
    A function to create bias variable
    :param shape: The shape of the bias
    :param name: The name of the bias
    :return: A bias-variable
    """
    bias = tf.get_variable(name='bias'+name, shape=shape, initializer=tf.constant_initializer(0))
    return bias


def create_path(_time):
    """
    Create a path according to current time
    :param _time: time
    :return: None
    """
    dir_temp = FLAGS.log_save_path
    if not os.path.exists(dir_temp):
        os.mkdir(dir_temp)
    if not os.path.exists(dir_temp + FLAGS.dataSet + '/'):
        os.mkdir(dir_temp + FLAGS.dataSet + '/')
    if not os.path.exists(dir_temp + FLAGS.dataSet + '/' + FLAGS.LossType + '/'):
        os.mkdir(dir_temp + FLAGS.dataSet + '/' + FLAGS.LossType + '/')
    if not os.path.exists(dir_temp + FLAGS.dataSet + '/' + FLAGS.LossType + '/' + _time + '/'):
        os.mkdir(dir_temp + FLAGS.dataSet + '/' + FLAGS.LossType + '/' + _time + '/')


def distance(emb1, emb2):
    """
    Calculate the sqrt l2 distance between two embedding
    :param emb1: embedding 1
    :param emb2: embedding 2
    :return: The distance
    """
    return tf.sqrt(tf.reduce_sum(tf.square(emb1-emb2), axis=1, keep_dims=True))


def bn_block(embedding, normal, is_Training, name, reuse=False):
    """
    Batch Normalization Block
    :param embedding: embedding
    :param normal: If this is True, BN will be conducted
    :param is_Training: Whether is training or not
    :param name: The name of the variable scope
    :param reuse: Whether reuse this block
    :return:
    """
    if normal:
        with tf.variable_scope(name, reuse=reuse):
            embedding = tf.layers.batch_normalization(
                    inputs=embedding, center=True,
                    scale=True, training=is_Training, fused=True
            )
        print("BN layer: "+name+" is applied")
    return embedding

def normalize_block(embedding, normal, is_Training, name, reuse=False):
    """
    Batch Normalization Block
    :param embedding: embedding
    :param normal: If this is True, BN will be conducted
    :param is_Training: Whether is training or not
    :param name: The name of the variable scope
    :param reuse: Whether reuse this block
    :return:
    """
    if normal:
        with tf.variable_scope(name, reuse=reuse):
            embedding = tf.nn.l2_normalize(
                    embedding, axis=1
            )
        print("normalize layer: "+name+" is applied")
    return embedding

def Generator(embedding, is_Training=True):
        # generator fc3
    if FLAGS.ADD_NOISE:
        noise_rand = tf.random_normal(shape=tf.shape(embedding), mean=0.0, stddev=(50) / (255), dtype=tf.float32)
        embedding_g = fc_block(
            embedding + noise_rand, in_d=1024, out_d=2048,
            name='generator1', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
        )
    else:
        embedding_g = fc_block(
            embedding, in_d=1024, out_d=2048,
            name='generator1', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
        )
    # embedding_g = tf.nn.dropout(embedding_g, keep_prob=0.9)
    embedding_g = fc_block(
        embedding_g, in_d=2048, out_d=1024,
        name='generator2', is_bn=True, is_relu=True, is_Training=is_Training, is_Tanh=True
    )
    embedding_g = fc_block(
        embedding_g, in_d=1024, out_d=1024,
        name='generator3', is_bn=False, is_relu=False, is_Training=is_Training
    )
    return embedding_g

def Discriminator(embedding, is_Training=True, name_diy=''):
    net = layers.fully_connected(embedding, 512, activation_fn=tf.nn.leaky_relu,
                           weights_initializer=tf.glorot_uniform_initializer())
    net = tf.layers.batch_normalization(
        inputs=net, center=True,
        scale=True, training=is_Training, fused=True
    )
    net = tf.nn.dropout(net, keep_prob=0.4)
    net = layers.fully_connected(net, 1, activation_fn=None,
                                 weights_initializer=tf.glorot_uniform_initializer())
    return net
    # embedding = fc_block(
    #     embedding, in_d=1024, out_d=512,
    #     name='discriminator1'+ name_diy, is_bn=True, is_relu=False, is_Training=is_Training, is_LeakyReLU=True
    # )
    # embedding = tf.nn.dropout(embedding, keep_prob=0.9)
    # label = fc_block(
    #     embedding, in_d=512, out_d=1,
    #     name='discriminator2'+ name_diy, is_bn=False, is_relu=False, is_Training=is_Training
    # )
    # return label

def fc_block(embedding, in_d, out_d, name, is_bn, is_relu, is_Training=True, reuse=False, is_LeakyReLU=False, is_Tanh=False):
    """
    Fully-connected Block
    :param embedding: embedding
    :param in_d: the input dimension
    :param out_d: the output dimension
    :param name: the name
    :param is_bn: whether apply BN
    :param is_relu: whether use relu
    :param is_Training: whether is Training
    :param reuse: whether reuse this block
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        W_fc = weight_variable([in_d, out_d], name + "w")
        b_fc = bias_variable([out_d], name + "b")
        embedding = tf.matmul(embedding, W_fc) + b_fc
        assert not (is_LeakyReLU and is_Tanh)

        if is_LeakyReLU:
            embedding = tf.nn.leaky_relu(embedding)
        elif is_Tanh:
            embedding = tf.nn.tanh(embedding)
        elif is_relu:
            embedding = tf.nn.relu(embedding)

        if is_bn:
            embedding = bn_block(embedding, normal=True, is_Training=is_Training, name=name + 'BN')
        return embedding


def conv2d_block(feature, kernel_size, strides, padding, name, is_bn, is_relu, is_Training=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        W_2d = weight_variable(kernel_size, name + "w")
        # b_2d = bias_variable(kernel_size, name + "b")
        feature = tf.nn.conv2d(feature, W_2d, strides=strides, padding=padding)
        if is_relu:
            feature = tf.nn.relu(feature)
        if is_bn:
            feature = bn_block(feature, normal=True, is_Training=is_Training, name=name + 'BN')
        return feature


def training(loss, lr,  var_scope='None', g_truncate=False, g_limit=2.):
    loss = tf.cond(tf.is_nan(loss), true_fn=lambda: tf.constant(0.), false_fn=lambda: loss)
    with tf.name_scope('Training'):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            if g_truncate:

                optimizer = tf.train.AdamOptimizer(lr)
                if var_scope != 'None':
                    var_list=[]
                    for m in var_scope.split('*'):
                        var_list .append( tl.layers.get_variables_with_name(m, True, True))
                    gvs = optimizer.compute_gradients(loss=loss, var_list=var_list)

                else:
                    gvs = optimizer.compute_gradients(loss=loss)
                capped_gvs = [(tf.clip_by_value(grad, -g_limit, g_limit), var) for grad, var in gvs]
                train_step = optimizer.apply_gradients(capped_gvs)
                """
                optimizer = tf.train.AdamOptimizer(lr)
                if var_scope != 'None':
                    var_list = tl.layers.get_variables_with_name(var_scope, True, True)
                    gvs = optimizer.compute_gradients(loss=loss, var_list=var_list)

                else:
                    gvs = optimizer.compute_gradients(loss=loss)
                cleaned_gvs = [(tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad), val) for grad, val in gvs]
                train_step = optimizer.apply_gradients(cleaned_gvs)
                """
            else:
                if var_scope != 'None':
                    var_list = []
                    for m in var_scope.split('*'):
                        var_list += tl.layers.get_variables_with_name(m, True, True)
                        if m == 'Classifier':
                            var_list += tl.layers.get_variables_with_name('resnet_model', True, True)
                    train_step = tf.train.AdamOptimizer(lr).minimize(loss, var_list=var_list)
                else:
                    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
            return train_step


def batch_data(batch):
    x_batch_data, c_batch_data = batch
    # change images to [B,S,S,C]
    x_batch_data = np.transpose(x_batch_data[:, [2, 1, 0], :, :], (0, 2, 3, 1))
    # Reduce mean
    x_batch_data = x_batch_data - image_mean
    Label_raw = c_batch_data
    return x_batch_data, Label_raw


class learning_rate:
    def __init__(self, init_lr,  decay_step, init_tar, cycle, margin, is_raising=True):
        self.lr = init_lr
        self.decay_step = decay_step
        self.tar = init_tar
        self.margin = margin
        self.cycle = cycle
        self.tar_temp = []
        self.counter = 0
        self.is_raising = is_raising

    def update(self, tar):
        self.tar_temp.append(tar)
        self.counter = self.counter + 1
        if self.counter >= self.cycle:
            if np.mean(self.tar_temp) <= self.tar + self.margin:
                self.lr = self.lr * self.decay_step
            self.counter = 0
            self.tar = np.mean(self.tar_temp)
            self.tar_temp = []

    def get_lr(self):
        return self.lr


class data_collector:
    def __init__(self, tag, init):
        self.collector = []
        self.tag = tag
        self.mean = init

    def update(self, var):
        if not np.isnan(var):
            self.collector.append(var)
            # print(self.tag + ' : %f' % var)
            if np.shape(self.collector)[0] >= 64:#FLAGS.batch_per_epoch:
                self.mean = np.mean(self.collector)
                self.collector = []
        else:
            print(self.tag + ' : is nan, not record' )

    def write_to_tfboard(self, eval_summary):
        eval_summary.value.add(tag=self.tag, simple_value=self.mean)

    def read(self):
        return self.mean


class data_collector2:
    def __init__(self, tag, init):
        self.collector = []
        self.tag = tag
        self.mean = init
        self.mean_pre = -1

    def update(self, var, step):
        if not np.isnan(var):
            if step>1000:
                self.mean_pre = self.mean
            self.collector.append(var)
            # print(self.tag + ' : %f' % var)
            if np.shape(self.collector)[0] >= 64:#FLAGS.batch_per_epoch:
                self.mean = np.mean(self.collector)
                self.collector = []
        else:
            print(self.tag + ' : is nan, not record' )

    def write_to_tfboard(self, eval_summary):
        eval_summary.value.add(tag=self.tag, simple_value=self.mean)

    def read(self):
        return self.mean
    def prev(self):
        return self.mean_pre
