import tensorflow as tf
import os
import numpy as np


def conv2d_sim(input,weight,strides,padding="valid",name="sim_convolution"):
    num_filters = tf.shape(input)[-1]
    with tf.variable_scope(name):
        bias = tf.get_variable("bias",[num_filters],dtype=tf.float32,initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(input,weight,[1]+strides+[1],padding=padding)
        conv = tf.nn.bias_add(conv,bias)
    return conv


def group_conv2d(input,weight,group,strides=[1,1],dilations=[1,1],pad_size=0,pad="VALID",name="group_convolution"):
    n,h,w,c = input.get_shape().as_list()
    num_filters = c*group
    convolve = lambda i,k:conv2d_sim(i,k,[1]+strides+[1],padding=pad)
    with tf.variable_scope(name):
        if pad_size>0:
            input = tf.pad(input,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]],name="pad_input")
        weight_group = tf.split(weight,num_or_size_splits=group,axis=3)
        input_group  = tf.split(input,num_or_size_splits=group,axis=3)
        conv_group = [convolve(i,k) for i,k in zip(input_group,weight_group)]
        conv       = tf.concat(conv_group,axis=3)
        bias   = tf.get_variable("bias",[num_filters],dtype=tf.float32,initializer=tf.zeros_initializer())
        conv   = tf.nn.conv2d(input,weight,[1]+strides+[1],padding=pad,dilations=[1]+dilations+[1])
        conv   = tf.nn.bias_add(conv,bias)
    return conv

def gaussian(window_size,sigma,channel=3):
    gauss_1D = np.array([np.exp(-(x-window_size//2)**2/float(2*sigma**2)) for x in range(window_size)],dtype=np.float)
    gauss_1D /= gauss_1D.sum()
    gauss_1D  = np.expand_dims(gauss_1D,axis=1)
    gauss_2D  = np.matmul(gauss_1D,np.transpose(gauss_1D))
    gauss_2D  = np.expand_dims(np.expand_dims(gauss_2D,axis=0),axis=0)
    window    = np.repeat(gauss_2D,channel,axis=0)
    #window    = np.reshape(window,[channel,1,window_size,window_size])
    return window


def ssim(im1,im2,window_size,sigma,channel=3,name="ssim"):
    assert_op = tf.Assert(tf.equal(channel,3),[channel],name="channel_assert")
    with tf.control_dependencies([assert_op]):
        gauss_window = gaussian(window_size,sigma,channel)
    with tf.variable_scope(name):
        write next day
        


if __name__=="__main__":
    window_size = 11
    sigma = 1.5
    gaussian_filter = gaussian(window_size,sigma)
    print(gaussian_filter)