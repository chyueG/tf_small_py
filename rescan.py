import tensorflow as tf
import numpy as np
import torch



def Conv2d(input,num_filters,ksize=[3,3],strides=[1,1],dilations=[1,1],pad="VALID",name="conv"):
    n,h,w,c = input.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable("weight",ksize+[c,num_filters],dtype=tf.float32,initializer=tf.glorot_uniform_initializer())
        bias   = tf.get_variable("bias",[num_filters],dtype=tf.float32,initializer=tf.zeros_initializer())
        conv   = tf.nn.conv2d(input,weight,[1]+strides+[1],padding=pad,dilations=[1]+dilations+[1])
        conv   = tf.nn.bias_add(conv,bias)
    return conv

def Snorm(input,name="switch_norm"):
    n,h,w,c = input.get_shape().as_list()
    eps     = 1e-3
    with tf.variable_scope(name):

        weight = tf.get_variable("weight",[1,1,1,c],dtype=tf.float32,initializer=tf.ones_initializer())
        bias   = tf.get_variable("bias",[c],dtype=tf.float32,initializer=tf.zeros_initializer())
        mean_weight = tf.get_variable("weight_mean",[3],dtype=tf.float32,initializer=tf.ones_initializer())
        var_weight  = tf.get_variable("weight_var",[3],dtype=tf.float32,initializer=tf.ones_initializer())

        # in batch
        mean_ins,var_ins = tf.nn.moments(input,axes=[1,2],keep_dims=True)

        mean_ln,var_ln   = tf.nn.moments(input,axes=[1,2,3],keep_dims=True)

        mean_bn,var_bn   = tf.nn.moments(input,axes=[0,1,2],keep_dims=True)

        mean_weight  = tf.nn.softmax(mean_weight)
        var_weight   = tf.nn.softmax(var_weight)

        mean  = mean_weight[0]*mean_ins + mean_weight[1]*mean_ln + mean_weight[2]*mean_bn
        var   = var_weight[0]*var_ins   + var_weight[1]*var_ln   + var_weight[2]*var_bn

        norm = (input-mean)/tf.sqrt(var+eps)
        norm = norm*weight + bias
        """
        offset = tf.get_variable("offset",[c],dtype=tf.float32,initializer=tf.zeros_initializer())
        scale  = tf.get_variable("scale",[c],dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.2))
        mean,variance = tf.nn.moments(input,axes=[0,1,2],keep_dims=False)
        variance_epsilon = 1e-3
        norm = tf.nn.batch_normalization(input,mean,variance,offset,scale,variance_epsilon=variance_epsilon)
        """
    return norm


def Global_Pool(input,name="global_pooling"):
    w,h,w,c  = input.get_shape().as_list()
    with tf.variable_scope(name):
        output = tf.nn.avg_pool(input,[1,h,w,1],[1,1,1,1],padding="VALID")
    return output



def SE(input,num_filters,ratio,name="SE_block"):
    with tf.variable_scope(name):
        output = Global_Pool(input)
        output = Conv2d(output,num_filters/ratio,[1,1],name="fc_1")
        output = tf.nn.relu(output)
        output = Conv2d(output,num_filters,[1,1],name="fc_2")
        output = tf.sigmoid(output,name="sigmoid")
        output = input*output
    return output

def RNN(input,num_filters,kernel_size,dilations,h=None,name="RNN"):
    with tf.variable_scope(name):
        pad_x = int(dilations[0]*(kernel_size-1)/2)
        input_x  = tf.pad(input,[[0,0],[pad_x,pad_x],[pad_x,pad_x],[0,0]],name="pad_x")
        conv_x = Conv2d(input_x,num_filters,kernel_size,dilations=dilations)

        pad_h  = int((kernel_size)/2)
        input_h = tf.pad(input,[[0,0],[pad_h,pad_h],[pad_h,pad_h],[0,0]],name="pad_h")
        conv_h = Conv2d(input_h,num_filters,kernel_size,name="conv_h")

        if h:
            z = tf.tanh(conv_x+conv_h)
        else:
            h = tf.tanh(conv_x)

        h = tf.nn.leaky_relu(h,alpha=0.2)
        return h,h

def GRU(input,num_filters,kernel_size,dilations,ratio,pair=None,name="GRU"):
    with tf.variable_scope(name):
        pad_x = int(dilations*(kernel_size-1)/2)
        input_x = tf.pad(input,[[0,0],[pad_x,pad_x],[pad_x,pad_x],[0,0]],name="input_x")
        conv_xf = Conv2d(input_x,num_filters,kernel_size,dilations,name="conv_xf")
        conv_xi = Conv2d(input_x,num_filters,kernel_size,dilations,name="conv_xi")
        conv_xo = Conv2d(input_x,num_filters,kernel_size,dilations,name="conv_xo")
        conv_xj = Conv2d(input_x,num_filters,kernel_size,dilations,name="conv_xj")

        pad_h   = int((kernel_size-1)/2)
        input_h = tf.pad(input,[[0,0],[pad_h,pad_h],[pad_h,pad_h],[0,0]],name="input_y")
        conv_hf = Conv2d(input_h,num_filters,kernel_size,name="conv_hf")
        conv_hi = Conv2d(input_h,num_filters,kernel_size,name="conv_hi")
        conv_ho = Conv2d(input_h,num_filters,kernel_size,name="conv_hi")
        conv_hj = Conv2d(input_h,num_filters,kernel_size,name="conv_hi")

        if pair:
            h,c = pair
            f   = tf.sigmoid(conv_xf+conv_hf)
            i   = tf.sigmoid(conv_xi+conv_hi)
            o   = tf.sigmoid(conv_xo+conv_ho)
            j   = tf.tanh(conv_xj+conv_hj)
            c   = f*c + i*j
            h   = o*c
        else:
            i   = tf.sigmoid(conv_xi)
            o   = tf.sigmoid(conv_xo)
            j   = tf.tanh(conv_xj)
            c   = i*j
            h   = o*c

        output = SE(h,num_filters,ratio)
        output = tf.nn.leaky_relu(output,alpha=0.2)
    return output,[output,c]