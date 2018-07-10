import tensorflow as tf
import copy

def SwitchNorm(input,data_format="NHWC",eps=1e-5):
    x = copy.deepcopy(input)
    if data_format =="NHWC":
        x = tf.transpose(x,[0,3,1,2])
        n,c,h,w = x.get_shape().as_list()
        x = tf.reshape(x,[n,c,-1])
    else:
        n,c,h,w = x.get_shape().as_list()
        x = tf.rehape(x,[n,c,-1])

    weight = tf.get_variable("weight",[1,c,1,1],dtype=tf.float32,initializer=tf.ones_initializer())
    bias   = tf.get_variable("bias",[1,c,1,1],dtype=tf.float32,initializer=tf.zeros_initializer())
    mean_weight = tf.get_variable("mean_weight",[3],dtype=tf.float32,initializer=tf.ones_initialzier())
    var_weight  = tf.get_variable("var_weight",[3],dtype=tf.float32,initializer=tf.ones_initialzier())


    with tf.variable_scope("in"):
        # w.h set to 1
        mean_in,var_in = tf.nn.moments(x,axes=-1,keep_dims=True)


    #diffenrent from pytorch version,which use approximation
    #mean_ln = mean_in.mean(1, keepdim=True)
    #temp = var_in + mean_in ** 2
    #var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2
    with tf.varaible_scope("ln"):
        # c set to 1
        mean_ln,var_ln = tf.nn.moments(x,axes=1,keep_dims=True)

    with tf.variable_scope("bn"):
        #set n,w,h to 1
        mean_bn,var_bn = tf.nn.moments(x,axes=[0,2],keep_dims=1)

    mean_weight = tf.nn.softmax(mean_weight)
    var_weight = tf.nn.softmax(var_weight)

    mean = mean_weight[0]*mean_in + mean_weight[1]*mean_ln + mean_weight[2]*mean_bn
    var  = var_weight[0]*var_in   + var_weight[1]*var_ln   + var_weight[2]*var_bn

    x = (x-mean)/tf.sqrt((var+eps))
    x = tf.reshape(x,[n,h,w,c])
    x = x*weight+bias
    return x

