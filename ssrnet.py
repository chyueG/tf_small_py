import tensorflow as tf
tf.set_random_seed(2**10)


def Conv2d(input,num_filters,ksize=[3,3],strides=[1,1],pad="VALID",name="conv"):
    n,h,w,c = input.get_shape().as_list()
    with tf.variable_scope(name):
        weight = tf.get_variable("weight",ksize+[c,num_filters],dtype=tf.float32,initializer=tf.glorot_uniform_initializer())
        bias   = tf.get_variable("bias",[num_filters],dtype=tf.float32,initializer=tf.zeros_initializer())
        conv   = tf.nn.conv2d(input,weight,[1]+strides+[1],padding=pad)
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


def basic_block(input,num_filter,branch,name,pool=True):
    with tf.variable_scope(name):
        x = Conv2d(input,num_filter,[3,3])
        x = Snorm(x)
        if branch == 1:
            x = tf.nn.relu(x)
        else:
            x = tf.tanh(x)
        if pool==True:
            x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],"VALID")
    return x

def fusion_block(input,stage,num_filter,name,pool=None,pkernel=1):
    with tf.variable_scope(name):
        output = Conv2d(input, num_filter, [1, 1])
        output = tf.nn.relu(output)
        if pool=="max":
            output = tf.nn.max_pool(output, [1, pkernel, pkernel, 1], [1, pkernel, pkernel, 1], padding="VALID")
        elif(pool=="ave"):
            output = tf.nn.avg_pool(output, [1, pkernel, pkernel, 1], [1, pkernel, pkernel, 1], padding="VALID")
        else:
            output = output
        output = Global_Pool(output)
        fc = tf.nn.dropout(output, keep_prob=0.2)
        fc = Conv2d(fc, stage, [1, 1], name="full_conv")
        fc  = tf.nn.relu(fc)
    return output,fc

def multiply_block(input1,input2,num_filter,name,activation=None):
    with tf.variable_scope(name):
        output = input1*input2
        #output = Global_Pool(output)
        output = Conv2d(output,num_filter,[1,1])
        if activation=="relu":
            output = tf.nn.relu(output)
        elif(activation=="tanh"):
            output = tf.tanh(output)
        else:
            output = output
    return output



def SSRbackbone(input,lambda_local=0.25,lambda_d=0.25,stage=[3,3,3],v=90):
    net = {}
    cnt = 0
    with tf.variable_scope("SSRNet"):
        with  tf.variable_scope("branch1"):
            x1 = basic_block(input,32,branch=1,name="first")
            x2 = basic_block(x1,32,branch=1,name="second")
            x3 = basic_block(x2,32,branch=1,name="third")
            x4 = basic_block(x3,32,branch=1,name="fourth",pool=False)
        net["x1"] = x1
        net["x2"] = x2
        net["x3"] = x3
        net["x4"] = x4

        with tf.variable_scope("branch2"):
            s1 = basic_block(input,16,branch=2,name="first")
            s2 = basic_block(s1,16,branch=2,name="second")
            s3 = basic_block(s2,16,branch=2,name="third")
            s4 = basic_block(s3,16,branch=2,name="fourth",pool=False)
        net["s1"] = s1
        net["s2"] = s2
        net["s3"] = s3
        net["s4"] = s4
        with tf.variable_scope("stage1"):
            s_conv,s_fc = fusion_block(s4,stage[0],10,name="s_stream")
            x_conv,x_fc = fusion_block(x4,stage[0],10,name="x_stream")
            delta1      = multiply_block(s_conv,x_conv,1,name="delta1",activation="tanh")
            delta1      = tf.squeeze(delta1,[1,2])
            com1        = multiply_block(s_fc,x_fc,2*stage[0],name="comb",activation="tanh")

            pred_1      = Conv2d(com1,stage[0],[1,1],name="pred_1")
            pred_1      = tf.nn.relu(pred_1)
            pred_1      = tf.squeeze(pred_1,[1,2])

            local_1     = Conv2d(com1,stage[0],[1,1],name="local_1")
            local_1     = tf.tanh(local_1)
            local_1     = tf.squeeze(local_1,[1,2])

            a = stage[0]*(stage[0]-1)/2+lambda_local*tf.reduce_sum(pred_1*local_1)
            a = a/(stage[0]*(1+lambda_d*delta1))

            net["pred1"] = pred_1
            net["local1"] = local_1
            net["delta1"] = delta1


        with tf.variable_scope("stage2"):
            s2_conv,s2_fc = fusion_block(s2,stage[1],10,name="s2_stream",pool="max",pkernel=4)
            x2_conv,x2_fc = fusion_block(x2,stage[1],10,name="x2_stream",pool="ave",pkernel=4)

            delta2 = multiply_block(s2_conv, x2_conv, 1, name="delta2", activation="tanh")
            delta2 = tf.squeeze(delta2,[1,2])
            com2 = multiply_block(s2_fc, x2_fc, 2 * stage[1], name="comb", activation="tanh")

            pred_2 = Conv2d(com2, stage[1], [1, 1],name="pred_2")
            pred_2 = tf.nn.relu(pred_2)
            pred_2 = tf.squeeze(pred_2,[1,2])

            local_2 = Conv2d(com2, stage[1], [1, 1],name="local_2")
            local_2 = tf.tanh(local_2)
            local_2 = tf.squeeze(local_2,[1,2])

            b = stage[1]*(stage[1]-1)/2+lambda_local*tf.reduce_sum(pred_2*local_2)
            b = b/(stage[0]*(1+lambda_d*delta1))/(stage[1]*(1+lambda_d*delta2))

            net["delta2"] = delta2
            net["pred2"]  = pred_2
            net["local2"] = local_2

        with tf.variable_scope("stage3"):
            s3_conv,s3_fc = fusion_block(s1,stage[2],10,name="s3_stream",pool="max",pkernel=8)
            x3_conv,x3_fc = fusion_block(x1,stage[2],10,name="x3_stream",pool="ave",pkernel=8)
            delta3 = multiply_block(s3_conv, x3_conv, 1, name="delta3", activation="tanh")
            delta3 = tf.squeeze(delta3,[1,2])
            com3 = multiply_block(s3_fc, x3_fc, 2 * stage[2], name="comb", activation="tanh")

            pred_3 = Conv2d(com3, stage[2], [1, 1],name="pred_3")
            pred_3 = tf.nn.relu(pred_3)
            pred_3 = tf.squeeze(pred_3,[1,2])

            local_3 = Conv2d(com3, stage[2], [1, 1],name="local_3")
            local_3 = tf.tanh(local_3)
            local_3 = tf.squeeze(local_3,[1,2])

            c = stage[2]*(stage[2]-1)/2 + lambda_local*tf.reduce_sum(pred_3*local_3)
            c = c/(stage[0]*(1+lambda_d*delta1))/(stage[1]*(1+lambda_d*delta2))/(stage[2]*(1+lambda_d*delta3))

            net["delta3"] = delta3
            net["pred3"]  = pred_3
            net["local3"] = local_3


        pred = (a+b+c)*v
        pred = tf.squeeze(pred,[1])
    return pred,net,[a,b,c]

if __name__=="__main__":
    input = tf.placeholder(tf.float32,[16,64,64,3],name="input")
    net,pred = SSRbackbone(input)
    print("hello")
