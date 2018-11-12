import tensorflow as tf

class tf_toolkit:
    '''
    this toolkit is tensorflow layer module 
    '''
    @classmethod
    def conv(cls,x,filters,k_size,s_size,name,padding = "SAME", **k):
        '''
        parameter :
        filters : number of filter
        k_size : kernel size
        s_size : stride size
        name : layer name
        padding : "SAME" is True --> padding, "VALID" : is False --> No padding
        '''
        net = tf.layers.conv2d(inputs = x,filters=filters,
                               kernel_size=(k_size,k_size),
                               strides = (s_size,s_size),
                               padding = padding,
                               name = name,
                               **k)
        
        return net
        
    @classmethod
    def conv_tran(cls,x,filters,k_size, s_size,name, padding = "SAME",**k):
        '''
        parameter :
        filters : number of filter
        k_size : kernel size
        s_size : stride size
        name : layer name
        padding : "SAME" is True --> padding, "VALID" : is False --> No padding
        '''
        net = tf.layers.conv2d_transpose(inputs = x,
                                         filters = filters, 
                                         kernel_size=(k_size,k_size),
                                         strides = (s_size,s_size),
                                         padding = padding,
                                         name = name,
                                         **k)
        
        return net
    
    @classmethod
    def batch_norm(cls,x,is_train = True, **k):
        return tf.layers.batch_normalization(inputs=x, 
                                             training=is_train,
                                             **k)
    
    
class pix2pix_tf_toolkit:
    @classmethod
    def conv(cls,x,filters,k_size,s_size,name,padding = "SAME", bn = True,**k):
        '''
        convolution layer adding BN parameter
        if bn is True --> batch_normalization
           bn is False --> no_batch_normalization
        '''
        
        net = tf.layers.conv2d(inputs = x,filters=filters,
                               kernel_size=(k_size,k_size),
                               strides = (s_size,s_size),
                               padding = padding,
                               name = name,
                               **k)
        if bn :
            net = tf_toolkit.batch_norm(net)
        return net
    @classmethod
    def conv_tran(cls,x,filters,k_size, s_size,name, padding = "SAME", bn = True,**k):
        '''
        convolution layer adding BN parameter
        if bn is True --> batch_normalization
           bn is False --> no_batch_normalization
        '''
        net = tf.layers.conv2d_transpose(inputs = x,
                                         filters = filters, 
                                         kernel_size=(k_size,k_size),
                                         strides = (s_size,s_size),
                                         padding = padding,
                                         name = name,
                                         **k)
        if bn :
            net = tf_toolkit.batch_norm(net)
        return net

class activation :
    '''
    tensorflow activation function
    '''
       
    @classmethod
    def relu(cls, x):
        return tf.nn.relu(x)
    @classmethod
    def tanh(cls,x):
        return tf.nn.tanh(x)
    @classmethod
    def lrelu(cls, x ,leak = 0.2):
        return tf.maximum(x,leak*x)
    @classmethod
    def sigmoid(cls,x):
        return tf.nn.sigmoid(x)
    

    
