import tensorflow as tf
import scipy
import numpy as np


def lrelu(x, leak = 0.2):
    return tf.maximum(x, x*leak)

def conv_concat(x,y):
    bs = tf.shape(y)[0]
    return tf.concat([x,y*tf.ones(shape = [bs,28,28,10])],axis = 3)

def inverse_transform(images):
        return (images+1.)/2.
def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path,image)
    
def save_images(images, size, image_path):
    return imsave(inverse_transform(images),size,image_path)
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')
        
def conv(x,filters,kernel_size,strides,name, padding="VALID"):
    return tf.layers.conv2d(inputs = x,
                                       filters = filters,
                                       kernel_size = (kernel_size,kernel_size),
                                       strides = (strides,strides), kernel_initializer= tf.truncated_normal_initializer(stddev = 0.02),
                                       name = name)
def dense(x,units,name):
        return tf.layers.dense(inputs = x, 
                               units = units,
                              kernel_initializer = tf.truncated_normal_initializer(stddev = 0.02),
                              name = name)
def conv_trans(x,filters,kernel_size,strides,name, padding="SAME"):
    return tf.layers.conv2d_transpose(inputs = x,
                                       filters = filters,
                                       kernel_size = (kernel_size,kernel_size),
                                       strides = (strides,strides),
                                       kernel_initializer = tf.truncated_normal_initializer(stddev = 0.02),
                                       padding= padding ,
                                       name = name)

def bn(x,is_train):
    return tf.layers.batch_normalization(inputs = x, training=is_train)