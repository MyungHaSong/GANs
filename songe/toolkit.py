import numpy as np
import tensorflow as tf
import os

class preprocessing:
    '''
image preprocessing toolkit
    '''
    @classmethod
    def normalize(cls,im):
        '''
    image normalization 
        '''
        return im * (2.0 / 255.0) - 1
    @classmethod
    def denormalize(cls,im):
        '''
    image denormalization
        '''
        return (im + 1.) / 2.  
    

