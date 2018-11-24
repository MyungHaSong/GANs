import numpy as np
import tensorflow as tf
import os
import wget
import zipfile
import tarfile
import shutil
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




class dataset_download:
    
    @classmethod
    def CycleGAN_download_dataset(cls,dataset):
        '''
        reference : http://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/


        dataset list : 
        1. ae_photoz.zip        2.apple2orange.zip
        2. apple2orange.zip     4.cezanne2photo.zip
        5. cityscapes.zip       6.facades.zip
        7. horse2zebra.zip      8.iphone2dslr_flower.zip     etc...

        paramter : 
        dataset(str) --> dataset name  ex) 'horse2zebra'
        '''

        path = 'datasets/CycleGAN/'
        if not os.path.exists(path):
            os.mkdir(path)

        url = 'http://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'
        if os.path.exists(path+dataset+'.zip'):
            print("Already have done!")
        else:
            wget.download(url+dataset+'.zip',out = path)
        zipfile_ = zipfile.ZipFile(path+dataset+'.zip')
        zipfile_.extractall(path)
        zipfile_.close()
        os.remove(path +dataset+'.zip')
        
    @classmethod
    def tarfile_downloader(cls,url,name,path):
        '''
        download tarfile and extract all file 
        
        parameters :
        url(str) --> dataset url
        name(str) --> tar file name
        path(str) --> Desired route to download

        '''
        wget.download(url,out = path)
        with tarfile.open(path + '/' + name) as tar:
            tar.extractall()
            tar.close()
            
        os.remove(path + '/' + name)