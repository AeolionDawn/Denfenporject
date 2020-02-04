"""
    从将本地图片转换为numpy数组格式,并加载为数据集
"""

import os
import tensorflow as tf

from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from PIL import Image
import cv2



def load_sample(sample_dir,shuffleflag = False):
    '''递归读取文件。只支持一级。返回文件名、数值标签、数值对应的标签名'''
    print ('loading sample  dataset..')
    lfilenames = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):#递归遍历文件夹
        for filename in filenames:                            #遍历所有文件名
            #print(dirnames)
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)               #添加文件名
            labelsnames.append( dirpath.split('\\')[-1] )#添加文件名对应的标签

    lab= list(sorted(set(labelsnames)))  #生成标签名称列表
    labdict=dict( zip( lab  ,list(range(len(lab)))  )) #生成字典

    labels = [labdict[i] for i in labelsnames]

    a=np.asarray(lfilenames)
    b=np.asarray(labels)
    c=np.asarray(lab)

    if shuffleflag == True:
        return shuffle(a,b),c
    else:
        return (a,b),c

#单通道图像处理
def img_to_array_L(data_filenames, img_rows, img_cols, channels):
        # 如何将train_filenames转换为numpy数组
        data = np.zeros((len(data_filenames), img_rows, img_cols))
        i = 0
        for data_filename in data_filenames:
                image = Image.open(data_filename)
                image_arr = np.array(image)
                image_arr = (image_arr / 255) - .5
                data[i] = image_arr
                i += 1
        data = np.asarray(data).reshape(-1, img_rows, img_cols, channels).astype(np.float32)

        return data

#3通道图像处理
def img_to_array_RGB(data_filenames,img_rows,img_cols,channels):
        # 如何将train_filenames转换为numpy数组
        data = np.zeros((len(data_filenames), img_rows, img_cols, channels))
        i = 0
        j = 0
        for data_filename in data_filenames:
            image = Image.open(data_filename)
            if image.getbands() == ('L',):
                image = image.convert('RGB')
                j += 1
            image_arr = np.array(image)
            image_arr = (image_arr / 255) - .5
            data[i] = image_arr
            i += 1
        # print('L_model_png:{}'.format(j))
        data = np.asarray(data).reshape(-1, img_rows, img_cols, channels).astype(np.float32)
        print(data.shape)

        return data

class Setup_mnist_png():
        def __init__(self):
            print('加载数据集:')
            img_rows=28
            img_cols=28
            nb_classes=10
            self.channels=1
            sample_dir=r'./data/mnist/mnist_train'
            testsample_dir=r'./data/mnist/mnist_test'
            (train_filenames,train_labels),_ =load_sample(sample_dir,shuffleflag=False) #载入文件名称与标签
            (test_filenames,test_labels),_ =load_sample(testsample_dir,shuffleflag=False) #载入文件名称与标签
            print(train_filenames)
            print(len(train_filenames))

            self.classes=['0','1','2','3','4','5','6','7','8','9']

            train_data=img_to_array_L(train_filenames, img_rows, img_cols, self.channels)
            self.test_data=img_to_array_L(test_filenames, img_rows, img_cols, self.channels)
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            self.test_labels = np_utils.to_categorical(test_labels, nb_classes)

            VALIDATION_SIZE = 5000

            self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
            self.validation_labels = train_labels[:VALIDATION_SIZE]
            self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
            self.train_labels = train_labels[VALIDATION_SIZE:]


            self.weight, self.height, self.nchannels = self.train_data.shape[1:4]
            self.image_size = self.weight
            self.nb_classes = self.train_labels.shape[1]
            print('数据集加载完成')

class Setup_cinic10_png():
        def __init__(self):
            print('加载数据集')
            img_rows = 32
            img_cols = 32
            nb_classes = 10
            self.channels = 3
            sample_dir = r"./data/CINIC-10/train"
            testsample_dir = r"./data/CINIC-10/test"
            validationsample_dir = r'./data/CINIC-10/valid'
            (train_filenames, train_labels), _ = load_sample(sample_dir, shuffleflag=False)  # 载入文件名称与标签
            (test_filenames, test_labels), _ = load_sample(testsample_dir, shuffleflag=False)  # 载入文件名称与标签
            (validation_filenames, validation_labels), _ = load_sample(validationsample_dir,
                                                                       shuffleflag=False)  # 载入文件名称与标签

            self.classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
            self.train_data = img_to_array_RGB(train_filenames, img_rows, img_cols, self.channels)
            self.test_data = img_to_array_RGB(test_filenames, img_rows, img_cols, self.channels)
            self.validation_data=img_to_array_RGB(validation_filenames,img_rows,img_cols,self.channels)
            #labels one_hot
            self.train_labels = np_utils.to_categorical(train_labels, nb_classes)
            self.test_labels = np_utils.to_categorical(test_labels, nb_classes)
            self.validation_labels=np_utils.to_categorical(validation_labels,nb_classes)

            self.weight, self.height, self.nchannels = self.train_data.shape[1:4]
            self.image_size = self.weight
            self.nb_classes = self.train_labels.shape[1]
            print('数据集加载完成')

class Setup_cifar10_png():
    def __init__(self):
            print('加载数据集')
            img_rows = 32
            img_cols = 32
            nb_classes = 10
            self.channels = 3
            sample_dir = r"./data/cifar-10/train"
            testsample_dir = r"./data/cifar-10/test"
            (train_filenames, train_labels), _ = load_sample(sample_dir, shuffleflag=False)  # 载入文件名称与标签
            (test_filenames, test_labels), _ = load_sample(testsample_dir, shuffleflag=False)  # 载入文件名称与标签

            print(train_filenames)
            print(test_filenames)

            self.classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

            train_data = img_to_array_RGB(train_filenames, img_rows, img_cols, self.channels)
            self.test_data = img_to_array_RGB(test_filenames, img_rows, img_cols, self.channels)

            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            self.test_labels = np_utils.to_categorical(test_labels, nb_classes)

            VALIDATION_SIZE = 5000

            self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
            self.validation_labels = train_labels[:VALIDATION_SIZE]
            self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
            self.train_labels = train_labels[VALIDATION_SIZE:]

            self.weight, self.height, self.nchannels = self.train_data.shape[1:4]
            self.image_size = self.weight
            self.nb_classes = self.train_labels.shape[1]
            print('数据集加载完成')


class Setup_fashion_mnist_png():
        def __init__(self):
            print('加载数据集:')
            img_rows=28
            img_cols=28
            nb_classes=10
            self.channels=1
            sample_dir=r'./data/fashion/train'
            testsample_dir=r'./data/fashion/test'
            (train_filenames,train_labels),_ =load_sample(sample_dir,shuffleflag=False) #载入文件名称与标签
            (test_filenames,test_labels),_ =load_sample(testsample_dir,shuffleflag=False) #载入文件名称与标签
            print(train_filenames)
            print(len(train_filenames))

            self.classes=['Ankle boot','Bag','Coat','Dress','Pullover','Sandal','Shirt','Sneaker','T-shirt_or_top','Trouser']

            class_labels={'Ankle boot':0,#短靴
                    'Bag':1,#包
                    'Coat':2,#外套
                    'Dress':3,#裙子
                    'Pullover':4,#套衫
                    'Sandal':5,#凉鞋
                    'Shirt':6,#衬衫
                    'Sneaker':7,#运动鞋
                    'T-shirt_or_top':8,#T恤
                    'Trouser':9#裤子
                    }

            train_data=img_to_array_L(train_filenames, img_rows, img_cols, self.channels)
            self.test_data=img_to_array_L(test_filenames, img_rows, img_cols, self.channels)
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            self.test_labels = np_utils.to_categorical(test_labels, nb_classes)

            VALIDATION_SIZE = 5000

            self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
            self.validation_labels = train_labels[:VALIDATION_SIZE]
            self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
            self.train_labels = train_labels[VALIDATION_SIZE:]


            self.weight, self.height, self.nchannels = self.train_data.shape[1:4]
            self.image_size = self.weight
            self.nb_classes = self.train_labels.shape[1]
            print('数据集加载完成')



# a=Setup_fashion_mnist_png()
# print(a.train_data)
# print(a.train_data.shape)
# print(a.train_labels)
# print(a.train_labels.shape)









