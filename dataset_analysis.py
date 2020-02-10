#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：ZhangJinYun lbert time:19-7-10
#暂时不用该数据集处理了

import numpy as np
import  gzip
import zipfile
from tensorflow.examples.tutorials.mnist import input_data
import array
import functools
import gzip
import operator
import os
import struct
import tempfile
import sys
import tensorflow as tf

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


utils=tf.keras.utils
# def extract_data(filename, num_images):
#     with gzip.open(filename) as bytestream:
#         bytestream.read(16)
#         buf = bytestream.read(num_images*28*28)
#         data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
#         data = (data / 255) - 0.5
#         data = data.reshape(num_images, 28, 28, 1)
#         return data
#
# def extract_labels(filename, num_images):
#     with gzip.open(filename) as bytestream:
#         bytestream.read(8)
#         buf = bytestream.read(1 * num_images)
#         labels = np.frombuffer(buf, dtype=np.uint8)
#     return (np.arange(10) == labels[:, None]).astype(np.float32)


def load_batch(fpath):
    f = open(fpath, "rb").read()
    size = 32 * 32 * 3 + 1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i * size:(i + 1) * size], dtype=np.uint8)
        lab = np.identity(10)[arr[0]]#能够直接转换为one-hot编码
        img = arr[1:].reshape((3, 32, 32)).transpose((1, 2, 0))

        labels.append(lab)
        images.append((img / 255) - .5)
    return np.array(images), np.array(labels)

def unzip(dataset_path):
    zfile=zipfile.ZipFile(dataset_path,'r')
    for fileM in zfile.namelist():
        zfile.extract(fileM,'images/')

    zfile.close()

def maybe_download_file(url, datadir=None, force=False):

  try:
    from urllib.request import urlretrieve
  except ImportError:
    from urllib import urlretrieve
    #url下载文件地址，dest_file下载到本地的地址
  def reporthook(a, b, c):
    """
            显示下载进度
            :param a: 已经下载的数据块
            :param b: 数据块的大小
            :param c: 远程文件大小
            :return: None
            """
    print("\rdownloading: %5.1f%%" % (a * b * 100.0 / c), end="")
  if not datadir:
    datadir = tempfile.gettempdir()
  file_name = url[url.rfind("/")+1:]
  dest_file = os.path.join(datadir, file_name)

  #用isfile来判断目录下是否有已下载文件就行了
  isfile = os.path.isfile(dest_file)

  if force or not isfile:
    urlretrieve(url, dest_file,reporthook=reporthook)
  return dest_file


def download_and_parse_mnist_file(file_name, datadir=None, force=False):

  url = os.path.join('http://yann.lecun.com/exdb/mnist/', file_name)
  file_name = maybe_download_file(url, datadir=datadir, force=force)

  # Open the file and unzip it if necessary
  if os.path.splitext(file_name)[1] == '.gz':
    open_fn = gzip.open
  else:
    open_fn = open

  # Parse the file
  with open_fn(file_name, 'rb') as file_descriptor:
    header = file_descriptor.read(4)
    assert len(header) == 4

    zeros, data_type, n_dims = struct.unpack('>HBB', header)
    assert zeros == 0

    hex_to_data_type = {
        0x08: 'B',
        0x09: 'b',
        0x0b: 'h',
        0x0c: 'i',
        0x0d: 'f',
        0x0e: 'd'}
    data_type = hex_to_data_type[data_type]

    # data_type unicode to ascii conversion (Python2 fix)
    if sys.version_info[0] < 3:
      data_type = data_type.encode('ascii', 'ignore')

    dim_sizes = struct.unpack(
        '>' + 'I' * n_dims,
        file_descriptor.read(4 * n_dims))

    data = array.array(data_type, file_descriptor.read())
    data.byteswap()

    desired_items = functools.reduce(operator.mul, dim_sizes)
    assert len(data) == desired_items
    return np.array(data).reshape(dim_sizes)

class Setup_cifar10():
    def __init__(self):
        train_data = []
        train_labels = []

        # if not os.path.exists("cifar-10-batches-bin"):
        #     urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        #                                "cifar-data.tar.gz")
        #     os.popen("tar -xzf cifar-data.tar.gz").read()

        for i in range(5):
            r, s = load_batch("data/cifar-10-batches-bin/data_batch_" + str(i + 1) + ".bin")
            train_data.extend(r)
            train_labels.extend(s)

        train_data = np.array(train_data, dtype=np.float32)
        train_labels = np.array(train_labels)

        self.test_data, self.test_labels = load_batch("data/cifar-10-batches-bin/test_batch.bin")

        VALIDATION_SIZE = 5000

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

        self.weight, self.height, self.nchannels = self.train_data.shape[1:4]
        self.image_size = self.weight
        self.nb_classes = self.train_labels.shape[1]

class Setup_mnist():
    def __init__(self,datadir='data',train_start=0,train_end=60000,test_start=0,test_end=10000):
        # train_data = extract_data("data/train-images.idx3-ubyte.gz", 60000)
        # train_labels = extract_labels("data/train-labels.idx1-ubyte.gz", 60000)
        # test_data = extract_data("data/t10k-images.idx3-ubyte.gz", 10000)
        # test_labels = extract_labels("data/t10k-labels.idx1-ubyte.gz", 10000)


        """
          Load and preprocess MNIST dataset
          :param datadir: path to folder where data should be stored
          :param train_start: index of first training set example
          :param train_end: index of last training set example
          :param test_start: index of first test set example
          :param test_end: index of last test set example
          :return: tuple of four arrays containing training data, training labels,
                   testing data and testing labels.
          """
        assert isinstance(train_start, int)
        assert isinstance(train_end, int)
        assert isinstance(test_start, int)
        assert isinstance(test_end, int)

        X_train = download_and_parse_mnist_file(
            'train-images.idx3-ubyte.gz', datadir=datadir) / 255.
        Y_train = download_and_parse_mnist_file(
            'train-labels.idx1-ubyte.gz', datadir=datadir)
        X_test = download_and_parse_mnist_file(
            't10k-images.idx3-ubyte.gz', datadir=datadir) / 255.
        Y_test = download_and_parse_mnist_file(
            't10k-labels.idx1-ubyte.gz', datadir=datadir)

        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

        Y_train = utils.to_categorical(Y_train, 10)
        Y_test = utils.to_categorical(Y_test, 10)

        train_data = X_train.astype('float32')
        train_labels = Y_train.astype('float32')
        test_data = X_test.astype('float32')
        test_labels = Y_test.astype('float32')

        self.tag='mnist'
        self.classes=['0','1','2','3','4','5','6','7','8','9']

        # VALIDATION_SIZE = 5000
        #
        # self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        # self.validation_labels = train_labels[:VALIDATION_SIZE]
        # self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        # self.train_labels = train_labels[VALIDATION_SIZE:]

        self.validation_data=test_data
        self.validation_labels=test_labels

        self.train_data=train_data[train_start:train_end]
        self.train_labels=train_labels[train_start:train_end]
        self.test_data=test_data[test_start:test_end]
        self.test_labels=test_labels[test_start:test_end]

        # confirm load
        print('Training data has {} rows'.format(train_data.shape[0]))
        print('Test data has {} rows'.format(test_data.shape[0]))

        self.weight, self.height, self.nchannels = self.train_data.shape[1:4]
        self.image_size = self.weight
        self.nb_classes = self.train_labels.shape[1]

class Setup_mnist_fashion():
    def __init__(self,train_start=0,train_end=60000,test_start=0,test_end=10000):
        # train_data = extract_data("data/fashion/train-images-idx3-ubyte.gz", 55000)
        # train_labels = extract_labels("data/fashion/train-labels-idx1-ubyte.gz", 55000)
        # self.test_data = extract_data("data/fashion/t10k-images-idx3-ubyte.gz", 10000)
        # self.test_labels = extract_labels("data/fashion/t10k-labels-idx1-ubyte.gz", 10000)
        #
        # VALIDATION_SIZE = 5000
        #
        # self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        # self.validation_labels = train_labels[:VALIDATION_SIZE]
        # self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        # self.train_labels = train_labels[VALIDATION_SIZE:]
        #
        # self.weight, self.height, self.nchannels = self.train_data.shape[1:4]
        # self.image_size = self.weight
        # self.nb_classes = self.train_labels.shape[1]

        data = input_data.read_data_sets('data/fashion')

        train_data,train_labels = data.train.images, data.train.labels
        test_data, test_labels = data.test.images, data.test.labels


        # reshape data to add colour channel
        img_rows, img_cols = 28, 28

        self.tag='mnist_fashion'
        self.classes = ['T-shirt/top',
                       'Trouser',
                       'Pullover',
                       'Dress',
                       'Coat',
                       'Sandal',
                       'Shirt',
                       'Sneaker',
                       'Bag',
                       'Ankle boot']

        #注意与dataset_analysis_png中对应的标签是完全不一致的
        class_label = {'T-shirt/top':0,
                       'Trouser':1,
                       'Pullover':2,
                       'Dress':3,
                       'Coat':4,
                       'Sandal':5,
                       'Shirt':6,
                       'Sneaker':7,
                       'Bag':8,
                       'Ankle boot':9}

        train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1).astype('float32')
        test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1).astype('float32')
        # confirm data reshape
        # print('Training data shape: {}'.format(train_data.shape))
        # print('Test data shape: {}'.format(test_data.shape))

        self.train_data = train_data[train_start:train_end]
        self.train_labels = train_labels[train_start:train_end]
        self.test_data = test_data[test_start:test_end]
        self.test_labels = test_labels[test_start:test_end]

        # confirm load
        print('Training data has {} rows'.format(train_data.shape[0]))
        print('Test data has {} rows'.format(test_data.shape[0]))

        # one-hot encode outputs
        self.nb_classes = 10
        self.train_labels= np_utils.to_categorical(train_labels, self.nb_classes)
        self.test_labels = np_utils.to_categorical(test_labels, self.nb_classes)


        self.validation_data = self.test_data
        self.validation_labels = self.test_labels

        self.weight, self.height, self.nchannels = self.train_data.shape[1:4]
        self.image_size = self.weight

class Setup_cinic10():
    def __init__(self):
        batch_size=32
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        class_label = ['airplane', 'automobile', 'brid', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.train_generator = train_datagen.flow_from_directory('data/CINIC-10/train',
                                                            target_size=(32, 32),
                                                            batch_size=batch_size,
                                                            class_mode='categorical')

        self.validation_generator = test_datagen.flow_from_directory('data/CINIC-10/valid',
                                                                target_size=(32, 32),
                                                                batch_size=batch_size,
                                                                class_mode='categorical')

        self.test_generator = test_datagen.flow_from_directory('data/CINIC-10/test',
                                                          target_size=(32, 32),
                                                          batch_size=batch_size,
                                                          class_mode='categorical')

        self.nb_classes = 10

        self.weight, self.height, self.nchannels =(32,32,3)
        self.image_size = self.weight



