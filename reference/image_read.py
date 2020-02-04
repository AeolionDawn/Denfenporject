# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-06-22

'''
    图片读取的多种方式,并转换为numpy数组进行运算
    通道数为1的图片读取后通道维度并不会显示
'''
import numpy as np
from keras.models import load_model
import cv2

model=load_model('models/mnist_model.h5')


image_path='images/adv_1.png'
image_path2='images/inputs_1.png'

#法一.cv直接读并转换
ii = cv2.imread(image_path)
print('origin',ii.shape)#(28,28,3)
gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
# print(gray_image)
print('cv2',gray_image.shape)#(28,28)

#法二.cv灰度图读取
def img_read(image_path):
    gray_image_2=cv2.imread(image_path,0)
    print('cv2',gray_image_2.shape)#(28,28)
    gray_image_2 = gray_image_2.astype('float32')
    gray_image_2 /=255
    gray_image_2=np.expand_dims(gray_image_2,axis=-1)
    gray_image_2=np.expand_dims(gray_image_2,axis=0)
    print('cv2_expand',gray_image_2.shape)#(1,28,28,1)
    return gray_image_2

#法三.PIL读取
from PIL import Image
img=np.array(Image.open(image_path).convert('L'),'f')
print('PIL',img.shape)#(28,28)

#法四.keras读取
from keras.preprocessing import image
img_keras=image.load_img(image_path,target_size=(28,28))
img_keras=image.img_to_array(img_keras)
print('keras',img_keras.shape)#(28,28,3)
img_keras=np.expand_dims(img_keras,axis=0)
print('keras_expand',img_keras.shape)#(1,28,28,3)

#法五.scipy读取
from scipy.misc import imread
result = imread(image_path)
print('scipy',result.shape)#(28,28)
result = result.astype('float32')
result /= 255
result = result[:, :, np.newaxis]  # 增添最后一个维度免得不够
print('scipy_expand',result.shape)#(28,28,1)

#拓展一 第一维度拓展
print(image.shape)#(28,28,1)
minibatch =[image]
minibatch=np.array(minibatch)
print("$$$$$$")
print(minibatch.shape)#(1,28,28,1)





img=img_read(image_path)
img2=img_read(image_path2)

# print(model.predict(img))
# print(np.argmax(model.predict(img),axis=1))
#
# print(model.predict(img2))
# print(np.argmax(model.predict(img2),axis=1))

# from dataset_analysis_png import Setup_mnist_png
# data=Setup_mnist_png()
# score=model.evaluate(data.test_data,data.test_labels,verbose=1)
# print('loss',score[0])
# print('acc',score[1])


