# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-06-22

'''
    模型预测效果/对抗样本效果/防御方法效果 综合测试
'''
import numpy as np
from keras.models import load_model
import cv2
from dataset_analysis_png import Setup_fashion_mnist_png
from dataset_analysis import Setup_mnist_fashion
import numpy as np
import tensorflow as tf
import os

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0   '


# data=Setup_fashion_mnist_png()

from scipy.misc import imread


def img_read(image_path):
    gray_image_2=cv2.imread(image_path,0)
    # print('cv2',gray_image_2.shape)#(28,28)
    gray_image_2 = gray_image_2.astype('float32')
    gray_image_2 =gray_image_2/255-0.5
    gray_image_2=np.expand_dims(gray_image_2,axis=-1)
    gray_image_2=np.expand_dims(gray_image_2,axis=0)
    # print('cv2_expand',gray_image_2.shape)#(1,28,28,1)
    return gray_image_2

def main():
    imge_inputs=[]
    imge_adv=[]
    for i in range(9):
        imge_inputs.append(img_read(inputs_save_dir+'inputs_'+str(i)+'.png'))
        imge_adv.append(img_read(adv_save_dir + 'adv_' + str(i) + '.png'))
        print('origin_labels',np.argmax([i]))
        print('adv_predict_labels', np.argmax(model.predict(imge_adv[i]), axis=1))
        # imge[i]=np.array(imge[i])
        # print(imge[i])

#读取整个数据集进行分类预测
def evaluation(model,data):
    img_inputs=[]
    img_adv=[]
    img_defend=[]

    for i in range(len(data.test_data)):
        img_inputs.append(img_read(inputs_save_dir+'inputs_'+str(i)+'.png'))
        img_adv.append(img_read(adv_save_dir+'adv_'+str(i)+'.png'))
        img_defend.append(img_read(defend_save_dir+'defend_'+str(i)+'.png'))

    img_inputs=np.array(img_inputs)
    img_adv=np.array(img_adv)
    img_defend=np.array(img_defend)
    print(img_inputs.shape)
    print('model acc',model.evaluate(data.test_data,data.test_labels,verbose=1)[1])#91.5%
    print('inputs acc',model.evaluate(img_inputs,data.test_labels,verbose=1)[1])#由test_data转化的图片再读取,应该有91.5%
    print('adv acc',model.evaluate(img_adv,data.test_labels,verbose=1)[1])#22%
    print('defend acc',model.evaluate(img_defend,data.test_labels,verbose=1)[1])#25%

#读取单张或多张图片进行分类预测
def evaluation_one_png(img_number,inputs_save_dir,adv_save_dir,defend_save_dir):

    i=img_number

    inputs = img_read(inputs_save_dir + 'inputs_' + str(i) + '.png')
    adv = img_read(adv_save_dir + 'adv_' + str(i) + '.png')
    defend = img_read(defend_save_dir + 'defend_' + str(i) + '.png')
    # print('inputs_origin_labels',np.argmax(data.test_labels[i]))
    # print('inputs_predict_labels', np.argmax(model.predict(inputs), axis=1))#干净样本预测标签
    print('adv_predict_labels', np.argmax(model.predict(adv), axis=1))  # 对抗样本预测标签
    print('defend_predict_labels', np.argmax(model.predict(defend), axis=1))#防御样本预测标签

    #其他模型产生的对抗样本,对本模型无效,对抗样本迁移性暂不成立
    # adv_test_0=img_read(adv_save_dir_test+'cw_0_0_0_5.png')
    # adv_test_1=img_read(adv_save_dir_test+'cw_0_0_2_2.png')
    # adv_test_2=img_read(adv_save_dir_test+'cw_0_0_3_0.png')
    # adv_test_3=img_read(adv_save_dir_test+'cw_0_0_4_4.png')
    # adv_test_4=img_read(adv_save_dir_test+'cw_0_0_8_0.png')
    # print('adv_predict_labels', np.argmax(model.predict(adv_test_0), axis=1))
    # print('adv_predict_labels', np.argmax(model.predict(adv_test_1), axis=1))
    # print('adv_predict_labels', np.argmax(model.predict(adv_test_2), axis=1))
    # print('adv_predict_labels', np.argmax(model.predict(adv_test_3), axis=1))
    # print('adv_predict_labels', np.argmax(model.predict(adv_test_4), axis=1))
    #
    # adv_test_5=img_read(adv_save_dir_test+'cw_1_2_1_2.png')
    # adv_test_6=img_read(adv_save_dir_test+'cw_1_2_2_3.png')
    # adv_test_7=img_read(adv_save_dir_test+'cw_1_2_3_3.png')
    # print('adv_predict_labels',np.argmax(model.predict(adv_test_5)))
    # print('adv_predict_labels',np.argmax(model.predict(adv_test_6)))
    # print('adv_predict_labels',np.argmax(model.predict(adv_test_7)))

class_labels = {'Ankle boot': 0,  # 短靴
                    'Bag': 1,  # 包
                    'Coat': 2,  # 外套
                    'Dress': 3,  # 裙子
                    'Pullover': 4,  # 套衫
                    'Sandal': 5,  # 凉鞋
                    'Shirt': 6,  # 衬衫
                    'Sneaker': 7,  # 运动鞋
                    'T-shirt_or_top': 8,  # T恤
                    'Trouser': 9  # 裤子
                    }

if __name__=='__main__':
    #注意模型,必须与数据集相匹配
    model=load_model('models/mnist_model.h5')

    inputs_save_dir = 'images_inputs/'
    inputs_save_dir_temp = 'images_inputs_temp/'
    adv_save_dir = 'images_adv/'
    adv_save_dir_temp='adv_images_cw/'
    defend_save_dir = 'images_defend/'
    defend_save_dir_temp='images_defend_temp/'

    adv_save_dir_test='advimgs_test/'


    # data=Setup_fashion_mnist_png()

    # evaluation(model,data)

    #对adv_images_cw/对抗样本的测试
    for i in range(9):
        evaluation_one_png(i,inputs_save_dir,adv_save_dir,defend_save_dir_temp)





