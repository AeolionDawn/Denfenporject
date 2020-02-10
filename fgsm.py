#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：ZhangJinYun lbert time:19-9-25

'''
    用fgsm在fashion_mnist上进行一轮攻击并保存图片
'''
import time
import keras.backend as K
import numpy as np
import os
import cv2
import dataset_analysis_png as dap
import dataset_analysis as da
from keras.models import load_model
from PIL import Image

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'



def img_read(data_dir):
    data_filenames=[]
    for j in range(0,8):
        data_filenames.append(data_dir+str(j+1)+'.png')
    print(data_filenames)

    data = np.zeros((img_number, img_rows, img_cols))
    i=0
    for data_filename in data_filenames:
        image = Image.open(data_filename)
        image_arr = np.array(image)
        image_arr = (image_arr / 255) - .5
        data[i] = image_arr
        i+=1
    data = np.asarray(data).reshape(-1, img_rows, img_cols, channels).astype(np.float32)
    return data

def img_predict(model,data):
        preds = model.predict(data)
        preds = np.argmax(preds, axis=1)
        print('{}'.format(preds))

def fgsm(model,dataset):

        model = model
        epsilon = 0.3
        prev_probs = []

        data=dataset.test_data
        x_adv=data
        x_noise = np.zeros_like(data)  # 给一个数组,生成一个shape相同的全0数组
        sess = K.get_session()

        initial_class = np.argmax(model.predict(data), axis=1)
        timestart = time.time()

        #单步FGSM攻击
        # One hot encode the initial class
        # target = K.one_hot(initial_class, 10)

        target = K.one_hot(initial_class, dataset.nb_classes)

        # Get the loss and gradient of the loss wrt the inputs
        loss = K.categorical_crossentropy(target, model.output)
        grads = K.gradients(loss, model.input)

        # Get the sign of the gradient
        delta = K.sign(grads[0])
        x_noise = x_noise + epsilon * delta

        # Perturb the image
        x_adv = x_adv+ epsilon * delta

        # Get the new image and predictions
        x_adv = sess.run(x_adv, feed_dict={model.input: data})

        preds = model.predict(x_adv)
        # Store the probability of the target class
        # prev_probs.append(preds[0][initial_class])
        #
        preds = np.argmax(preds, axis=1)

        # success_indices=active_indices[np.where(adv_label[active_indices]!=target_label[active_indices][0])]
        # active_indices=active_indices[np.where(adv_label[active_indices]==target_label[active_indices][0])]

        # print(np.argmax(model.predict(x_adv),axis=1))
        # print("使用fgsm对抗样本攻击后的分类准确率:{:.2f}\n".format(np.sum(preds == dataset.test_labels) / dataset.test_labels.size))

        score = model.evaluate(x_adv, dataset.test_labels)
        loss=score[0]
        print('损失函数为:{:.2f}\n'.format(loss))
        acc=score[1]
        print("使用fgsm对抗样本攻击后的分类准确率:{:.2f}\n".format(acc))

        #保存对抗样本图片
        # choice=input('是否保存对抗样本图片,yes 或 no:')
        # if choice=='yes':
        #     adv_save_dir = './images/images_adv_test/'
        #     # if not os.path.isdir(adv_save_dir):
        #     #     os.makedirs(adv_save_dir)
        #
        #     for i in range(0, len(x_adv)):
        #         cv2.imwrite(adv_save_dir + str(i + 1) + '.png', (x_adv[i] + 0.5) * 255)

        timeend = time.time()
        print("花费", round((timeend - timestart) / 60,2), "分钟完成1轮攻击次数.")
        timeConsume=round((timeend - timestart), 2)
        print('平均每次攻击花费:', timeConsume, '秒')

        return x_adv,preds,loss,acc,timeConsume

# if __name__=='__main__':
#     img_rows = 28
#     img_cols = 28
#     channels = 1
#     img_number = 8
#
#     # inputs_save_dir = './images/temp/images_inputs_temp/mnist_train_'
#     # adv_save_dir_1 = './images/images_adv_test/'
#     # adv_save_dir_2 = './images/temp/images_adv_temp/adv_'
#
#     #数据集
#     data=da.Setup_mnist(test_start=0,test_end=10000)
#
#     #单张图片测试
#     # data=img_read(inputs_save_dir)
#
#     model=load_model('models/keras_mnist_trained_model.h5')
#     model_defend=load_model('models/keras_mnist_trained_model_student_20.h5')
#     # print('原样本分类正确率',model.evaluate(data.test_data,data.test_labels)[1])
#     # if not os.path.isdir(adv_save_dir):
#     #     os.makedirs(adv_save_dir)
#     # data1=img_read(adv_save_dir_1)
#     # data2=img_read(adv_save_dir_2)
#     x_adv,preds,loss,acc,timeConsume=fgsm(model, data)
#     print('----------------------')
#     x_adv1,preds1,loss1,acc1,timeConsume1=fgsm(model_defend,data)
#     # img_predict(model,data1)
#     # img_predict(model,data2)
