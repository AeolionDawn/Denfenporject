#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：ZhangJinYun lbert time:19-7-6


from tensorflow.python import keras


import tensorflow as tf
from setup_mnist import MNIST
import os
from l2_attack import CarliniL2
import cv2
import time
import random
from keras.utils import np_utils

import os
import warnings

warnings.filterwarnings('ignore')

def train_distillation(model, data, file_name, num_epochs=10, batch_size=128, train_temp=1):
    # now train the teacher at the given temperature
    print("开始训练教师模型:")
    timestart=time.time()
    teacher = train_again(model, data, file_name.replace('.h5', "_teacher_"+str(num_epochs)+'.h5'), num_epochs, batch_size, train_temp)

    # evaluate the labels at temperature t
    predicted = teacher.predict(data.train_data)
    with tf.Session() as sess:
        y = sess.run(tf.nn.softmax(predicted/train_temp))#软目标
        # print(y)
        data.train_labels = y

    # train the student model at temperature t
    print("开始训练学生模型:")
    print('蒸馏温度：',train_temp)

    file_name_new=file_name.replace('.h5', "_student_" + str(num_epochs) + '.h5')
    student =train_again (model, data, file_name_new, num_epochs, batch_size, train_temp)#直接用软目标训练学生网络并进行预测

    timeend=time.time()
    # and finally we predict at temperature 1
    # predicted = student.predict(data.train_data)
    print("增强完成,输出增强后的模型为{}".format(file_name_new))
    time_consume=round((timeend - timestart)/60,2)
    print("模型增强所用时间为", time_consume, "分钟")


    return student,time_consume

#由用户选择是否对模型进行进一步评估
    # print('是否对该模型进行评估')
    # while (True):
    #     choice = input('输入yes或no:')
    #     if choice == 'yes':
    #         #using fgsm to evaluate the model
    #         print('采用fgsm攻击算法对模型进行测试')
    #         print('对该模型进行评估:')
    #         print('1.原模型干净样本评估')
    #         score =model_select.model.evaluate(data.test_data, data.test_labels, verbose=1)
    #         print('测试的损失函数:', score[0])
    #         print('测试的分类准确率:', score[1])
    #
    #         print('2.原模型对抗样本评估')
    #         # evaluation(model_select.model,data)
    #         fgsm(model_select.model,data)
    #
    #         print('3.原模型增强后干净样本评估')
    #         score =student.evaluate(data.test_data, data.test_labels, verbose=1)
    #         print('测试的损失函数:', score[0])
    #         print('测试的分类准确率:', score[1])
    #
    #         print('4.原模型增强后对抗样本模型评估')
    #         # evaluation(student,data)
    #         fgsm(model_select.model,data)
    #         break
    #     elif choice == 'no':
    #         print('不评估,退出系统')
    #         os._exit(0)
    #     else:
    #         print('请输入正确的选择!')

def train_again(model,data, file_name, num_epochs=50, batch_size=128, train_temp=1):

    # model = Model(model.input, model.layer[]);

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    # model.fit_generator(data.train_generator,
    #           steps_per_epoch=len(data.train_generator)//batch_size,
    #           validation_data=data.validation_generator,
    #           verbose=1,
    #           epochs=num_epochs,
    #           shuffle=True)
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              verbose=1,
              nb_epoch=num_epochs,
              shuffle=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    if file_name != None:
        model.save(file_name)
        pass
    return model




