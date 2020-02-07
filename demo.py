import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request
from utils import pause
from utils import press_any_key_exit

# from keras.models import load_model

from distillation import train_distillation
from adversarial_training import Adversarial_training

# 选取数据集
import dataset_analysis as da
import dataset_analysis_png as dap

from BIM import BIM
from fgsm import fgsm
from model_evaluation import evaluation
import  UI


from l2_attack import CarliniL2
import cv2
import random

from tensorflow.python.platform import app, flags
FLAGS=flags.FLAGS

import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

load_model=tf.keras.models.load_model

#direct read the trained model
MODEL_PATH="models_test/tf_keras_mnist_model.h5"

#hyper-parameter for distillation
NB_EPOCHS_1 = 1
BATCH_SIZE_1 = 128
LEARNING_RATE_1 = .001
TRAIN_TEMP=100

#hyper-parameter for adversarial training
NB_EPOCHS_2 = 5
BATCH_SIZE_2 = 128
LEARNING_RATE_2 = .001

#hyper-parameter for model compression

#hyper-parameter for HGD


#模型的选择和加载
def model_select(model_path):
    print("加载模型文件:")
    print('the model path is:%s' % (model_path))
    pause()
    model = load_model(model_path)
    print("加载完成")

    return  model

def model_robust_strengthen(model,model_path,dataset):
    print("请选择神经网络鲁棒性增强的方法:")
    print("1:蒸馏防御方法\n")
    print("2:对抗训练方法\n")
    print("3:图像压缩方法\n")
    print('4:退出系统\n')
    while True:
        choice=input('请输入对应序号：')
        if choice=='1':
            print("开始对模型进行蒸馏防御增强")
            defenseModel,model_up_time=train_distillation(model, dataset, model_path, nb_epochs=FLAGS.nb_epochs_1, train_temp=FLAGS.train_temp)
            return defenseModel,model_up_time
        elif choice=='2':
            print("开始对模型进行对抗训练防御增强")
            Adversarial_training(model, dataset, learning_rate=FLAGS.learning_rate_2, batch_size=FLAGS.batch_size_2, nb_epochs=FLAGS.nb_epochs_2)
            break
        elif choice=='3':
            print("开始对对抗样本进行comdefend图像压缩")
            break
        elif choice=='4':
            print("不进行鲁棒性增强,退出程序")
            os._exit(0)
        else :
            print("选择错误!请重新选择")

# def evaluation(sess,model,adv,labels,targets,targeted=True,imag_tensor=None):
#     if imag_tensor !=None:
#         pass
#     else:
#         after_attack_label=model.predict(adv)
#         train_correct_prediction=tf.equal(tf.argmax(after_attack_label),tf.argmax(targets,1))
#         train_accuracy=tf.reduce_mean(tf.cast(train_correct_prediction,tf.float32))
#         if targeted:
#             print("this algorithm attack accuracy is",sess.run(train_accuracy))
#         else:
#             print("this algorithm attack accuracy is",1-sess.run(train_accuracy))


def main(args=None):
    print("进行神经网络鲁棒性增强")
    pause()
    press_any_key_exit("任意键以继续\n")

    #模型准备
    model = model_select(FLAGS.model_path)  # 输入模型路径

    #如果重新训练模型

        #数据集准备
    # dataset_1=Dataset_select(dap.Setup_mnist_fashion())#数据集处理方法修改,输入数据集路径
    # dataset=Dataset_select(da.Setup_mnist())#选择数据集
    data = da.Setup_mnist(train_start=0, train_end=30000, test_start=0, test_end=10000)
    dataset_test = da.Setup_mnist(train_start=0, train_end=60000, test_start=0, test_end=10000)
        #模型再训练
    model_defend_display,model_up_time=model_robust_strengthen(model, FLAGS.model_path, data)#模型防御方法选择

    #如果不训练，单独读取某个以训练好的模型
    # model_path=Flags.model_path
    # model_defend = load_model(model_path)

    #测试防御效果
    # eval = evaluation(Model.model, dataset_test, model_defend, fgsm)

    # 界面展示
    # UI.defense_display(dataset_test,eval.preds1,eval.preds2,model_up_time,eval.x_adv1)
    # UI.defense_display(dataset_test,eval.preds1,eval.preds2,eval.x_adv1)


if __name__=="__main__":

    flags.DEFINE_string('model_path', MODEL_PATH,
                       'direct read the trained model')

    flags.DEFINE_integer('nb_epochs_1', NB_EPOCHS_1,
                         'Number of epochs to train model')
    flags.DEFINE_integer('batch_size_1', BATCH_SIZE_1, 'Size of training batches')
    flags.DEFINE_float('learning_rate_1', LEARNING_RATE_1,
                       'Learning rate for training')
    flags.DEFINE_integer('train_temp', TRAIN_TEMP,
                         'Number of epochs to train model')

    flags.DEFINE_integer('nb_epochs_2', NB_EPOCHS_2,
                         'Number of epochs to train model')
    flags.DEFINE_integer('batch_size_2', BATCH_SIZE_2, 'Size of training batches')
    flags.DEFINE_float('learning_rate_2', LEARNING_RATE_2,
                       'Learning rate for training')

    tf.app.run()


