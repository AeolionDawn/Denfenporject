#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：ZhangJinYun lbert time:19-9-26

'''
    模型预测标签的三种方式
'''
model=load_model('models/mnist_model_fashion.h5')

data=Setup_fashion_mnist_png()
# data=fashion_mnist()
# evaluation(model,data)
# evaluation_one_png(5)
# print(len(data.test_data))

#keras
print(model.evaluate(data.test_data,data.test_labels,verbose=1)[1])


#tensorflow
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    after_attack_label=model.predict(data.test_data)
    print(after_attack_label.shape)
    print(data.test_labels.shape)
    train_correct_prediction=tf.equal(tf.argmax(after_attack_label,1),tf.argmax(data.test_labels,1))
    train_accuracy=tf.reduce_mean(tf.cast(train_correct_prediction,tf.float32))
    print("this algorithm attack accuracy is",1-sess.run(train_accuracy))#非targeted为什么是1-

#numpy
preds=model.predict(data.test_data,verbose=1)
preds=np.argmax(preds,axis=1)
for i in range(10):
    # print('preds',preds[i])
    # print('test_labels',np.argmax(data.test_labels[i]))
    print(preds[i]==np.argmax(data.test_labels[i]))
    print(preds[i]==data.test_labels[i])
    #0;
    #[1,0,0,0,0]
# print('test_labels',np.argmax(data.test_labels,axis=1))
acc=np.sum(preds==np.argmax(data.test_labels,axis=1))/data.test_labels.shape[0]
print('acc',acc)
