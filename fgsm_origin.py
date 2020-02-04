#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：ZhangJinYun lbert time:19-10-26


model = model
epochs = 100
epsilon = 0.1
prev_probs = []

x_adv = dataset.test_data  # 测试统一使用test_data
x_noise = np.zeros_like(dataset.test_data)  # 给一个数组,生成一个shape相同的全0数组
sess = K.get_session()

initial_class = np.argmax(model.predict(dataset.test_data), axis=1)
timestart = time.time()

for i in range(epochs):
    # One hot encode the initial class
    target = K.one_hot(initial_class, dataset.nb_classes)

    # Get the loss and gradient of the loss wrt the inputs
    loss = K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.input)

    # Get the sign of the gradient
    delta = K.sign(grads[0])
    x_noise = x_noise + epsilon * delta

    # Perturb the image
    x_adv = x_adv + epsilon * delta

    # Get the new image and predictions
    x_adv = sess.run(x_adv, feed_dict={model.input: dataset.test_data})

    preds = model.predict(x_adv)
    # Store the probability of the target class
    # prev_probs.append(preds[0][initial_class])
    #
    # # preds = np.argmax(preds, axis=1)
    # if (i + 1) % 10 == 0:
    #     print("经过" + str(i + 1) + " 轮攻击后 : ")
    #     print("使用fgsm对抗样本攻击后的分类准确率:{:.2f}\n".format(np.sum(preds == dataset.test_labels) / dataset.test_labels.size))

    if (i + 1) % 10 == 0:
        score = model.evaluate(x_adv, dataset.test_labels)
        print("经过" + str(i + 1) + " 轮攻击后 : ")
        print('损失函数为:{:.2f}\n'.format(score[0]))
        print("使用fgsm对抗样本攻击后的分类准确率:{:.2f}\n".format(score[1]))

timeend = time.time()
print("花费", (timeend - timestart) / 60, "分钟完成", epochs, "轮攻击次数.")
print('平均每次攻击花费:', (timeend - timestart) / epochs, '秒')
