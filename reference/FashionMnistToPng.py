#将fashion_mnist二进制文件转换为图片数据集
#好好学习!!!!!!
#对文件的各个形式的调用要烂熟于心,有什么库可以直接调用?不要光靠搜,回想自己已有的资源,同时多读原理!!!!

import cv2
import numpy as np
import os
import gzip

img_path = 'data/fashion/train-images-idx3-ubyte.gz'
label_path = 'data/fashion/train-labels-idx1-ubyte.gz'
save_path = 'data/fashion/train'
img_path2 = 'data/fashion/t10k-images-idx3-ubyte.gz'
label_path2 = 'data/fashion/t10k-labels-idx1-ubyte.gz'
save_path2 = 'data/fashion/test'

classes = {0: 'T-shirt_or_top',
           1: 'Trouser',
           2: 'Pullover',
           3: 'Dress',
           4: 'Coat',
           5: 'Sandal',
           6: 'Shirt',
           7: 'Sneaker',
           8: 'Bag',
           9: 'Ankle boot'}

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

train_data=extract_data(img_path,60000)
print(train_data.shape)
print(train_data[0].shape)
train_labels = extract_labels(label_path, 60000)
test_data=extract_data(img_path2,10000)
test_labels=extract_labels(label_path2,10000)
print(train_labels.shape)
print(train_labels[0])
print(np.argmax(train_labels[0]))
print(np.argmax(train_labels[59999]))
print(test_labels.shape)


def save(img_path, label_path, savepath, num):
    f = open(img_path, 'rb')
    la_f = open(label_path, 'rb')
    la_f.read(8)
    f.read(16)
    dict = {}
    for n in range(num):
        # image = []
        # for i in range(28 * 28):
        #     image.append(ord(f.read(1)))
        # image = np.array(image).reshape(28, 28)
        # name = classes[ord(la_f.read(1))]
        image=test_data[n]
        name = classes[np.argmax(test_labels[n])]
        filepath = os.path.join(savepath, name)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        if name not in dict:
            dict[name] = 1
        else:
            dict[name] += 1
        png = str(dict[name]) + '.png'
        save_path = os.path.join(filepath, png)
        cv2.imwrite(save_path, image)
    la_f.close()
    f.close()


# save(img_path, label_path, save_path, num=60000)
save(img_path2, label_path2, save_path2, num=10000)