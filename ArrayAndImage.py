# encoding:utf-8
"""
关于图片的一些操作：
①图片转化为数组并存为二进制文件；
②从二进制文件中读取数据并重新恢复为图片
"""

from __future__ import print_function
import numpy as np
import PIL.Image as Image
import pickle as p
import matplotlib.pyplot as pyplot


class Operation(object):
    image_base_path = "images/"
    data_base_path = "data/"

    def image_to_array(self, filenames):
        """
        图片转化为数组并存为二进制文件；
        :param filenames:文件列表
        :return:
        """
        n = filenames.__len__()  # 获取图片的个数
        result = np.array([])  # 创建一个空的一维数组
        print("开始将图片转为数组")
        for i in range(n):
            image = Image.open(self.image_base_path + filenames[i])
            r, g, b = image.split()  # rgb通道分离
            # 注意：下面一定要reshpae(1024)使其变为一维数组，否则拼接的数据会出现错误，导致无法恢复图片
            r_arr = np.array(r).reshape(1024)
            g_arr = np.array(g).reshape(1024)
            b_arr = np.array(b).reshape(1024)
            # 行拼接，类似于接火车；最终结果：共n行，一行3072列，为一张图片的rgb值
            image_arr = np.concatenate((r_arr, g_arr, b_arr))
            result = np.concatenate((result, image_arr))

        result = result.reshape((n, 3072))  # 将一维数组转化为count行3072列的二维数组
        print("转为数组成功，开始保存到文件")
        file_path = self.data_base_path + "data2.bin"
        with open(file_path, mode='wb') as f:
            p.dump(result, f)
        print("保存文件成功")

    def array_to_image(self, filename):
        """
        从二进制文件中读取数据并重新恢复为图片
        :param filename:
        :return:
        """
        with open(self.data_base_path + filename, mode='rb') as f:
            arr = p.load(f)  # 加载并反序列化数据
        rows = arr.shape[0]
        arr = arr.reshape(rows, 3, 32, 32)
        for index in range(rows):
            a = arr[index]
            # 得到RGB通道
            r = Image.fromarray(a[0]).convert('L')
            g = Image.fromarray(a[1]).convert('L')
            b = Image.fromarray(a[2]).convert('L')
            image = Image.merge("RGB", (r, g, b))
            # 显示图片
            pyplot.imshow(image)
            pyplot.show()
            image.save(self.image_base_path + "result" + str(index) + ".png", 'png')

if __name__ == "__main__":
    my_operator = Operation()
    images = []
    for j in range(5):
        images.append(str(j) + ".png")
    my_operator.image_to_array(images)
    my_operator.array_to_image("data2.bin")