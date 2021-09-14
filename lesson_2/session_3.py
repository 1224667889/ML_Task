"""
参考地址：https://www.freesion.com/article/2645896122/
"""
import pickle
import pickle
import operator
import numpy as np
import pandas as pd
import time


# K近邻算法
class KNN:
    def __init__(self):
        pass

    def train(self, x, y):
        self.xtr = x
        self.ytr = y

    def predict(self, x, k):
        num = x.shape[0]
        Ypred = np.zeros(num)
        for i in range(num):
            # 利用欧式距离
            distance = np.sum((self.xtr - x[i, :])**2, axis=1)**0.5
            # 对距离结果排序，得到从小到大索引
            sortedDistanceIndexs = distance.argsort()
            # k近邻的k循环,统计前k个距离最小的样本
            countDict = {}
            for j in range(k):
                countY = self.ytr[sortedDistanceIndexs[j]]          # 得到前k个从小到大索引的样本类别
                countDict[countY] = countDict.get(countY, 0) + 1    # 统计出现不存在则为0

            # 对前k个距离最小做value排序,找出统计次数最多的类别，作为预测类别
            sortedCountDict = sorted(countDict.items(), key=operator.itemgetter(1), reverse=True)
            Ypred[i] = sortedCountDict[0][0]
        return Ypred


# 数据获取
def unpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def show_dataset(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        print(d[b'data'].shape)
        print(len(d[b'labels']))
        print(len(d[b'filenames']))


if __name__ == '__main__':
    # 展示数据集
    # (10000, 3072)
    # 10000
    # 10000
    show_dataset('F://数据集/cifar-10-batches-py/data_batch_1')

    # KNN对图像集做分类，计算准确率
    top_num = 50
    train_data = dict()
    train_data.update(unpickle(f'F://数据集/cifar-10-batches-py/data_batch_{1}'))
    test_data = unpickle('F://数据集/cifar-10-batches-py/test_batch')

    knn = KNN()
    # 输入数据集和标签
    t1 = time.time()
    knn.train(train_data[b'data'], np.array(train_data[b'labels']))
    t2 = time.time()
    Ypred = knn.predict(test_data[b'data'][:top_num, :], 500)
    accur = np.sum(np.array(Ypred) == np.array(test_data[b'labels'][:top_num])) / len(Ypred)
    t3 = time.time()
    print(f'数据集长度：{len(train_data[b"data"])}')
    print(f'验证集长度：{len(test_data[b"data"])}')
    print(f'训练消耗：{t2-t1}s')
    print(f'预测消耗：{t3-t2}s')
    print(f'准确率：{accur*100}%')
