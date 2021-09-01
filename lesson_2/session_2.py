import matplotlib.pyplot as plt
from sklearn import datasets
import time

inf = 99999999


def dist(d1, d2) -> float:
    # 欧几里得距离
    dis = 0
    for index in range(4):
        dis += (d1[index]-d2[index])**2
    return dis**1/2


def KNN(data, label, k=1) -> (list, float):
    # 生成样本
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i, d1 in enumerate(data):
        # 留一法遍历
        dis_list = []
        for j, d2 in enumerate(data):
            dis_list.append(dist(d1, d2) if i != j else inf)
        # 计算分度精度
        matrix_temp_list = [0, 0, 0]
        for _ in range(k):
            matrix_temp_list[label[dis_list.index(min(dis_list))]] += 1
            dis_list[dis_list.index(min(dis_list))] = inf
        matrix[label[i]][matrix_temp_list.index(max(matrix_temp_list))] += 1
    return matrix, (matrix[0][0]+matrix[1][1]+matrix[2][2])/len(data)


if __name__ == '__main__':
    data = datasets.load_iris().data
    label = datasets.load_iris().target

    y = []
    t1 = time.time()
    for K in range(1, 21):
        matrix_list, accuracy = KNN(data, label, K)
        print(f'K={K} matrix={matrix_list} accuracy={accuracy}')
        y.append(accuracy)
    t2 = time.time()
    print(f"耗时:{t2-t1}s")
    plt.plot(list(range(1, 21)), y)
    plt.show()
