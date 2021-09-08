import numpy
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt

# 老师提供的三个参数计算函数↓
from lesson_3.face_images.clustering_performance import clusteringMetrics
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

X, y = datasets.make_blobs(
    n_samples=400,
    shuffle=True,
)


# 画出初始图像
def draw_init(figure: plt.Figure, location=121) -> None:
    plot = figure.add_subplot(location)
    plot.set_title('data by make_blobs()')
    plot.scatter(X[:, 0], X[:, 1], c=y)
    figure.savefig(f"blob/0.png")


# 画出KMS聚类后图像 - 调用库
def draw_kms(figure: plt.Figure, location=122) -> None:
    plot = figure.add_subplot(location)
    plot.set_title('data by make_blobs()')

    # n_cluster聚类中心数 max_iter最大迭代次数
    kms = KMeans(n_clusters=3, max_iter=1000)
    # 预测样本类型
    y_sample = kms.fit_predict(X, y)
    # 获取聚类中心
    centerPoints = kms.cluster_centers_

    plot.scatter(X[:, 0], X[:, 1], c=y)
    plot.scatter(centerPoints[:, 0], centerPoints[:, 1], s=100, marker='*', c='b')
    ACC = accuracy_score(y, y_sample)
    NMI = normalized_mutual_info_score(y, y_sample)
    ARI = adjusted_rand_score(y, y_sample)
    # ACC, NMI, ARI = clusteringMetrics(y, y_sample)
    print(f"ACC = {ACC} NMI = {NMI} ARI= {ARI}")
    figure.savefig(f"blob/-1.png")


# 画出KMS聚类后图像 - 自定义
def my_draw_kms(figure: plt.Figure, location=122) -> None:
    import random
    K = 3       # 聚类中心数
    centerPoints = []   # 聚类中心
    data = X        # 初始数据
    # 测试数据↓
    # data = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
    for i in range(K):
        # 随机选择K个点，作为聚类中心
        index = random.randint(0, len(data) - 1)
        centerPoints.append(data[index])
        # data = np.delete(data, index, axis=0)
    # 转化为numpy.narray类型，方便计算
    centerPoints = np.array(centerPoints)
    # print(f'data:\n{data}')
    # print(f'centerPoints:\n{centerPoints}')
    # 向量法计算每个聚类点到其他点的距离 - 欧式距离
    distances = numpy.array([
        np.sum((data - centerPoints[i, :])**2, axis=1)**0.5
        for i in range(K)
    ])
    # print(f'distances:\n{distances}')
    # 计算各点最邻近的聚类中心
    # distances.argmin(axis=0)：点到最近距离中心距离的值，向量形式
    # np.unravel_index：获得点划分给哪个聚类中心，取值范围0~K-1，向量形式
    y_sample = np.unravel_index(distances.argmin(axis=0), distances.shape)[1]
    # print(f'y_sample:\n{y_sample}')

    ACCs = []
    NMIs = []
    ARIs = []
    # 记录当前状态
    def draw_step(tag):
        figure = plt.figure()
        plot = figure.add_subplot()
        colors = []
        for color in y_sample:
            if color == 0:
                colors.append('r')
            elif color == 1:
                colors.append('g')
            else:
                colors.append('y')
        plot.scatter(data[:, 0], data[:, 1], c=colors)
        plot.scatter(centerPoints[:, 0], centerPoints[:, 1], s=100, marker='*', c='b')
        ACC = accuracy_score(y, y_sample)
        NMI = normalized_mutual_info_score(y, y_sample)
        ARI = adjusted_rand_score(y, y_sample)
        # ACC, NMI, ARI = clusteringMetrics(y, y_sample)
        ACCs.append(ACC)
        NMIs.append(NMI)
        ARIs.append(ARI)
        print(f'n={tag} ACC = {ACC} NMI = {NMI} ARI= {ARI}')
        plot.set_title(f'n = {tag} ACC = {ACC} NMI = {NMI} ARI= {ARI}')
        figure.savefig(f"blob/{location}.png")
    import time
    t1 = time.time()
    draw_step(1)
    t2 = time.time()
    for tag in range(10):
        for i in range(K):
            indexs = np.where(y_sample == i)
            n = len(indexs[0])
            temp_array = data[indexs]
            s1 = np.sum(temp_array[:, 0])
            s2 = np.sum(temp_array[:, 1])
            centerPoints[i][0], centerPoints[i][1] = s1 / n, s2 / n

        location += 1
        distances = numpy.array([
            np.sum((data - centerPoints[i, :])**2, axis=1)**0.5
            for i in range(K)
        ])
        y_sample = np.unravel_index(distances.argmin(axis=0), distances.shape)[1]
        draw_step(tag + 2)
        # print(temp_array, s1, s2)
        # print(indexs, n)

    # for tag in range(10):
        # # 下面要更新聚类点
        # for i in range(K):
        #     # 计算下一个聚类中心的概率
        #     # P(x) = D(x) / sum(D(x)**2)
        #     P = distances[i, :] / np.sum(distances[i, :]**2)
        #     # 选取最大概率的点替换当前聚类中心
        #     index = np.unravel_index(P.argmax(), P.shape)[0]
        #     centerPoints[i] = data[index]
        # # 重新计算距离并绘制
        # location += 1
        # distances = numpy.array([
        #     np.sum((data - centerPoints[i, :])**2, axis=1)**0.5
        #     for i in range(K)
        # ])
        # draw_step(tag + 2)
    figure = plt.figure()
    plot = figure.add_subplot()
    plot.plot(range(len(ACCs)), ACCs, label="ACC")
    plot.plot(range(len(NMIs)), NMIs, label="NMI")
    plot.plot(range(len(ARIs)), ARIs, label="ARI")
    plt.legend()
    figure.savefig(f"blob/sum.png")


if __name__ == '__main__':
    fig = plt.figure()
    draw_init(fig)
    draw_kms(fig)
    my_draw_kms(fig, 0)
    plt.tight_layout()
    # plt.show()

