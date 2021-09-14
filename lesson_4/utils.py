from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os


def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def clusteringMetrics(trueLabel, predictiveLabel):
    # Clustering accuracy
    ACC = cluster_acc(trueLabel, predictiveLabel)

    # Normalized mutual information
    NMI = normalized_mutual_info_score(trueLabel, predictiveLabel)

    # Adjusted rand index
    ARI = adjusted_rand_score(trueLabel, predictiveLabel)

    return ACC, NMI, ARI


def plot_gallery(images, h, w):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * 10, 2.4 * 2))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(str(i), size=12)
        plt.xticks(())
        plt.yticks(())

    for i in range(10):
        plt.subplot(2, 10, 10 + i + 1)
        plt.imshow(images[i].reshape((h, w)))
        plt.title(str(i), size=12)
        plt.xticks(())
        plt.yticks(())


def draw_bars(ACCs, NMIs, ARIs):
    plt.rcParams['font.sans-serif'] = ['SimHei']

    _, ax = plt.subplots()
    bar_width = 0.3

    index = np.arange(8)

    ax.bar(index, ACCs, bar_width, label='ACC')
    ax.bar(index + bar_width, NMIs, bar_width, label='NMI')
    ax.bar(index + 2 * bar_width, ARIs, bar_width, label='ARI')

    # 支持中文
    ax.set_ylabel('聚类性能指标')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(range(8))
    ax.legend()
    plt.title("PCA降维 - 聚类性能")


def face_dataset():
    photo_name = []
    imgs = os.listdir('../lesson_3/face_images/imgs')
    y = []
    for i, img in enumerate(imgs):
        photo = os.listdir(f'../lesson_3/face_images/imgs/{img}')
        for name in photo:
            photo_name.append(f'../lesson_3/face_images/imgs/{img}/{name}')
            y.append(i)

    photos = []
    for path in photo_name:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BAYER_GB2GRAY)
        image = cv2.resize(image, (180, 200))
        # 转为1-D
        image = image.reshape(image.size, 1)
        photos.append(image)
    photos = np.array(photos)
    return photos, y

