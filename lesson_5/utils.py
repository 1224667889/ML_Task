import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from lesson_4.utils import clusteringMetrics
import matplotlib.pyplot as plt
from PIL import Image


def draw_plots(X, digits, title):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    # 归一化
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    # 画出样本点
    for i in range(X.shape[0]):
        # 在样本点所在位置画出样本点的数字标签
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(digits.target[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 5})
    shown_images = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    for i in range(digits.data.shape[0]):
            # 计算距离
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < .4:
                # 距离太近，不显示
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            ax.text(X[i][0], X[i][1], s=int(digits.target[i]), fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


def createDatabase(path, image_size=40, image_column=10, image_row=10):
    to_image = Image.new('RGB', (image_column * image_size, image_row * image_size))
    imgs = os.listdir(path)
    x = []
    y = []
    for i, p in enumerate(imgs):
        img = Image.open(f'./{path}/{p}').resize((image_size, image_size), Image.ANTIALIAS)
        to_image.paste(img, ((i % 80) * image_size, (i // 80) * image_size))
        image = cv2.imread(f'./{path}/{p}').astype(np.float32)
        image = cv2.resize(image, (180, 200))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        x.append(image)
        y.append(str(i // 80 + 1))
    x = np.array(x)
    y = np.array(y)
    return x, y, to_image
