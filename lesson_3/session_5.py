from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from lesson_4.utils import cluster_acc

photo_name = []
imgs = os.listdir('face_images/imgs')
y = []
i = 0
for img in imgs:
    photo = os.listdir(f'./face_images/imgs/{img}')
    for name in photo:
        photo_name.append(f'face_images/imgs/{img}/{name}')
        y.append(i)
        i += 1

photos = pd.DataFrame()
for path in photo_name:
    photo = imgplt.imread(path)
    photo = photo.reshape(1, -1)
    photo = pd.DataFrame(photo)
    photos = photos.append(photo, ignore_index=True)
photos = photos.values


def draw():
    fig, ax = plt.subplots(nrows=10, ncols=20, figsize=[10, 5], dpi=80)
    plt.subplots_adjust(wspace=0, hspace=0)
    count = 0
    for i in range(10):
        for j in range(20):
            ax[i, j].imshow(result[count])
            count += 1
    plt.xticks([])
    plt.yticks([])
    plt.show()


def Kms(imgs, n_clusters):
    clf = KMeans(n_clusters=n_clusters)
    clf.fit(imgs)
    y_predict = clf.predict(imgs)
    centers = clf.cluster_centers_
    ret = centers[y_predict].astype("int64").reshape(200, 200, 180, 3)
    return ret, y_predict


if __name__ == '__main__':
    for i in range(1, 9):
        import time
        t = time.time()
        result, y_predict = Kms(photos, 10)
        print(i, time.time() - t)

    # 输出值
    ACC = cluster_acc(y, y_predict)
    NMI = normalized_mutual_info_score(y, y_predict)
    ARI = adjusted_rand_score(y, y_predict)
    print(f"ACC = {ACC} NMI = {NMI} ARI= {ARI}")

    # 绘制图像
    fig, ax = plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=[10, 5], dpi=80)
    plt.subplots_adjust(wspace=0, hspace=0)
    count = 0
    for i in range(10):
        for j in range(20):
            ax[i, j].imshow(result[count])
            count += 1
    plt.xticks([])
    plt.yticks([])
    plt.show()
