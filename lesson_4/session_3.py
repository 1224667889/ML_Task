"""绘制PCA聚类性能变化图"""
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from utils import clusteringMetrics, draw_bars, face_dataset
import matplotlib.pyplot as plt


# 计算当pca降维后特征个数与聚类性能
def clusteringMetricsList(X):
    import time
    ACCs = []
    NMIs = []
    ARIs = []
    for n_components in range(1, 9):
        t = time.time()
        pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)

        X_train_pca = pca.transform(X)
        k_model = k_means(X_train_pca, n_clusters=10)
        cluster_label_circle = k_model[1]
        ACC, NMI, ARI = clusteringMetrics(y, cluster_label_circle)
        # print(n_components, time.time() - t)
        ACCs.append(ACC)
        NMIs.append(NMI)
        ARIs.append(ARI)
        print(f"n_components = {n_components} ACC = {ACC} NMI = {NMI} ARI= {ARI}")
    return ACCs, NMIs, ARIs


if __name__ == '__main__':
    n_samples, h, w = 200, 200, 180
    photos, y = face_dataset()
    X = photos.reshape(n_samples, h*w)
    ACCs, NMIs, ARIs = clusteringMetricsList(X)
    draw_bars(ACCs, NMIs, ARIs)
    plt.show()

