"""绘制PCA特征图"""

from sklearn.decomposition import PCA
from utils import plot_gallery, face_dataset
import matplotlib.pyplot as plt

if __name__ == '__main__':
    n_samples, h, w = 200, 200, 180
    photos, y = face_dataset()
    X = photos.reshape(n_samples, h*w)
    pca = PCA(n_components=10, svd_solver='randomized', whiten=True).fit(X)
    # 绘制前十张特征图 GRAY + RGB
    eigenfaces = pca.components_.reshape((10, h, w))
    plot_gallery(eigenfaces, h, w)
    plt.show()
