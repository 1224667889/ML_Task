from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


digits = load_digits()
X = digits.data
labels = digits.target

X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=22)


def draw():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    img = np.zeros((120, 120))
    for i in range(15):
        ix = 8 * i
        for j in range(15):
            iy = 8 * j
            img[ix:ix + 8, iy:iy + 8] = X[i * 30 + j].reshape((8, 8))
    plt.figure()
    plt.imshow(img)
    plt.title("字符数据可视化")
    plt.show()


if __name__ == '__main__':
    draw()
    # 要用到KNN，调用前面写的
    from lesson_2.session_3 import KNN

    PCA_ACCs = []
    LDA_ACCs = []
    for K in range(1, 10):
        pca = PCA(n_components=K, svd_solver='randomized', whiten=True).fit(X)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        knn_PCA = KNN()
        # 输入数据集和标签
        t1 = time.time()
        knn_PCA.train(X_train_pca, labels_train)
        t2 = time.time()
        PCA_pred = knn_PCA.predict(X_test_pca, K)
        PCA_ACC = np.sum(np.array(PCA_pred) == np.array(labels_test)) / len(PCA_pred)
        t3 = time.time()
        print(f'PCA K={K} 训练消耗：{t2-t1}s\t预测消耗：{t3-t2}s\t准确率：{PCA_ACC*100}%')
        PCA_ACCs.append(PCA_ACC)

    for K in range(1, 10):
        lda = LDA(n_components=K).fit(X, labels)
        X_train_lda = lda.transform(X_train)
        X_test_lda = lda.transform(X_test)
        knn_LDA = KNN()
        # 输入数据集和标签
        t4 = time.time()
        knn_LDA.train(X_train_lda, labels_train)
        t5 = time.time()
        LDA_pred = knn_LDA.predict(X_test_lda, K)
        LDA_ACC = np.sum(np.array(LDA_pred) == np.array(labels_test)) / len(LDA_pred)
        t6 = time.time()
        print(f'LDA K={K} 训练消耗：{t5-t4}s\t预测消耗：{t6-t5}s\t准确率：{LDA_ACC*100}%')
        LDA_ACCs.append(LDA_ACC)

    _, ax = plt.subplots()
    bar_width = 0.3
    index = np.arange(9)
    ax.bar(index, PCA_ACCs, bar_width, label='ACC')
    ax.bar(index + bar_width, LDA_ACCs, bar_width, label='NMI')
    ax.legend()
    plt.show()


