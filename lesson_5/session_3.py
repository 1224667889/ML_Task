import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import utils
import time
from sklearn.neighbors import KNeighborsClassifier as KNN

if __name__ == '__main__':
    X, labels, to_image = utils.createDatabase("17flowers")
    X = X.reshape(X.shape[0], 200 * 180)
    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=22)

    plt.figure()
    plt.imshow(to_image)
    plt.show()

    # from lesson_2.session_3 import KNN

    PCA_ACCs = []
    LDA_ACCs = []
    for K in range(1, 10):
        t0 = time.time()
        pca = PCA(n_components=K).fit(X_train)
        x_train_pca = pca.transform(X_train)
        x_test_pca = pca.transform(X_test)
        print("|", time.time() - t0, "|")
        knn = KNN()
        t1 = time.time()
        knn.fit(x_train_pca, labels_train)
        t2 = time.time()
        PCA_pred = knn.predict(x_test_pca)
        t3 = time.time()
        ACC_PCA = np.sum(np.array(PCA_pred) == np.array(labels_test)) / len(PCA_pred)
        # print(f'PCA K={K} 训练消耗：{t2-t1}s\t预测消耗：{t3-t2}s\t准确率：{ACC_PCA*100}%')
        # print(f'|K={K}|{t2-t1}s|{t3-t2}s|{ACC_PCA*100}%|')
        PCA_ACCs.append(ACC_PCA)
    print("------------------")
    for K in range(1, 10):
        t0 = time.time()
        lda = LDA(n_components=K).fit(X_train, labels_train)
        x_train_lda = lda.transform(X_train)
        x_test_lda = lda.transform(X_test)
        print("|", time.time() - t0, "|")
        knn = KNN()
        t1 = time.time()
        knn.fit(x_train_lda, labels_train)
        t2 = time.time()
        LDA_pred = knn.predict(x_test_lda)
        t3 = time.time()
        ACC_LDA = np.sum(np.array(LDA_pred) == np.array(labels_test)) / len(LDA_pred)
        # print(f'LDA K={K} 训练消耗：{t2-t1}s\t预测消耗：{t3-t2}s\t准确率：{ACC_LDA*100}%')
        # print(f'|K={K}|{t2-t1}s|{t3-t2}s|{ACC_LDA*100}%|')
        LDA_ACCs.append(ACC_LDA)

    _, ax = plt.subplots()
    bar_width = 0.3
    index = np.arange(9)
    ax.bar(index, PCA_ACCs, bar_width, label='PCA')
    ax.bar(index + bar_width, LDA_ACCs, bar_width, label='LCD')
    ax.legend()
    plt.show()
