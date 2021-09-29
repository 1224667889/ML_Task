from lesson_2.session_3 import unpickle
# import torchvision.datasets
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from lesson_7.session_2 import load_cifar10_by_batch
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def cul_in_test(X_train, X_test, labels_train, labels_test, title):
    print(f'------------------------------{title}------------------------------')
    knn = KNN()
    # 输入数据集和标签
    t1 = time.time()
    knn.fit(X_train, labels_train)
    t2 = time.time()
    knn_pred = knn.predict(X_test)
    knn_ACC = np.sum(np.array(knn_pred) == np.array(labels_test)) / len(knn_pred)
    t3 = time.time()
    print(f'KNN 训练消耗：{t2-t1}s\t预测消耗：{t3-t2}s\t准确率：{knn_ACC*100}%')

    gnb = GaussianNB()
    t1 = time.time()
    gnb.fit(X_train, labels_train)
    t2 = time.time()
    gnb_pred = gnb.predict(X_test)
    gnb_ACC = np.sum(np.array(gnb_pred) == np.array(labels_test)) / len(gnb_pred)
    t3 = time.time()
    print(f'NaiveBayes 训练消耗：{t2-t1}s\t预测消耗：{t3-t2}s\t准确率：{gnb_ACC*100}%')

    lr = LogisticRegression()
    t1 = time.time()
    lr.fit(X_train, labels_train)
    t2 = time.time()
    lr_pred = lr.predict(X_test)
    lr_ACC = np.sum(np.array(lr_pred) == np.array(labels_test)) / len(lr_pred)
    t3 = time.time()
    print(f'LogisticRegression 训练消耗：{t2-t1}s\t预测消耗：{t3-t2}s\t准确率：{lr_ACC*100}%')

    svm = SVC()
    t1 = time.time()
    svm.fit(X_train, labels_train)
    t2 = time.time()
    svm_pred = svm.predict(X_test)
    svm_ACC = np.sum(np.array(svm_pred) == np.array(labels_test)) / len(svm_pred)
    t3 = time.time()
    print(f'SVM 训练消耗：{t2-t1}s\t预测消耗：{t3-t2}s\t准确率：{svm_ACC*100}%')
    plt.title(title)
    plt.bar(["KNN", "NaiveBayes", "Logistic", "SVM"],
            [knn_ACC, gnb_ACC, lr_ACC, svm_ACC])
    plt.show()


if __name__ == '__main__':
    for i in range(1, 4):
        X_train, X_test, labels_train, labels_test = load_cifar10_by_batch(i)
        cul_in_test(X_train, X_test, labels_train, labels_test, f"batch_{i}")
