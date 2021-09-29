import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.naive_bayes import GaussianNB
from lesson_5 import utils
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier as KNN
from lesson_4.utils import face_dataset
from sklearn.linear_model import LogisticRegression


def cul(X, labels, title):
    print(f'------------------------------{title}------------------------------')
    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=22)

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


if __name__ == '__main__':
    X, labels = face_dataset()
    X = X.reshape(X.shape[0], 200 * 180)
    cul(X, labels, "faces")

    digits = load_digits()
    X = digits.data
    labels = digits.target
    cul(X, labels, "digits")

    X, labels, _ = utils.createDatabase("../lesson_5/17flowers")
    X = X.reshape(X.shape[0], 200 * 180)
    cul(X, labels, "17flowers")
