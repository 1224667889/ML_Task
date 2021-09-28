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
    # X, labels = face_dataset()
    # X = X.reshape(X.shape[0], 200 * 180)
    # cul(X, labels, "faces")
    #
    # digits = load_digits()
    # X = digits.data
    # labels = digits.target
    # cul(X, labels, "digits")
    #
    # X, labels, _ = utils.createDatabase("../lesson_5/17flowers")
    # X = X.reshape(X.shape[0], 200 * 180)
    # cul(X, labels, "17flowers")
    print('''------------------------------faces------------------------------
KNN 训练消耗：0.0009953975677490234s	预测消耗：0.03288698196411133s	准确率：100.0%
NaiveBayes 训练消耗：0.05487775802612305s	预测消耗：0.10073065757751465s	准确率：100.0%
LogisticRegression 训练消耗：1.9667131900787354s	预测消耗：0.003989458084106445s	准确率：100.0%
------------------------------digits------------------------------
KNN 训练消耗：0.0010228157043457031s	预测消耗：0.014956474304199219s	准确率：98.88888888888889%
NaiveBayes 训练消耗：0.0009953975677490234s	预测消耗：0.0009775161743164062s	准确率：82.22222222222221%
LogisticRegression 训练消耗：0.09575295448303223s	预测消耗：0.001013040542602539s	准确率：97.22222222222221%
------------------------------17flowers------------------------------
KNN 训练消耗：0.03091740608215332s	预测消耗：0.8118228912353516s	准确率：20.22058823529412%
NaiveBayes 训练消耗：0.2832474708557129s	预测消耗：1.4531168937683105s	准确率：29.77941176470588%
LogisticRegression 训练消耗：14.897114515304565s	预测消耗：0.034906864166259766s	准确率：29.044117647058826%''')
