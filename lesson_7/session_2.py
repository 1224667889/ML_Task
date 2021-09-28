from lesson_2.session_3 import unpickle
# import torchvision.datasets
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression


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


def load_cifar10_by_batch(batch=1):
    train_data = dict()
    train_data.update(unpickle(f'F://数据集/cifar-10-batches-py/data_batch_{batch}'))
    test_data = unpickle('F://数据集/cifar-10-batches-py/test_batch')
    X_train = train_data[b'data']
    labels_train = np.array(train_data[b'labels'])
    X_test = test_data[b'data']
    labels_test = np.array(test_data[b'labels'])
    return X_train, X_test, labels_train, labels_test


if __name__ == '__main__':
    # for i in range(1, 4):
    #     X_train, X_test, labels_train, labels_test = load_cifar10_by_batch(i)
    #     cul_in_test(X_train, X_test, labels_train, labels_test, f"batch_{i}")
    print('''------------------------------batch_1------------------------------
KNN 训练消耗：0.0009975433349609375s	预测消耗：4.8719658851623535s	准确率：28.970000000000002%
NaiveBayes 训练消耗：0.2922179698944092s	预测消耗：2.216071605682373s	准确率：29.299999999999997%
LogisticRegression 训练消耗：9.600942373275757s	预测消耗：0.11668753623962402s	准确率：37.169999999999995%
------------------------------batch_2------------------------------
KNN 训练消耗：0.0009975433349609375s	预测消耗：4.921832323074341s	准确率：28.88%
NaiveBayes 训练消耗：0.294238805770874s	预测消耗：2.3186562061309814s	准确率：29.849999999999998%
LogisticRegression 训练消耗：10.866927862167358s	预测消耗：0.13663434982299805s	准确率：37.44%
------------------------------batch_3------------------------------
KNN 训练消耗：0.0009975433349609375s	预测消耗：5.5262157917022705s	准确率：30.28%
NaiveBayes 训练消耗：0.3061807155609131s	预测消耗：2.447453260421753s	准确率：29.580000000000002%
LogisticRegression 训练消耗：10.486356735229492s	预测消耗：0.10471963882446289s	准确率：36.730000000000004%''')