import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import SVC


# 画出初始图像
def draw(X, labels, title='data by make_circles()') -> None:
    plt.figure()
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()


# 分类
def classify(model, X_train, labels_train):
    model.fit(X_train, labels_train)
    pred = model.predict(X_train)
    acc = np.sum(np.array(pred) == np.array(labels_train)) / len(pred)
    return pred, acc


X_train, labels_train = datasets.make_circles(
    n_samples=1000,
    shuffle=True,
    noise=.1,
    random_state=4103,
    factor=.1
)
if __name__ == '__main__':
    # 初始值
    draw(X_train, labels_train)

    # KNN
    knn_pred, knn_acc = classify(KNN(), X_train, labels_train)
    draw(X_train, knn_pred, "KNN")
    print(f"KNN ACC: {knn_acc}")

    # NaiveBayes
    nb_pred, nb_acc = classify(GaussianNB(), X_train, labels_train)
    draw(X_train, nb_pred, "NaiveBayes")
    print(f"NaiveBayes ACC: {nb_acc}")

    # Logistic
    lr_pred, lr_acc = classify(LogisticRegression(), X_train, labels_train)
    draw(X_train, lr_pred, "Logistic")
    print(f"Logistic ACC: {lr_acc}")

    # SVM
    svm_pred, svm_acc = classify(SVC(C=1.0, kernel='rbf', gamma='scale'), X_train, labels_train)
    draw(X_train, svm_pred, "SVM C=1.0 kernel=rbf gamma=scale")
    print(f"SVM ACC: {svm_acc}")

    # 柱状图
    plt.bar(["KNN", "NaiveBayes", "Logistic", "SVM"],
            [knn_acc, nb_acc, lr_acc, svm_acc])
    plt.show()


