from resource import EmailFeatureGeneration
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np


def draw_confusion_matrix(labels, y_pred):
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    from sklearn.metrics import confusion_matrix
    maxtrix = confusion_matrix(labels, y_pred, labels=[0, 1])
    tn, fp = maxtrix[0]
    fn, tp = maxtrix[1]

    plt.matshow(maxtrix, fignum=1)
    plt.xlabel('预测类型')
    plt.ylabel('实际类型')
    font = {
        'size': 32,
        'color': "white"
    }
    total = tn + fp + fn + tp
    plt.text(0.1, 0.1, "{:.2f}".format(tn / total), fontdict=font)
    plt.text(0.9, 0.1, "{:.2f}".format(fp / total), fontdict=font)
    plt.text(0.1, 0.9, "{:.2f}".format(fn / total), fontdict=font)
    plt.text(0.9, 0.9, "{:.2f}".format(tp / total), fontdict=font)

    plt.show()
    return tn, fp, fn, tp


X, labels = EmailFeatureGeneration.Text2Vector()
X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=22)
if __name__ == '__main__':
    # GNB
    gnb = GaussianNB()
    gnb.fit(X_train, labels_train)
    gnb_pred = gnb.predict(X_test)
    gnb_ACC = np.sum(np.array(gnb_pred) == np.array(labels_test)) / len(gnb_pred)

    tn, fp, fn, tp = draw_confusion_matrix(labels_test, gnb_pred)

    print("====Tabular====")
    print("\t0ᴬ\t1ᴬ")
    print(f"0ᴾ\t{tn}\t{fp}")
    print(f"1ᴾ\t{fn}\t{tp}")
    print("Note: classᴾ = Predicted, classᴬ = Actual")
    print("===============")
    print(f"ACC = {gnb_ACC}")


