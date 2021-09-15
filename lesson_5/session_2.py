from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from utils import draw_plots
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


digits = load_digits()
X = digits.data
labels = digits.target
X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=22)


if __name__ == '__main__':
    pca = PCA(n_components=9)
    X_pca = pca.fit_transform(X)
    draw_plots(X_pca, digits, "PCA")

    lda = LDA(n_components=9)
    X_lda = lda.fit_transform(X, labels)
    draw_plots(X_lda, digits, "LDA")
    plt.show()



