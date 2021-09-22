from resource import EmailFeatureGeneration
from sklearn.model_selection import train_test_split


X, labels = EmailFeatureGeneration.Text2Vector()
X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=22)
if __name__ == '__main__':
    from session_1 import cul
    cul(X, labels, "emails")



