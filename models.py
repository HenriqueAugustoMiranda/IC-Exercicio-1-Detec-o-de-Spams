from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score


def KNN(X_train, X_test, y_train, y_test, n_neighbors = 5, weights='distance'):

    KNC = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric='cosine',
        algorithm='brute'
    )
    KNC.fit(X_train, y_train)
    predict = KNC.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    f1 = f1_score(y_test, predict, average="macro")

    return accuracy, f1


def SVM(X_train, X_test, y_train, y_test, C=1, loss='squared_hinge', max_iter=3000):

    svm = LinearSVC(
        C=C,
        loss=loss,
        max_iter=max_iter,
        random_state=42
    )
    svm.fit(X_train, y_train)
    predict = svm.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    f1 = f1_score(y_test, predict, average="macro")

    return accuracy, f1


def DTC(X_train, X_test, y_train, y_test, criterion = 'entropy', max_depth = 30, min_samples_split = 10):
    
    dtc = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    dtc.fit(X_train, y_train)
    predict = dtc.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    f1 = f1_score(y_test, predict, average="macro")

    return accuracy, f1
