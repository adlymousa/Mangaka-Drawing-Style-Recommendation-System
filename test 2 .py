import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
#Saving featureVectors to a csv file
header1 = ["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26"]
header2 = ["MangaName"]
#Writing the SVM
def Build_Data_Set(features = header1, features1 = header2):

    data_df = pd.DataFrame.from_csv("D:/semesters/graduation project - manga/features.csv")
    #data_df = data_df[:250]
    X = np.array(data_df[features].values)

    data_df2 = pd.DataFrame.from_csv("D:/semesters/graduation project - manga/mangaNames.csv")
    oldY = np.array(data_df2[features1].values)
    y=[]
    for name in oldY:
        y.append(name[0])
    return X,y

def Analysis():
    X,y = Build_Data_Set()
    print(X)
    print(y)
    C = 1.0  # SVM regularization parameter
    clf = (svm.SVC(kernel='rbf', gamma=0.7, C=C))
    clf = clf.fit(X, y)
    #value = clf.predict([[2.,1,47,11]]);
    #print(value)


    """
    models = [(svm.SVC(kernel='rbf', gamma=0.7, C=C))]
    models = (clf.fit(X, y) for clf in models)
    xx = np.linspace(0,5)
    yy = np.linspace(0,185)
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    titles = ('SVC with RBF kernel')
    for clf, title in zip(models, titles):
        sub.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        sub.set_xlim(xx.min(), xx.max())
        sub.set_ylim(yy.min(), yy.max())
        sub.set_xlabel('Sepal length')
        sub.set_ylabel('Sepal width')
        sub.set_xticks(())
        sub.set_yticks(())
        sub.set_title(title)
    plt.show()"""
Analysis()
