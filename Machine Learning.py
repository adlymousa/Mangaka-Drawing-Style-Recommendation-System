import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from numpy import array

#Saving featureVectors to a csv file
header1 = ["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26","f27","f28","f29","f30"]
header2 = ["MangaName"]
#Writing the SVM
predictFeatures = [];
def Build_Data_Set(features = header1, features1 = header2):
    mangaNameToPredict = "AisazuNihaIrarenai"
    data_df = pd.DataFrame.from_csv("D:/semesters/graduation project - manga/features.csv")
    X = np.array(data_df[features].values)

    data_df2 = pd.DataFrame.from_csv("D:/semesters/graduation project - manga/mangaNames.csv")
    oldY = np.array(data_df2[features1].values)
    y=[]
    counterManga = 0
    for name in oldY:
        y.append(name[0])
    for mangaX in X:
        counterFeature = 0
        for value in mangaX:
            mangaX[counterFeature] = mangaX[counterFeature]
            counterFeature += 1
        counterManga += 1
    mangaIndex = y.index(mangaNameToPredict);
    predictFeatures.append(X[mangaIndex])
    y = np.delete(y, mangaIndex, 0)
    X = np.delete(X, mangaIndex, 0)

    return X,y

def Analysis():
    X,y = Build_Data_Set()
    X = array( X )
    y = array( y )
    C = 100 # SVM regularization parameter
    #clf = (svm.SVC(kernel='rbf', gamma=0.7, C=C))
    clf1 = (svm.SVC(kernel='linear', C=C))
    #clf2 = (svm.LinearSVC(C=C))
    #clf3 = (svm.SVC(kernel='poly', degree=3, C=C))
    clf1 = clf1.fit(X, y)
    #clf2 = clf2.fit(X, y)
    #clf3 = clf3.fit(X, y)
    value1 = clf1.predict(predictFeatures);
    #value2 = clf2.predict(predictFeatures);
    #value3 = clf3.predict(predictFeatures);
    print(value1)
    #print(value2)
    #print(value3)
    """models = [(svm.SVC(kernel='rbf', gamma=0.7, C=C))]
    models = (clf.fit(X, y) for clf in models)
    xx = np.linspace(0,30)
    yy = np.linspace(0,30)
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    titles = ('SVC with RBF kernel')
    #c = np.random.random(109)
    c=[]
    for x in range(0, 109):
        if x==0 :
            c.append(2)
        elif x==107 :
            c.append(1)
        else :
            c.append(0)
    c = np.array(c)
    for clf, title in zip(models, titles):
        sub.scatter(X[:, 0], X[:, 1], c=c, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        sub.set_xlim(xx.min(), xx.max())
        sub.set_ylim(yy.min(), yy.max())
        sub.set_xlabel('Sepal length')
        sub.set_ylabel('Sepal width')
        sub.set_xticks(())
        sub.set_yticks(())
        sub.set_title(title)
    plt.show()"""
Analysis()
