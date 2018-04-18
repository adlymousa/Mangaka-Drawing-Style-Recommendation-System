import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from numpy import array
from scipy.stats.stats import pearsonr

#Saving featureVectors to a csv file
header1 = ["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26","f27","f28","f29","f30"]
header2 = ["MangaName"]
#Writing the SVM
mangaNameToPredict = "LoveHina_vol01"
predictFeatures = [];
def Build_Data_Set(features = header1, features1 = header2):
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
    #y = np.delete(y, mangaIndex, 0)
    #X = np.delete(X, mangaIndex, 0)

    return X,y

def Analysis():
    X,y = Build_Data_Set()
    X = array( X )
    y = array( y )
    C = 100 # SVM regularization parameter
    models = [(svm.SVC(kernel='rbf', gamma=0.7, C=C))]
    models = (clf.fit(X, y) for clf in models)
    xx = np.linspace(0,1000)
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    titles = []
    for i in range(0,30):
        titles.append('SVC with RBF kernel '+str(i))
    #c = np.random.random(109)
    c=[]
    for x in range(0, 109):
        if x==44 or x==45 :
            c.append(1)
        elif x==30 or x==31 :
            c.append(2)
        elif x==57 or x==56 :
            c.append(3)
        elif x==78 or x==79 :
            c.append(4)
        elif x==82 or x==83 :
            c.append(5)
        else :
            c.append(0)
    c = np.array(c)
    MangaID = []
    for i in range(0,109) :
        MangaID.append(i+5)
    """"""""""""""""""""""""""""""""""""""""""
    FeatureIndex = 13
    FeatureIndexSecond = 12
    """"""""""""""""""""""""""""""""""""""""""
    FeaturesFile = open('D:/semesters/graduation project - manga/CollarationMargin80%.csv', 'w+')
    FeaturesFile.write(",0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27\n")
    for featureIndex in range(0,28):
        FeaturesFile.write(str(featureIndex)+",")
        for featureIndexSecond in range(0,28):
            value = pearsonr(X[:, featureIndexSecond], X[:, featureIndex])
            if (value[0] > 0.8 or value[0] < -0.8) and featureIndexSecond!= featureIndex:
                FeaturesFile.write(str(value[0])+",")
            else :
                FeaturesFile.write(",")
        FeaturesFile.write("\n")
Analysis()
