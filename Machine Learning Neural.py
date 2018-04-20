import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from numpy import array
from sklearn.neural_network import MLPClassifier
header1 = ["f11","f12","f13","f14","f15","f16","f18","f23","f24","f25","f26","f27","f28"]
header2 = ["MangaName"]
data_df2 = pd.DataFrame.from_csv("D:/semesters/graduation project - manga/mangaNames.csv")
oldY = np.array(data_df2[header2].values)
y=[]
for name in oldY:
    y.append(name[0])
FeaturesFile = open('D:/semesters/graduation project - manga/predicted.csv', 'w+')
FeaturesFile.write("Manga,Prediction 1,Prediction 2,Prediction 3\n")
for mangaNameToPredictName in y :
    PredictFileString = mangaNameToPredictName
    mangaNameToPredict = mangaNameToPredictName
    predictFeatures = [];
    deleteManga = [];
    def Build_Data_Set(features = header1, features1 = header2, toDelete = deleteManga):
        data_df = pd.DataFrame.from_csv("D:/semesters/graduation project - manga/features.csv")
        X = np.array(data_df[features].values)
        data_df2 = pd.DataFrame.from_csv("D:/semesters/graduation project - manga/mangaNames.csv")
        oldY = np.array(data_df2[features1].values)
        y=[]
        for name in oldY:
            y.append(name[0])
        for mangaName in deleteManga :
            try :
                mangaIndex = y.index(mangaName);
            except:
                mangaIndex = y.tolist().index(mangaName);
            if len(toDelete)==1 :
                predictFeatures.append(X[mangaIndex])
            y = np.delete(y, mangaIndex, 0)
            X = np.delete(X, mangaIndex, 0)
        return X,y

    def Analysis():
        X,y = Build_Data_Set()
        X = array( X )
        y = array( y )
        C = 100 # SVM regularization parameter
        clf1 = MLPClassifier(solver='lbfgs',  activation='relu', alpha=1,hidden_layer_sizes=(13,6,3), random_state=1, learning_rate_init=0.01,max_iter=20000)
        clf1 = clf1.fit(X, y)
        value1 = clf1.predict(predictFeatures);
        print(value1[0])
        return value1[0]
    deleteManga.append(mangaNameToPredict)
    x = Analysis()
    PredictFileString += ","+x
    deleteManga.append(x)
    x = Analysis()
    PredictFileString += ","+x
    deleteManga.append(x)
    x = Analysis()
    PredictFileString += ","+x
    PredictFileString += "\n"
    FeaturesFile.write(PredictFileString)
