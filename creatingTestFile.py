import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import pandas as pd
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np
import os
from sklearn.externals import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer


test = pd.read_csv('data/test.csv', encoding='latin-1')

analyser = SentimentIntensityAnalyzer()


pos = pd.Series(index=test.index)
neut = pd.Series(index=test.index)
neg = pd.Series(index=test.index)
comp = pd.Series(index=test.index)
numQuest = pd.Series(index=test.index)
numChar = pd.Series(index=test.index)
numPer = pd.Series(index=test.index)

print(test.shape[0])
for i in range(test.shape[0]):
    snt = analyser.polarity_scores(test.loc[i]["SentimentText"])
    pos[i] = snt["pos"]
    neut[i] = snt["neu"]
    neg[i] = snt["neg"]
    comp[i] = snt["compound"]
    numQuest[i] = test.loc[i]["SentimentText"].count("?")
    numChar[i] = len(test.loc[i]["SentimentText"])
    numPer[i] = test.loc[i]["SentimentText"].count(".")


dfForAnalysis = pd.DataFrame(index=test.index)
dfForAnalysis['Positive'] = pos
dfForAnalysis['Neutral'] = neut
dfForAnalysis['Negative'] = neg
dfForAnalysis['Compound'] = comp
dfForAnalysis['numQuest'] = numQuest
dfForAnalysis['numChars'] = numChar
dfForAnalysis['numPers'] = numPer

clf = joblib.load('svcVaderWithPer.joblib')
sent = clf.predict(dfForAnalysis)


finalOutput = pd.DataFrame(index=test.index)
finalOutput["ItemID"] = test["ItemID"]
sentDS = pd.Series(sent)
finalOutput["Sentiment"] = sent


finalOutput.to_csv("finalOutput.csv")
