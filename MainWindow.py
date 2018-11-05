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


analyser = SentimentIntensityAnalyzer()
def tokenizePhrase(phrase):
    snt = analyser.polarity_scores(phrase)
    pos = snt["pos"]
    neut = snt["neu"]
    neg = snt["neg"]
    comp = snt["compound"]
    numQuest = phrase.count("?")
    numChar = len(phrase)
    T = [pos, neut, neg, comp, numQuest, numChar]
    df = pd.DataFrame([T])
    return df


clf = joblib.load('svcVader.joblib')


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Analyze Twwet'
        self.left = 100
        self.top = 100
        self.width = 900
        self.height = 900
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setObjectName("Main")
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280, 40)

        # Create a button in the window
        self.button = QPushButton('Analyze Tweet', self)
        self.button.move(20, 80)
        self.button.adjustSize()

        # Create a button in the window
        self.resetButton = QPushButton('Reset', self)
        self.resetButton.adjustSize()
        self.resetButton.move(20, 160)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.resetButton.clicked.connect(self.on_reset_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        #QMessageBox.question(self, 'Message - pythonspot.com', "You typed: " + textboxValue, QMessageBox.Ok,
                             #QMessageBox.Ok)
        #self.textbox.setText("")

        sent = clf.predict(tokenizePhrase(textboxValue))
        conf = clf.decision_function((tokenizePhrase(textboxValue)))
        print(conf)
        if abs(conf)<0.25:
            self.setStyleSheet("QWidget#Main {background-image: url(cloudy.png) 0 0 0 0 stretch stretch;}")
        else:
            if sent[0] == 0:
                self.setStyleSheet("QWidget#Main {background-image: url(rainy.jpg) 0 0 0 0 stretch stretch;}")
            else:
                self.setStyleSheet("QWidget#Main {background-image: url(sunny.jpg) 0 0 0 0 stretch stretch;}")


    @pyqtSlot()
    def on_reset_click(self):
        self.textbox.setText("")
        self.setStyleSheet("QWidget#Main {background-image: none;}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())