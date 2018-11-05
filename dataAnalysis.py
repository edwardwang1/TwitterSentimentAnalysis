# Import Statements
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
import re


def split(string, maxsplit=0):
    import re
    delimiters = " ", ".", "?", "!",
    regexPattern = '|'.join(map(re.escape, delimiters))
    rawlist = re.split(regexPattern, string, maxsplit)
    rawlist = [x for x in rawlist if x.strip()]
    rawlist = [x.lower() for x in rawlist]
    return rawlist


#Load the data

train = pd.read_csv('data/train.csv', encoding='latin-1')

X_raw = train.drop(labels="ItemID", axis=1)
X_raw = X_raw.drop(labels="Sentiment", axis=1)

Y_train = train["Sentiment"]


X_train = pd.DataFrame(index=Y_train.index)



##Counting Number of Positive and Negative Words
with open("negativeWords.txt", "r") as f:
    negText = f.read()
negTokens = negText.split("\n") # This splits the text file into tokens on the new line character
negTokens[-1:] = [] # This strips out the final empty item


with open("positiveWords.txt", "r") as f:
    posText = f.read()
posTokens = posText.split("\n") # This splits the text file into tokens on the new line character
posTokens[-1:] = [] # This strips out the final empty item


def countPositive(phrase):
    listOfWords = split(phrase)
    posScore = 0
    for word in listOfWords:
        if word in posTokens:
            posScore += 1
    return posScore

def countNegative(phrase):
    listOfWords = split(phrase)
    negScore = 0
    for word in listOfWords:
        if word in negTokens:
            negScore += 1
    return negScore

def getNumWords(phrase):
    listOfWords = split(phrase)
    return len(listOfWords)


#Creating series that counts num of exclaimatino marks and num question makrs
#and num of pos and neg words
# and len of words
# and number len of characters
numPos = pd.Series(index=Y_train.index)
numNeg = pd.Series(index=Y_train.index)
numEx = pd.Series(index=Y_train.index)
numQuest = pd.Series(index=Y_train.index)
numQuest = pd.Series(index=Y_train.index)
numChar = pd.Series(index=Y_train.index)
numWords = pd.Series(index=Y_train.index)

for i in range(Y_train.size):
    numPos[i] = countPositive(X_raw.loc[i]["SentimentText"])
    numNeg[i] = countNegative(X_raw.loc[i]["SentimentText"])
    numEx[i] = X_raw.loc[i]["SentimentText"].count("!")
    numQuest[i] = X_raw.loc[i]["SentimentText"].count("?")
    numChar[i] = len(X_raw.loc[i]["SentimentText"])
    numWords[i] = getNumWords(X_raw.loc[i]["SentimentText"])



#Adding Series to TrainDF
X_train['numPos'] = numPos
X_train['numNeg'] = numNeg
X_train['numEx'] = numEx
X_train['numQuest'] = numQuest
X_train['numChars'] = numChar
X_train['numWords'] = numWords



print(X_train.head().to_string())

X_train.to_csv("testData2.csv")

#
#
# print(train.head().to_string())
# print(X_train.head().to_string())
#
#
#
#
# print(X_raw.loc[1]["SentimentText"])
# for i in range(5):
#     a = split(X_raw.loc[i]["SentimentText"])
#     print(a)


