# Import Statements
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the data

train = pd.read_csv('data/train.csv', encoding='latin-1')

X_raw = train.drop(labels="ItemID", axis=1)
X_raw = X_raw.drop(labels="Sentiment", axis=1)

Y_train = train["Sentiment"]
X_train = pd.DataFrame(index=Y_train.index)

# Using Vader
analyser = SentimentIntensityAnalyzer()

pos = pd.Series(index=Y_train.index)
neut = pd.Series(index=Y_train.index)
neg = pd.Series(index=Y_train.index)
comp = pd.Series(index=Y_train.index)
numQuest = pd.Series(index=Y_train.index)
numChar = pd.Series(index=Y_train.index)
numPer = pd.Series(index=Y_train.index)

for i in range(Y_train.size):
    snt = analyser.polarity_scores(X_raw.loc[i]["SentimentText"])
    pos[i] = snt["pos"]
    neut[i] = snt["neu"]
    neg[i] = snt["neg"]
    comp[i] = snt["compound"]
    numQuest[i] = X_raw.loc[i]["SentimentText"].count("?")
    numChar[i] = len(X_raw.loc[i]["SentimentText"])
    numPer[i] = X_raw.loc[i]["SentimentText"].count(".")


# Adding Series to TrainDF
X_train['Positive'] = pos
X_train['Neutral'] = neut
X_train['Negative'] = neg
X_train['Compound'] = comp
X_train['numQuest'] = numQuest
X_train['numChars'] = numChar
X_train['numPers'] = numPer

print(X_train.head().to_string())

X_train.to_csv("vaderAnalysisWithPeriods.csv")

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
