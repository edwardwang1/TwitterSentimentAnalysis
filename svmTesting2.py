import pandas as pd
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np
import os
from sklearn.externals import joblib

newModel = 1


train = pd.read_csv('data/train.csv', encoding='latin-1')
curatedData = pd.read_csv("vaderAnalysisWithPeriods.csv", encoding='latin-1')

Y_train = train["Sentiment"]
X_train = curatedData

# print(Y_train.size)
# print(X_train.shape[0])
#
# print(Y_train.head().to_string())
# print(X_train.head().to_string())

numValidation = int(Y_train.size/10)
print(numValidation)

X_valid, Y_valid = X_train[-numValidation:], Y_train[-numValidation:]
X_train, Y_train = X_train[:-numValidation], Y_train[:-numValidation]


Y_valid_arr = Y_valid.values
np.savetxt("y_valid", Y_valid_arr)



if newModel:
    #clf = svm.SVC(gamma=0.01)
    clf = LinearSVC(random_state=0, tol=1e-10, verbose=True, max_iter=10000, dual=False, C=0.1)
    clf.fit(X_train, Y_train)

else:
    clf = joblib.load('svcVader.joblib')


##SavingTrainedModel
joblib.dump(clf, 'svcVaderWithPer.joblib')

print("finished training")

clfOutputs = clf.predict(X_valid)

###Checking Accuracy
#for i in range(len(clfOutputs)):
numCorrect = 0
for i in range(len(Y_valid_arr)):
    if clfOutputs[i] == Y_valid_arr[i]:
        numCorrect += 1


print("num Correct is ", numCorrect)
np.savetxt("labelsFromVader.csv", clfOutputs)

percentCorrect = float(numCorrect) / Y_valid.size * 100

print("% Correct is: " , percentCorrect)






