from sklearn import naive_bayes
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import itertools

# Alpha = np.arange(0.1, 1, 0.1)
#
# dataset_test = np.loadtxt('ds1Train.csv', delimiter=',')
# train_features = [d[:-1] for d in dataset_test]
# train_labels = [d[-1] for d in dataset_test]
#
# dataset_valid = np.loadtxt('ds1Val.csv', delimiter=',')
# valid_features = [d[:-1] for d in dataset_valid]
# valid_labels = [d[-1] for d in dataset_valid]
#
# train_result_acc = []
# valid_result_acc = []
# train_result_fmsr = []
# valid_result_fmsr = []
# avgfmeasure = 0
# # for alpha in Alpha:
# classifier = naive_bayes.BernoulliNB(fit_prior=False)
# classifier.fit(train_features, train_labels)
# pred_train = classifier.predict(train_features)
#
# accuracy = accuracy_score(train_labels, pred_train)
# train_result_acc.append(accuracy)
# fmeasure = f1_score(train_labels, pred_train, average=None)
# for i in range(len(fmeasure)):
#     avgfmeasure += fmeasure[i]

# avgfmeasure = avgfmeasure / len(fmeasure)
# train_result_fmsr.append(avgfmeasure)
# #
#     pred_valid = classifier.predict(valid_features)
#     accuracy = accuracy_score(valid_labels, pred_valid)
#     valid_result_acc.append(accuracy)
#     fmeasure = f1_score(valid_labels, pred_valid, average=None)
#     for i in range(len(fmeasure)):
#         avgfmeasure += fmeasure[i]
#
#     avgfmeasure = avgfmeasure / len(fmeasure)
#     valid_result_fmsr.append(avgfmeasure)

# avgPercision = 0
# avgRecall = 0
# avgfmeasure = 0
# accuracy = accuracy_score(valid_labels, pred_valid)
# line1, = plt.plot(Alpha, train_result_acc, 'b', label='Train Accuracy')
# line2, = plt.plot(Alpha, valid_result_acc, 'r', label='Validation Accuracy')
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('Accuracy')
# plt.xlabel('Alpha')
# plt.show()
# ********************Main Part **************************
dataset_test = np.loadtxt('ds1Train.csv', delimiter=',')
train_features = [d[:-1] for d in dataset_test]
train_labels = [d[-1] for d in dataset_test]

classifier = naive_bayes.BernoulliNB(alpha=0.5)
classifier.fit(train_features, train_labels)

dataset_valid = np.loadtxt('ds1Val.csv', delimiter=',')
valid_features = [d[:-1] for d in dataset_valid]
valid_labels = [d[-1] for d in dataset_valid]


dataset_tst = np.loadtxt('ds1Test.csv', delimiter=',')
tst_features = [d[:-1] for d in dataset_valid]
tst_labels = [d[-1] for d in dataset_valid]

model_name = 'ModelNB-ds1'
with open(model_name, 'wb') as file:
    pickle.dump(classifier, file)

with open(model_name, 'rb')as file:
    loadedModel = pickle.load(file)
result_valid = loadedModel.predict(valid_features)
result_test = loadedModel.predict(tst_features)


with open('ds1Val-nb.csv', 'w') as file:
    for i in range(len(result_valid)):
        file.write('%d,%d\n' % (i + 1, result_valid[i]))


with open('ds1Test-nb.csv', 'w') as file:
    for i in range(len(result_test)):
        file.write('%d,%d\n' % (i + 1, result_test[i]))

# accuracy = accuracy_score(valid_labels, result_valid)
# avgPercision = 0
# avgRecall = 0
# avgfmeasure = 0
#
# precision = precision_score(valid_labels, result_valid, average=None)
# for i in range(len(precision)):
#     avgPercision += precision[i]
# avgPercision = avgPercision/len(precision)
#
# recall = recall_score(valid_labels, result_valid, average=None)
# for i in range(len(recall)):
#     avgRecall += recall[i]
#
# avgRecall = avgRecall / len(recall)
#
# fmeasure = f1_score(valid_labels, result_valid, average=None)
# for i in range(len(fmeasure)):
#     avgfmeasure += fmeasure[i]
#
# avgfmeasure = avgfmeasure / len(fmeasure)
# print(accuracy)
# print(avgPercision)
# print(avgRecall)
# print(avgfmeasure)
