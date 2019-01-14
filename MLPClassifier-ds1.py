from sklearn.neural_network import MLPClassifier
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
#
dataset_test = np.loadtxt('ds1Train.csv', delimiter=',')
train_features = [d[:-1] for d in dataset_test]
train_labels = [d[-1] for d in dataset_test]
classifier = MLPClassifier(hidden_layer_sizes=(150,), activation='logistic', solver='adam')
classifier.fit(train_features, train_labels)

dataset_valid = np.loadtxt('ds1Val.csv', delimiter=',')
valid_features = [d[:-1] for d in dataset_valid]
valid_labels = [d[-1] for d in dataset_valid]
validation_predict = classifier.predict(valid_features)

dataset_tst = np.loadtxt('ds1Test.csv', delimiter=',')
tst_features = [d[:-1] for d in dataset_valid]
tst_labels = [d[-1] for d in dataset_valid]

model_name = 'ModelMLP-ds1'
with open(model_name, 'wb') as file:
    pickle.dump(classifier, file)

with open(model_name, 'rb')as file:
    loadedModel = pickle.load(file)
result_valid = loadedModel.predict(valid_features)
result_test = loadedModel.predict(tst_features)


with open('ds1Val-3.csv', 'w') as file:
    for i in range(len(result_valid)):
        file.write('%d,%d\n' % (i + 1, result_valid[i]))


with open('ds1Test-3.csv ', 'w') as file:
    for i in range(len(result_test)):
        file.write('%d,%d\n' % (i + 1, result_test[i]))

#
# accuracy = accuracy_score(valid_labels, validation_predict)
# avgPercision = 0
# avgRecall = 0
# avgfmeasure = 0
#
# precision = precision_score(valid_labels, validation_predict, average=None)
# for i in range(len(precision)):
#     avgPercision += precision[i]
# avgPercision = avgPercision/len(precision)
#
# recall = recall_score(valid_labels, validation_predict, average=None)
# for i in range(len(recall)):
#     avgRecall += recall[i]
#
# avgRecall = avgRecall / len(recall)
#
# fmeasure = f1_score(valid_labels, validation_predict, average=None)
# for i in range(len(fmeasure)):
#     avgfmeasure += fmeasure[i]
#
# avgfmeasure = avgfmeasure / len(fmeasure)
# print(accuracy)
# print(avgPercision)
# print(avgRecall)
# print(avgfmeasure)

# ****************************Experiment part******************************
# class_name = np.arange(0, 51)
# hidden_layer_sizes = [100,110,120,130,140,150, 200, 210, 220, 230, 240, 250]
# nodes= [2,4,6]
# Solver = ['lbfgs', 'sgd', 'adam']
# Activation = ['identity', 'logistic', 'tanh', 'relu']
# min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
# min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
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
# for layer in hidden_layer_sizes:
# # nodes:
#     classifier = MLPClassifier(hidden_layer_sizes=(layer, ))
#     train_result_fmsr.append(str((layer, )))
#     classifier.fit(train_features, train_labels)
#     pred_train = classifier.predict(train_features)
#     accuracy = accuracy_score(train_labels, pred_train)
#     train_result_acc.append(accuracy)
        # fmeasure = f1_score(train_labels, pred_train, average=None)
        # for i in range(len(fmeasure)):
        #     avgfmeasure += fmeasure[i]
        #
        # avgfmeasure = avgfmeasure / len(fmeasure)
        # train_result_fmsr.append(avgfmeasure)

    # pred_valid = classifier.predict(valid_features)
    # accuracy = accuracy_score(valid_labels, pred_valid)
    # valid_result_acc.append(accuracy)
    #     # fmeasure = f1_score(valid_labels, pred_valid, average=None)
        # for i in range(len(fmeasure)):
        #     avgfmeasure += fmeasure[i]
        #
        # avgfmeasure = avgfmeasure / len(fmeasure)
        # valid_result_fmsr.append(avgfmeasure)

# avgPercision = 0
# avgRecall = 0
# avgfmeasure = 0
#
# precision = precision_score(valid_labels, pred_valid, average=None)
# for i in range(len(precision)):
#     avgPercision += precision[i]
# avgPercision = avgPercision / len(precision)
#
# recall = recall_score(valid_labels, pred_valid, average=None)
# for i in range(len(recall)):
#     avgRecall += recall[i]
#
# avgRecall = avgRecall / len(recall)
#
# fmeasure = f1_score(valid_labels, pred_valid, average=None)
# for i in range(len(fmeasure)):
#     avgfmeasure += fmeasure[i]
#
# avgfmeasure = avgfmeasure / len(fmeasure)
# print(accuracy)
# print(avgPercision)
# print(avgRecall)
# print(avgfmeasure)
# print(train_result_fmsr)
# line1, = plt.plot(train_result_fmsr, train_result_acc, 'b', label='Train Accuracy')
# line2, = plt.plot(train_result_fmsr, valid_result_acc, 'r', label='Validation Accuracy')
#
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('Accuracy')
# plt.xlabel('hidden_layer_sizes_ds1')
# plt.show()