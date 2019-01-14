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

class_name = np.arange(0, 10)
# Alpha = np.arange(0.1, 1, 0.1)
#
# dataset_test = np.loadtxt('ds2Train.csv', delimiter=',')
# train_features = [d[:-1] for d in dataset_test]
# train_labels = [d[-1] for d in dataset_test]
#
# dataset_valid = np.loadtxt('ds2Val.csv', delimiter=',')
# valid_features = [d[:-1] for d in dataset_valid]
# valid_labels = [d[-1] for d in dataset_valid]
#
# train_result_acc = []
# valid_result_acc = []
# train_result_fmsr = []
# valid_result_fmsr = []
# avgfmeasure = 0
# for alpha in Alpha:
#     classifier = naive_bayes.BernoulliNB(alpha=alpha)
#     classifier.fit(train_features, train_labels)
#     pred_train = classifier.predict(train_features)
#
#     accuracy = accuracy_score(train_labels, pred_train)
#     train_result_acc.append(accuracy)
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
# # # # Compute confusion matrix
dataset_test = np.loadtxt('ds2Train.csv', delimiter=',')
train_features = [d[:-1] for d in dataset_test]
train_labels = [d[-1] for d in dataset_test]

dataset_valid = np.loadtxt('ds2Val.csv', delimiter=',')
valid_features = [d[:-1] for d in dataset_valid]

classifier = naive_bayes.BernoulliNB(alpha=0.5)
classifier.fit(train_features, train_labels)
valid_labels = [d[-1] for d in dataset_valid]
result_valid = classifier.predict(valid_features)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cnf_matrix = confusion_matrix(valid_labels, result_valid)
np.set_printoptions(precision=0)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_name, normalize=False, title='Confusion matrix, without normalization for Naive Bayes')


print(cnf_matrix)
print(np.shape(cnf_matrix))

plt.show()
# **************************Main Part********************
# dataset_test = np.loadtxt('ds2Train.csv', delimiter=',')
# train_features = [d[:-1] for d in dataset_test]
# train_labels = [d[-1] for d in dataset_test]
#
# classifier = naive_bayes.BernoulliNB(alpha=0.5)
# classifier.fit(train_features, train_labels)
#
# dataset_valid = np.loadtxt('ds2Val.csv', delimiter=',')
# valid_features = [d[:-1] for d in dataset_valid]
# valid_labels = [d[-1] for d in dataset_valid]
#
#
# dataset_tst = np.loadtxt('ds2Test.csv', delimiter=',')
# tst_features = [d[:-1] for d in dataset_valid]
# tst_labels = [d[-1] for d in dataset_valid]
#
# model_name = 'ModelNB-ds2'
# with open(model_name, 'wb') as file:
#     pickle.dump(classifier, file)
#
# with open(model_name, 'rb')as file:
#     loadedModel = pickle.load(file)
# result_valid = loadedModel.predict(valid_features)
# result_test = loadedModel.predict(tst_features)
#
#
# with open('ds2Val-nb.cs', 'w') as file:
#     for i in range(len(result_valid)):
#         file.write('%d,%d\n' % (i + 1, result_valid[i]))
#
#
# with open('ds2Test-nb.csv', 'w') as file:
#     for i in range(len(result_test)):
#         file.write('%d,%d\n' % (i + 1, result_test[i]))

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
