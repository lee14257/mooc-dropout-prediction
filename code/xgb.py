import pandas as pd
import numpy as np
import time

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import pickle
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

# training dataset
data = np.array(pd.read_csv("dataset/features/combined_feature_train.csv"))[:,1:]
target = pd.read_csv("dataset/train/truth_train.csv", header=None, usecols=[1]).values.ravel()

# resampling
smt = SMOTETomek(ratio='auto')
data, target = smt.fit_sample(data, target)

training_start_time = time.time()

# flattening
n_samples = len(data)
data = data.reshape((n_samples, -1))

# parameter tuning
parameters = [{'n_estimators': [100, 200, 300, 400, 500],
              'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2]}]
classifier = GridSearchCV(XGBClassifier(), parameters, cv=5)
classifier.fit(data, target)

# output text file
output = open("outputs/xgb_combined_feature.txt", 'write')

print >> output, "============================ XGBoost ============================\n"
print >> output, "===== Best parameters set found on development set: ===================="
print >> output, classifier.best_params_

print >> output, "\nPeformance on each candidate H value:"
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print >> output, ("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# train the data using the tuned H parameters
best_n_parameter = classifier.best_params_['n_estimators']
best_lr_parameter = classifier.best_params_['learning_rate']
classifier = XGBClassifier(n_estimators=best_n_parameter, learning_rate=best_lr_parameter)
classifier.fit(data, target.ravel())

# save model
pickle.dump(classifier, open("outputs/xgboost_model_tuning.sav", 'wb'))

training_end_time = time.time()

# prediction on the training data
predicted = classifier.predict(data)

# compare the predicted results with the train_labels
print >> output, "======= XGBoost Classification Report for Training Combined Feature Vector: =======\n%s\n%s" % (classifier, metrics.classification_report(target, predicted))
print >> output, "Training Time: " + str(round(training_end_time - training_start_time, 3)) + "s\n"
print >> output, "Prediction Accuracy: %s\n" % (metrics.accuracy_score(target, predicted))
print >> output, "Prediction Accuracy (Balanced): %s\n" % (metrics.balanced_accuracy_score(target, predicted))
print >> output, "Prediction ROC/AUC Score: %s\n" % (metrics.roc_auc_score(target, (classifier.predict_proba(data))[:, 1]))
fpr, tpr, _ = metrics.roc_curve(target, (classifier.predict_proba(data))[:, 1])
roc_auc = metrics.auc(fpr, tpr)
print "Prediction ROC Curve for Train Data:\n"
plt.title('XGBoost ROC Curve For Training Combined Feature Vector')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print >> output, "Confusion Train matrix: \n%s\n\n" % metrics.confusion_matrix(target, predicted)


# test dataset
data = np.array(pd.read_csv("dataset/features/combined_feature_test.csv"))[:,1:]
target = pd.read_csv("dataset/test/truth_test.csv", header=None, usecols=[1]).values.ravel()

# flattening
n_samples = len(data)
data = data.reshape((n_samples, -1))

predicted = classifier.predict(data)

# compare the predicted results with the test_labels
print >> output, "======= XGBoost Classification Report for Testing Combined Feature Vector: =======\n%s\n%s" % (classifier, metrics.classification_report(target, predicted))
print >> output, "Prediction Accuracy: %s\n" % (metrics.accuracy_score(target, predicted))
print >> output, "Prediction Accuracy (Balanced): %s\n" % (metrics.balanced_accuracy_score(target, predicted))
print >> output, "Prediction ROC/AUC Score: %s\n" % (metrics.roc_auc_score(target, (classifier.predict_proba(data))[:, 1]))
fpr, tpr, _ = metrics.roc_curve(target, (classifier.predict_proba(data))[:, 1])
roc_auc = metrics.auc(fpr, tpr)
print "Prediction ROC Curve for Test Data:\n"
plt.title('XGBoost ROC Curve For Testing Combined Feature Vector')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print >> output, "Confusion Test Matrix: \n%s" % metrics.confusion_matrix(target, predicted)
