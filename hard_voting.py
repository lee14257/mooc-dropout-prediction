#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier



#main
train_truth = np.array(pd.read_csv("Data/train/truth_train.csv", names = ["enrollment_id", "truth"]))[:,-1]
train_data = np.array(pd.read_csv("features/combined_feature_train.csv"))[:,1:]
test_truth = np.array(pd.read_csv("Data/test/truth_test.csv", names = ["enrollment_id", "truth"]))[:,-1]
test_data = np.array(pd.read_csv("features/combined_feature_test.csv"))[:,1:]


# In[2]:


#Resampling functions
def resample_smotetomek(data, truth):   
    smt = SMOTETomek(ratio='auto')
    X_smt, y_smt = smt.fit_sample(data, truth)
    return X_smt, y_smt


# In[3]:


#Resample our data
train_data, train_truth = resample_smotetomek(train_data, train_truth)
print train_data.shape, train_truth.shape


# In[4]:


#Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=150, max_depth=None, random_state=0)

#Neural network classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=200, alpha=0.0001, random_state=1)

#Adaboost
xgb_classifier = XGBClassifier(n_estimators=500, learning_rate=0.2)


# In[5]:


#Voting classifier
voting_clf = VotingClassifier(estimators = [('rf', rf_classifier), ('mlp', mlp_classifier), ('xgb', xgb_classifier)], voting = 'hard')
voting_clf.fit(train_data[:], train_truth[:])


# In[6]:


#Predict using test data
train_expected = train_truth[:]
train_predicted = voting_clf.predict(train_data[:])
test_expected = test_truth[:]
test_predicted = voting_clf.predict(test_data[:])
print train_expected.shape
print train_predicted.shape
print test_expected.shape
print test_predicted.shape


# In[7]:


#Get Accuracies
print '--Voting--'
print voting_clf
print '\nTrain Accuracy: ', accuracy_score(train_expected, train_predicted)
print 'Test Accuracy: ', accuracy_score(test_expected, test_predicted)

print("\n\nClassification report for voting(training):\n%s"
      % (metrics.classification_report(train_expected, train_predicted)))
print("Confusion matrix for Random Forest(training):\n%s" % metrics.confusion_matrix(train_expected, train_predicted))

print("\n\nClassification report for voting(test):\n%s"
      % (metrics.classification_report(test_expected, test_predicted)))
print("\nConfusion matrix for Random Forest(test):\n%s" % metrics.confusion_matrix(test_expected, test_predicted))
print "prediction accuracy(balanced): ", (metrics.balanced_accuracy_score(test_expected, test_predicted))


# In[ ]:




