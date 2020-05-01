#standard import
import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
# read data
df = pd.read_csv('fake_or_real_news.csv')
df = df.set_index('Unnamed: 0')
y = df.label
df = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.3, random_state=50)
#tfidf
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
#count
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
# important features function
def most_informative_feature(vectorizer, classifier, n=10):

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]
    print("Important FAKE news features")
    for coef, feat in class1:
        print(class_labels[0], feat)
    print()
    print("Important REAL news features")
    for coef, feat in reversed(class2): #reversed order
        print(class_labels[1], feat)
# scorer function
def scorer(confusion_m):
    tn, fp, fn, tp = confusion_m.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall)
    print("Precision is: %0.3f" % precision)
    print("Recall is: %0.3f" % recall)
    print("F-1 Score is: %0.3f" % f1_score)
    print()
    
############ classification
# knn model
print("Result of K-NN model")
knn_matrix_score = []
for n_neighbors in np.arange(2,10):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(count_train, y_train)
    pred_knn = knn_model.predict(count_test)
    knn_score = metrics.accuracy_score(y_test, pred_knn)
    knn_matrix_score.append(knn_score)
knn_max_index = np.argmax(knn_matrix_score) + 2 # neighbor array start from 2
print ("Best number of neighbors is: %d" % knn_max_index)
print("Best accuracy of K-NN:   %0.3f" % knn_matrix_score[knn_max_index - 2]) #deduct two print out the right number
cm_knn = metrics.confusion_matrix(y_test, pred_knn, labels=['FAKE', 'REAL'])
scorer(cm_knn)

# rf model
print("Result of Random Forest model")
rf_matrix_score = []
for max_depth in np.arange(2,6):
    rf_model = RandomForestClassifier(max_depth=max_depth, random_state=0)
    rf_model.fit(tfidf_train, y_train)
    pred_rf = rf_model.predict(tfidf_test)
    rf_score = metrics.accuracy_score(y_test, pred_rf)
    rf_matrix_score.append(rf_score)
rf_max_index = np.argmax(rf_matrix_score) + 2 # neighbor array start from 2  
print ("Best number of max_depth is: %d" % rf_max_index) 
print("Best accuracy of RF:   %0.3f" % rf_matrix_score[rf_max_index - 2])
cm_rf = metrics.confusion_matrix(y_test, pred_rf, labels=['FAKE', 'REAL'])
scorer(cm_rf)

# nn model
print("Result of MLP Neural Net model")
nn_model = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
nn_model.fit(tfidf_train, y_train)
pred_nn = nn_model.predict(tfidf_test)
nn_score = metrics.accuracy_score(y_test, pred_nn)
print("Accuracy of Multi-layer Perceptron NN:   %0.3f" % nn_score)
cm_nn = metrics.confusion_matrix(y_test, pred_nn, labels=['FAKE', 'REAL'])
scorer(cm_nn)

########### features
# svm model
print("Result of Linear SVM model")
svm_model = LinearSVC(random_state=0, tol=1e-5)
svm_model.fit(tfidf_train, y_train)
pred_svm = svm_model.predict(tfidf_test)
svm_score = metrics.accuracy_score(y_test, pred_svm)
print("Accuracy of Linear SVM:   %0.3f" % svm_score)
cm_svm = metrics.confusion_matrix(y_test, pred_svm, labels=['FAKE', 'REAL'])
scorer(cm_svm)
most_informative_feature(tfidf_vectorizer, svm_model)

# Logistic Regression
print("Result of Logistic Regression model")
log_reg = LogisticRegression(random_state=0)
log_reg.fit(tfidf_train, y_train)
pred_log = log_reg.predict(tfidf_test)
log_score = metrics.accuracy_score(y_test, pred_log)
print("Logistic Regression accuracy:   %0.3f" % log_score)
log_cm = metrics.confusion_matrix(y_test, pred_log, labels=['FAKE', 'REAL'])
scorer(log_cm)
most_informative_feature(tfidf_vectorizer, log_reg)
