#######################################
#######################################
# -*- coding: utf-8 -*-

#######################################
#######################################

import operator
from sklearn.decomposition import PCA
# import sklearn
import pandas as pd
import warnings
import numpy as np
import os
import argparse
import autosklearn.classification
# import catboost
# from keras.models import Sequential
# from keras.layers import Dense
import sys
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.feature_selection import RFE
from catboost import *
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
# from sklearn import cross_validation
from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import hamming_loss
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
# from imblearn.ensemble import EasyEnsemble
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt    
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from skopt.space import Real, Categorical, Integer
import dask.dataframe as dd
import warnings
warnings.filterwarnings("ignore")


def preprocessing(dataset):
	global train, test, train_labels, test_labels
	dataset = pd.read_csv(dataset)
	labels = dataset['label']
	features = dataset[dataset.columns[1:25]]
	print(features)
	print(len(labels))
	# print(dataset.describe())
	train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.1,
                                                          random_state=12,
                                                          stratify=labels)
	sc = MinMaxScaler(feature_range=(0, 1))
	train = sc.fit_transform(train)
	test = sc.transform(test)
	# print(train)
	# print(test)
	# features = pd.DataFrame(features))
	# sm = ClusterCentroids(random_state=0)
	# sm = SMOTE(random_state=12)
	# sm = EditedNearestNeighbours()
	# sm = RepeatedEditedNearestNeighbours()
	# sm = AllKNN()
	# smt = RandomUnderSampler(random_state=2)
	# sm = RandomUnderSampler()
	# sm = TomekLinks(random_state=42)
	# sm = EasyEnsemble(random_state=0, n_subsets=10)
	# train, train_labels = sm.fit_sample(train, train_labels)
	# test, test_labels = smt.fit_sample(test, test_labels)
	# sc = Normalizer().fit(train)
	# sc = StandardScaler().fit(train)
	# sc = Binarizer(threshold=0.0).fit(train)
	# sc.fit(train)
	# train_normalize = sc.transform(train)
	return


def header(foutput):
	file = open(foutput, 'a')
	file.write("qParameter,Classifier,ACC,std_ACC,SE,std_SE,F1,std_F1,BACC,std_BACC,kappa,std_kappa,gmean,std_gmean")
	file.write("\n")
	return
	
	
def save_measures(classifier, foutput, scores):
	file = open(foutput, 'a')
	file.write("%s,%s,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f" % (i, classifier, scores['test_ACC'].mean(), 
	+ scores['test_ACC'].std(), scores['test_recall'].mean(), scores['test_recall'].std(), 
	+ scores['test_f1'].mean(), scores['test_f1'].std(), 
	+ scores['test_ACC_B'].mean(), scores['test_ACC_B'].std(),
	+ scores['test_kappa'].mean(), scores['test_kappa'].std(),
	+ scores['test_gmean'].mean(), scores['test_gmean'].std()))
	file.write("\n")
	return


def evaluate_model_holdout(classifier, model, finput, finput_two):
	df1 = pd.read_csv(finput)
	# df1 = pd.read_csv(finput, header=None)
	# features = df1[df1.columns[1:(len(df1.columns) - 1)]]
	train_labels = df1.iloc[:, -1]
	train = df1[df1.columns[1:(len(df1.columns) - 1)]]
	print(train)
	print(train_labels)

	df2 = pd.read_csv(finput_two)
	# df2 = pd.read_csv(finput_two, header=None)
	test_labels = df2.iloc[:, -1]
	test = df2[df2.columns[1:(len(df1.columns) - 1)]]
	print(test)
	print(test_labels)

	# imp = SimpleImputer(missing_values=-'NaN', strategy='constant', fill_value=0)
	# train = imp.fit_transform(train)
	# test = imp.transform(test)

	# train = train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
	# test = test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

	sc = StandardScaler()
	train = sc.fit_transform(train)
	test = sc.transform(test)

	# print(test)
	# sm = SMOTE(random_state=12)
	# sm = RandomUnderSampler()
	# train, train_labels = sm.fit_sample(train, train_labels)

	print("Amount of train: " + str(len(train)))
	print("Amount of test: " + str(len(test)))
	# traint, test, traint_labels, test_labels = train_test_split(train, train_labels, test_size=0.2, random_state=12, stratify=train_labels)

	n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
	max_features = ['auto', 'sqrt', 'log2']
	criterion = ['gini', 'entropy']
	max_depth = [int(x) for x in np.linspace(10, 300, num = 10)]
	max_depth.append(None)
	min_samples_split = [2, 5, 10, 14, 18, 22]
	min_samples_leaf = [1, 2, 4, 6, 8, 10]
	bootstrap = [True, False]
	
	# model = RandomForestClassifier(criterion='entropy', max_depth=30, max_features='sqrt', min_samples_leaf=4, min_samples_split=5, n_estimators=2000)

	random_grid = {'n_estimators': n_estimators,
				   'criterion': criterion,
    	           'max_depth': max_depth,
    	           'min_samples_split': min_samples_split,
    	           'min_samples_leaf': min_samples_leaf,
    	           'max_features': max_features,
    	           'bootstrap': bootstrap}

	# clf = model
	# rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
	# best_random = GridSearchCV(estimator = clf, param_grid = random_grid, cv = 10, verbose=2, n_jobs = -1)
	# best_random.fit(train, train_labels)
	# preds = best_random.predict(test)
	# rf_random.fit(train, train_labels)
	# best_random = rf_random.best_estimator_
	# print(best_random)
	# best_random.fit(train, train_labels)
	# preds = best_random.predict(test)


	# clf = autosklearn.classification.AutoSklearnClassifier(
    # time_left_for_this_task=120,
    # per_run_time_limit=30)
	# clf.fit(train, train_labels, dataset_name='peptides')
	# print(clf.show_models())

	# traint, val, traint_labels, val_labels = train_test_split(train, train_labels,
	# 														  test_size=0.3,
    #                                                         random_state=12,
    #                                                          stratify=train_labels)


	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3,1e-4,1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000,
																										  1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000, 3000]},
						{'kernel': ['sigmoid'], 'gamma': [1e-2,1e-3,1e-4,1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000,
																											  1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000, 3000]},
						{'kernel': ['linear'], 'gamma': [1e-2,1e-3,1e-4,1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000,
																											 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000, 3000]}]
	# clf = GridSearchCV(model, tuned_parameters, cv=5, refit=True, scoring='accuracy', n_jobs=-1, verbose=3, return_train_score=True)
	# clf = GridSearchCV(model, random_grid, cv=5, refit=True, scoring='accuracy', n_jobs=-1, verbose=3, return_train_score=True)
	# clf = RandomizedSearchCV(estimator=model, param_distributions=tuned_parameters, n_iter=100, cv=5, refit=True, verbose=2, random_state=42, n_jobs=-1)

	clf = model
	# feat = SelectKBest(mutual_info_classif, k=40).fit(train, train_labels)
	# train = feat.transform(train)
	# test = feat.transform(test)
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	scores = cross_validate(clf, train, train_labels, cv=kfold, scoring='accuracy')
	clf.fit(train, train_labels)
	# print(clf.keys())
	# print(clf.best_params_)
	# print(clf.best_estimator_)
	preds = clf.predict(test)
	accu = accuracy_score(test_labels, preds)
	recall = recall_score(test_labels, preds)
	f1 = f1_score(test_labels, preds)
	# if classifier == 'SVM':
	#  	auc = roc_auc_score(test_labels, clf.decision_function(test))
	# else:
	#  	auc = roc_auc_score(test_labels, clf.predict_proba(test)[:, 1])
	balanced = balanced_accuracy_score(test_labels, preds)
	gmean = geometric_mean_score(test_labels, preds)
	mcc = matthews_corrcoef(test_labels, preds)
	matriz = (pd.crosstab(test_labels, preds, rownames=["REAL"], colnames=["PREDITO"], margins=True))
	print("Classificador: %s" % (classifier))
	print("Predições %s" % (preds))
	print("Train Score (kfold=10): %s" % scores['test_score'].mean())
	print("Acurácia Teste: %s" % (accu))
	print("Recall: %s" % (recall))
	print("F1: %s" % (f1))
	# print("AUC: %s" % (auc))
	print("balanced: %s" % (balanced))
	print("gmean: %s" % (gmean))
	print("MCC: %s" % (mcc))
	print("%s" % (matriz))
	##### Importance ####
	# importances = model.feature_importances_
	# indices = np.argsort(importances)[::-1]
	# print(indices)
	# print("Feature ranking:")
	# for f in range(train.shape[1]):
	# 	print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
	# name = features.columns.values
	# names = [name[i] for i in indices]
	# print(names)
	# plt.figure()
	# plt.title("Feature importances - " + classifier)
	# plt.bar(range(features.shape[1]), importances[indices], color="r", align="center")
	# plt.xticks(range(features.shape[1]), names, rotation='vertical')
	# plt.xlim([-1, features.shape[1]])
	# plt.ylabel('Relative Importance')
	# plt.xlabel('Features')
	# plt.show()
	return


def evaluate_model_holdout_tuning(classifier, model, finput):
	colnames = np.loadtxt(finput, dtype=str, max_rows = 1, delimiter=',')
	types = []
	types.append(str)

	for i in range(len(colnames) - 2):
		types.append(np.float32)

	types.append(str)
	column_types = dict(zip(colnames, types))

	n_lines = sum(1 for row in open(finput))

	df = pd.DataFrame(columns=colnames)

	row_loops = 101 # read # lines at a time
	for i in range(1, n_lines, row_loops): 
		print(i)
		data = np.loadtxt(finput, dtype=str, skiprows=i, max_rows = row_loops, delimiter=',')
		df_new = pd.DataFrame(data[np.where(data[:,0] != 'nameseq')], columns=colnames)
		df = df.append(df_new.astype(column_types), ignore_index=True)

		del df_new
		del data

	labels = df.iloc[:, -1]
	features = df[df.columns[1:(len(df.columns) - 1)]]
	print(features)
	print(labels)
	train, test, train_labels, test_labels = train_test_split(features,
															  labels,
															  test_size=0.3,
															  random_state=12,
															  stratify=labels)
	sc = MinMaxScaler(feature_range=(0, 1))
	# sc = StandardScaler()
	train = sc.fit_transform(train)
	print(train)
	test = sc.transform(test)
	print(test)
	# print(test_labels)
	
	# sm = RandomUnderSampler()
	# train, train_labels = sm.fit_sample(train, train_labels)
	# clf = Sequential()
	# clf.add(Dense(12, input_dim=64, activation='relu'))
	# clf.add(Dense(8, activation='relu'))
	# clf.add(Dense(1, activation='sigmoid'))
	# clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# clf.fit(train, train_labels, epochs=1000, batch_size=10)
	# preds = clf.predict_classes(test)
	
	clf = model
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	scores = cross_validate(clf, train, train_labels, cv=kfold, scoring='accuracy')
	clf.fit(train, train_labels)
	preds = clf.predict(test)
	accu = accuracy_score(test_labels, preds)
	recall = recall_score(test_labels, preds)
	f1 = f1_score(test_labels, preds)
	if classifier == "SVM":
		auc = roc_auc_score(test_labels, clf.decision_function(test))
	else:
		auc = roc_auc_score(test_labels,clf.predict_proba(test)[:,1])
	balanced = balanced_accuracy_score(test_labels, preds)
	gmean = geometric_mean_score(test_labels, preds)
	mcc = matthews_corrcoef(test_labels, preds)
	matriz = (pd.crosstab(test_labels, preds, rownames=["REAL"], colnames=["PREDITO"], margins=True))
	print("Classificador: %s" % (classifier))
	print("Predições %s" % (preds))
	print("Train Score (kfold=10): %s" % scores['test_score'].mean())
	print("Acurácia Teste: %s" % (accu))
	print("Recall: %s" % (recall))
	print("F1: %s" % (f1))
	print("AUC: %s" % (auc))
	print("balanced: %s" % (balanced))
	print("gmean: %s" % (gmean))
	print("MCC: %s" % (mcc))
	print("%s" % (matriz))
	return	


def evaluate_model_holdout_multi(classifier, model, finput):
	df = pd.read_csv(finput)	
	labels = df.iloc[:, -1]
	features = df[df.columns[1:(len(df.columns) - 1)]]
	print(features)
	print(labels)
	train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.1,
                                                          random_state=12,
                                                          stratify=labels)
	sc = MinMaxScaler(feature_range=(0, 1))
	train = sc.fit_transform(train)
	print(train)
	test = sc.transform(test)
	# print(test)
	print(test_labels)
	clf = model
	clf.fit(train, train_labels)
	preds = clf.predict(test)
	recall = recall_score(test_labels, preds, pos_label='positive', average='macro')
	f1 = f1_score(test_labels, preds, pos_label='positive', average='macro')
	matriz = (pd.crosstab(test_labels, preds, rownames=["REAL"], colnames=["PREDITO"], margins=True))
	print("Classificador: %s" % (classifier))
	print("Predições %s" % (preds))
	print("Recall: %s" % (recall))
	print("F1: %s" % (f1))
	print("%s" % (matriz))
	return


def evaluate_model_cross(classifier, model, finput):
	#####################################
	colnames = np.loadtxt('D1/header.csv', dtype=str, max_rows = 1, delimiter=',')
	types = []
	types.append(str)

	for i in range(len(colnames) - 2):
		types.append(np.float32)

	types.append(str)
	column_types = dict(zip(colnames, types))

	#n_lines = sum(1 for row in open(finput))

	df = dd.read_csv(finput, dtype = column_types, names = colnames)

	X = df.iloc[:, 1:-1]
	print(X)
	y = df.iloc[:, -1]
	# y = df['label']
	print(y)
	#####################################
	pipe = Pipeline(steps=[
		('StandardScaler', StandardScaler()),
		('clf', model)])
	scoring = {'ACC': 'accuracy', 'recall': 'recall', 'f1': 'f1', 'ACC_B': 'balanced_accuracy', 'kappa': make_scorer(cohen_kappa_score), 'gmean': make_scorer(geometric_mean_score)}
	kfold = KFold(n_splits=10, shuffle=True, random_state=42)
	scores = cross_validate(pipe, X, y, cv=kfold, scoring=scoring)
	save_measures(classifier, foutput, scores)
	y_pred = cross_val_predict(pipe, X, y, cv=kfold)
	conf_mat = (pd.crosstab(y, y_pred, rownames=["REAL"], colnames=["PREDITO"], margins=True))
	# conf_mat = confusion_matrix(y, y_pred)
	print(conf_mat)
	# np.savetxt("scoresACC.csv", scores['test_ACC'], delimiter=",")
	return



##########################################################################
##########################################################################
if __name__ == "__main__":
	print("\n")
	print("###################################################################################")
	print("#####################   Arguments: -i input -o output -l    #######################")
	print("##########              Author: Robson Parmezan Bonidia                 ###########")
	print("###################################################################################")
	print("\n")
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='csv format file, E.g., dataset.csv')
	parser.add_argument('-o', '--output', help='CSV format file, E.g., test.csv')
	# parser.add_argument('-k', '--kmer', help='Range of k-mer, E.g., 1-mer (1) or 2-mer (1, 2) ...')
	# parser.add_argument('-e', '--entropy', help='Type of Entropy, E.g., Shannon or Tsallis')
	# parser.add_argument('-q', '--parameter', help='Tsallis - q parameter')
	args = parser.parse_args()
	finput = str(args.input)
	foutput = str(args.output)
	estimators = [('rf', RandomForestClassifier()),
				  ('Cat', CatBoostClassifier(logging_level='Silent')),
				  ('LR', LogisticRegression()),
				  ('AB', AdaBoostClassifier()),
				  ('KNN', KNeighborsClassifier())]
	experiments = { 
		# "GaussianNB" : GaussianNB(),
		# "DecisionTree" : DecisionTreeClassifier(criterion='gini', max_depth=2, max_leaf_nodes=2, random_state=63),
		# "GradientBoosting" : GradientBoostingClassifier(n_estimators=400, learning_rate=3.0, max_depth=1, random_state=63),
		# "RandomForest" : RandomForestClassifier(n_estimators=200),
		# "LogisticRegression" : LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5),
		# "SVM" : svm.SVC(),
		# "Bagging" : BaggingClassifier(svm.SVC(kernel='linear', C=1200, gamma=0.01)),
		# "Bagging" : BaggingClassifier(CatBoostClassifier(iterations=500, thread_count=-1, logging_level='Silent')),
		# "KNN" : KNeighborsClassifier(),
		# "Adaboost" : AdaBoostClassifier(),
		# "MLP" : MLPClassifier(),
		# "Catboost" : CatBoostClassifier(thread_count=2, verbose= True),
		"Catboost" : CatBoostClassifier(iterations=1000, thread_count=-1, logging_level='Silent'),
		# "HistGradientBoosting" : HistGradientBoostingClassifier(random_state=63),
		# "Stacking" : StackingClassifier(estimators = estimators, final_estimator = svm.SVC())
		# "RandomForest" : RandomForestClassifier(random_state=63, n_estimators=300, max_features='sqrt', criterion='entropy', max_depth=10)
		"RandomForest" : RandomForestClassifier(random_state=63, n_estimators=100),
		# "MLP" : MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 2), learning_rate_init=0.001, random_state=63),
		# "Catboost" : CatBoostClassifier(iterations=100, random_seed=63, logging_level = 'Silent')
	}
	# foutput = "results_Covid1.csv"
	header(foutput)
	for i in np.arange(6.0, 6.1, 1.0):
		i = round(i, 1)
		print("Round: %s" % (i))
		# finput = "COVID-19/q/" + str(i) + ".csv"
		# finput = "train_other_viruses.csv"
		print(finput)
		for classifier, model in experiments.items():
			# print(classifier)
			# print(model)
			#evaluate_model_holdout_tuning(classifier, model, finput)
			evaluate_model_cross(classifier, model, finput)
			# evaluate_model_holdout(classifier, model, finput, finput_two)
			# evaluate_model_holdout_multi(classifier, model, finput)
##########################################################################
##########################################################################
