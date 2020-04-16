import sklearn
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, plot_roc_curve, roc_curve
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import pickle
import os


import seaborn as sns

import numpy as np
import pandas as pd

import yaml
import confuse
import matplotlib.pyplot as plt



def configuration():
	# config_file = open("./config.yaml")
	# config = yaml.load(config_file, Loader = yaml.FullLoader)
	config = confuse.Configuration('/home/abishek/Financial_data',__name__)

	return config

def dataVisualize(card_data):
	'''
	card_data - card data pandas dataframe
	'''

	print(card_data.head(), "\n\n")

	print(card_data['Class'].value_counts(),"\n\n")

	zero_count = 0
	one_count = 0
	other_count = 0
	for category in card_data['Class']:
	    if category == 0:
	        zero_count += 1
	        
	    elif category == 1:
	        one_count += 1
	        
	    else:
	        other_count += 1

	print("Normal Transactions = ",zero_count*100/(zero_count+one_count+other_count),"%, ",zero_count,"instances")
	print("Fraudulent Transactions = ",one_count*100/(zero_count+one_count+other_count),"%, ",one_count,"instances")
	print("Other Transactions = ",other_count*100/(zero_count+one_count+other_count),"%, ",other_count,"instances \n\n")


	print(sns.countplot(x='Class',data=card_data))



class Classifier:
	'''
	name - type of classifier e.g. linearRegression
	card_data - pandas data frame of card data
	'''
	def __init__(self,name,card_data):
		self.card_data = card_data
		self.y = card_data['Class']
		self.X = card_data.drop(['Class','Time'],axis=1)	

		self.parameters = configuration()

		if name == 'logisticRegression':
			config = self.parameters['logisticRegression']

			self.clf = LogisticRegression(penalty = config['penalty'].get(), 
				                          dual = config['dual'].get(), 
				                          tol = config['tol'].get(), 
				                          C = config['C'].get(), 
				                          fit_intercept = config['fit_intercept'].get(), 
				                          intercept_scaling = config['intercept_scaling'].get(), 
				                          class_weight = config['class_weight'].get(), 
				                          random_state = config['random_state'].get(), 
				                          solver = config['solver'].get(), 
				                          max_iter = config['max_iter'].get(), 
				                          multi_class = config['multi_class'].get(), 
				                          verbose = config['verbose'].get(), 
				                          warm_start = config['warm_start'].get(), 
				                          n_jobs = config['n_jobs'].get(), 
				                          l1_ratio = config['l1_ratio'].get())

		elif name == 'svc':
			config = self.parameters['svc']			

			self.clf = SVC(C = config['C'].get(), 
				           kernel = config['kernel'].get(), 
				           degree = config['degree'].get(), 
				           gamma = config['gamma'].get(), 
				           coef0 = config['coef0'].get(), 
				           shrinking = config['shrinking'].get(), 
				           probability = config['probability'].get(), 
				           tol = config['tol'].get(), 
				           cache_size = config['cache_size'].get(), 
				           class_weight = config['class_weight'].get(), 
				           verbose = config['verbose'].get(), 
				           max_iter = config['max_iter'].get(), 
				           decision_function_shape = config['decision_function_shape'].get(), 
				           break_ties = config['break_ties'].get(), 
				           random_state = config['random_state'].get())


		elif name == 'randomForestClassifier':
			config = self.parameters['randomForestClassifier']

			self.clf = RandomForestClassifier(n_estimators = config['n_estimators'].get(), 
											  criterion = config['criterion'].get(), 
											  max_depth=config['max_depth'].get(), 
											  min_samples_split=config['min_samples_split'].get(), 
											  min_samples_leaf=config['min_samples_leaf'].get(), 
											  min_weight_fraction_leaf=config['min_weight_fraction_leaf'].get(), 
											  max_features=config['max_features'].get(), 
											  max_leaf_nodes=config['max_leaf_nodes'].get(), 
											  min_impurity_decrease=config['min_impurity_decrease'].get(), 
											  min_impurity_split=config['min_impurity_split'].get(), 
											  bootstrap=config['bootstrap'].get(), 
											  oob_score=config['oob_score'].get(), 
											  n_jobs=config['n_jobs'].get(), 
											  random_state=config['random_state'].get(), 
											  verbose=config['verbose'].get(), 
											  warm_start=config['warm_start'].get(), 
											  class_weight=config['class_weight'].get(), 
											  ccp_alpha=config['ccp_alpha'].get(), 
											  max_samples=config['max_samples'].get())
		elif name == 'loadSavedModel':
			print("enter path of model file")
			filename = str(input())

			while not(os.path.exists(filename)):
				raise NameError("Please enter valid model file path")

			# load the model from disk
			self.clf = pickle.load(open(filename, 'rb'))

		else:
			raise NameError("Please enter either 'logisticRegression' or 'svc' or 'randomForestClassifier'")

	def normalizeData(self):
		self.X = pd.DataFrame(scale(self.X))

	def upsampleMinorityClass(self):
		config = self.parameters['resample']

		# Separate majority and minority classes
		card_data_majority = self.card_data[self.card_data.Class==0]
		card_data_minority = self.card_data[self.card_data.Class==1]

		# Upsample minority class
		card_data_minority_upsampled = resample(card_data_minority,
		                                       replace = config['replace'].get(), # sample with replacement
		                                       n_samples = config['n_samples'].get(), # to match majority class
		                                       random_state = config['random_state'].get()) # reproducible results

		# Combine majority class with upsampled minority class
		card_data_upsampled = pd.concat([card_data_majority,card_data_minority_upsampled])


		self.X = card_data_upsampled.drop(['Class','Time'],axis=1)			
		self.y = card_data_upsampled['Class']

		# Display new class counts
		print(card_data_upsampled.Class.value_counts(),"\n\n")

	def train_test_split(self):
		config = self.parameters['train_test_split']
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,\
							             self.y,\
							             test_size = config['test_size'].get(),\
							             random_state = config['random_state'].get())


	def train(self):
		self.clf.fit(self.X_train,self.y_train)

	def predict_label(self):
		self.y_pred = self.clf.predict(self.X_test)


	def predict_probability(self):
		self.prob_y_pred = self.clf.predict_proba(self.X_test)

	def check_classifier_predicts_only_one_class(self):
		print("np.unique( self.y_pred )", np.unique( self.y_pred ),"\n\n" )

	def AreaUnderCurve(self):
		config = self.parameters['roc_auc_score']
		print("roc_auc_score",roc_auc_score(self.y_test, self.prob_y_pred[:,1], average = config['average'].get(),\
											              sample_weight = config['sample_weight'].get(),\
											              max_fpr = config['max_fpr'].get(),\
											              multi_class = config['multi_class'].get(),\
											              labels = config['labels'].get()),"\n\n")
	def Accuracy(self):
		print("accuracy_score", accuracy_score(self.y_test, self.y_pred),"\n\n" )

	def classificationReport(self):
		print("classification_report",classification_report(self.y_test,self.y_pred),"\n\n")

	def plot_ROC_curve(self):
		plot_roc_curve(self.clf, self.X_test, self.y_test)
		plt.show() 

	def plot_classifierVSrandomGuess_ROC(self):
		#Random guessing classifier
		ns_probs = np.random.randint(2, size=self.y_test.shape[0])

		# calculate roc curves
		ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
		lr_fpr, lr_tpr, _ = roc_curve(self.y_test, self.prob_y_pred[:,1])
		# plot the roc curve for the model
		plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
		plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
		# axis labels
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		# show the legend
		plt.legend()
		# show the plot
		plt.show()

	def saveModel(self,filename):
		# save the model to disk
		pickle.dump(self.clf, open(filename, 'wb'))


class stratified_Kfold_crossValidation(Classifier):

	def __init__(self,name,card_data):
		super().__init__(name,card_data) 

		config = self.parameters['StratifiedKFold']

		# Normalizing Data
		self.normalizeData()

		print("upsample minority class 'y' or 'n'")
		ans = input()
		if ans == 'y':
			self.upsampleMinorityClass()
		elif ans == 'n':
			pass
		elif ans !='y' and ans !='n':
			raise NameError("enter y or n")

		# Stratified k fold 
		skf = StratifiedKFold(n_splits=config['n_splits'].get())
		skf.get_n_splits(self.X,self.y)

		print("Strified k Fold Data:",skf,"\n\n")

		for train_index, test_index in skf.split(self.X,self.y):
			print("TRAIN:", train_index, "TEST:", test_index)
			self.X_train, self.X_test = self.X.iloc[train_index], self.X.iloc[test_index]
			self.y_train, self.y_test = self.y.iloc[train_index], self.y.iloc[test_index]

			# Training on the kth fold data set		
			self.train()

			# Predict on test set
			self.predict_label()

			# Determining probabilities on test set
			self.predict_probability()

			# Area Under Curve
			self.AreaUnderCurve()

			# Accuracy Score
			self.Accuracy()

			# Classification Report
			self.classificationReport()

			# ROC Curve Plot	
			self.plot_ROC_curve()

			# Comparing with random guess classifier
			self.plot_classifierVSrandomGuess_ROC()
