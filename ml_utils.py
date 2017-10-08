import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def normalize(data):
	"""
	features whitening
	"""
	
	mean_ = np.mean(data, axis=0)
	std_ = np.std(data, axis=0)

	return (data-mean_)/std_

def prepare_data(class_0, class_1):
	"""
	1-Merge, shuffle and normalize features;
	2-Create labels arrays;
	
	Inputs:
		Class0 and Class1 features matrices;
	Outputs:
		Shuffled and normalized data matrix - [#data_points, #features]
		Labels matrix - [#data_points]
	"""
	
	n_0 = class_0.shape[0]
	n_1 = class_1.shape[0]
	n = n_0 + n_1
	
	y_0 = np.zeros(n_0)
	y_1 = np.ones(n_1)
	
	y = np.hstack([y_0, y_1])
	x_class01 = np.vstack(class_0, class_1)
	
	indexes = np.random.permutation(n)
	
	x = normalize(x_class01)
	
	return x[indexes],y[indexes]

def train_classifiers(x,y):
	"""
	Instantiate and train classification models

	Inputs:
		x features matrix;
		y labels matrix;
	Outputs:
		list of trained models
	"""
	models = []

	# Trees
	
	dtree = tree.DecisionTreeClassifier()
	dtree.fit(x, y)
	models.append(dtree)
	
	forest = RandomForestClassifier(n_estimators=10)
	forest.fit(x, y)
	models.append(forest)
	
	# Logistic Regression

	log_reg = LogisticRegression()
	log_reg.fit(x,y)
	models.append(log_reg)

	# MLP
	
	mlp = MLPClassifier(solver='adam', activation='relu', learning_rate_init=1e-3, alpha=1e-5, max_iter=1000, hidden_layer_sizes=(5, 3), random_state=1)
	mlp.fit(x, y)
	models.append(mlp)
	
	# SVMs
	
	svc_rbf1 = SVC(kernel='rbf', C=0.1, gamma=0.1)
	svc_lin1 = SVC(kernel='linear', C=0.1)
	svc_poly1 = SVC(kernel='poly', C=0.1, degree=3)
	svc_rbf1.fit(x, y)
	svc_lin1.fit(x, y)
	svc_poly1.fit(x, y)
	models.append(svc_poly1)
	models.append(svc_rbf1)
	models.append(svc_lin1)
	
	svc_rbf2 = SVC(kernel='rbf', C=10.0, gamma=0.1)
	svc_lin2 = SVC(kernel='linear', C=10.0)
	svc_poly2 = SVC(kernel='poly', C=10.0, degree=3)
	svc_rbf2.fit(x, y)
	svc_lin2.fit(x, y)
	svc_poly2.fit(x, y)
	models.append(svc_poly2)
	models.append(svc_rbf2)
	models.append(svc_lin2)
	
	svc_rbf3 = SVC(kernel='rbf', C=100.0, gamma=0.1)
	svc_lin3 = SVC(kernel='linear', C=100.0)
	svc_poly3 = SVC(kernel='poly', C=100.0, degree=3)
	svc_rbf3.fit(x, y)
	svc_lin3.fit(x, y)
	svc_poly3.fit(x, y)
	models.append(svc_poly3)
	models.append(svc_rbf3)
	models.append(svc_lin3)

	# Ensemble of all models

	ensemble_model = VotingClassifier(estimators=models_list)
	ensemble_model.fit(x,y)

	return models, ensemble_model

def predict_from_list(x, models_list):
	"""
	Instantiate and train classification models

	Inputs:
		list of models
	Outputs:
		list of predictions
	"""

	predictions = []

	for model in models_list:
		predictions.append(model.predict(x))

	return predictions

def test_from_list(x, y, models_list):
	"""
	Instantiate and train classification models

	Inputs:
		list of models
	Outputs:
		list of accuracies
	"""

	accuracies = []

	for model in models_list:
		y_pred = model.predict(x)
		accuracies.append(accuracy_score(y, y_pred))

	return accuracies
