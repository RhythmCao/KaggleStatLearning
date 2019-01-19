#coding=utf8
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

class Classifier():
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = None
    
    def forward(self, inputs, label=None):
        result = self.classifier.predict(inputs)
        if label is not None:
            score = self.classifier.score(inputs, label)
            return result, score
        return result
    
    def train(self, fit_x, fit_y):
        self.classifier.fit(fit_x, fit_y)
        score = self.classifier.score(fit_x, fit_y)
        return score

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def get_cv_accuracy(self, train_data, train_label, cv=5):
        return np.mean(cross_val_score(self.classifier, train_data, train_label, cv=cv))

    def save_model(self, path):
        pickle.dump(self.classifier, open(path, 'wb'))

    def load_model(self, path):
        self.classifier = pickle.load(open(path, 'rb'))

class LogisticModel(Classifier):
    def __init__(self, penalty='l2', C=1.0, solver='liblinear', cv=5, tol=1e-4, random_state=999):
        super(Classifier, self).__init__()
        assert solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        if type(C) in [list, tuple]:
            self.classifier = LogisticRegressionCV(penalty=penalty, Cs=list(C), solver=solver, 
                multi_class='auto', cv=cv, tol=tol, random_state=random_state)
        else:
            self.classifier = LogisticRegressionCV(penalty=penalty, C=C, solver=solver, 
                multi_class='auto', cv=cv, tol=tol, random_state=random_state)

    def get_best_paras(self):
        return self.classifier.C_

class RidgeModel(Classifier):
    def __init__(self, alpha=1.0, cv=None):
        super(Classifier, self).__init__()
        if type(alpha) in [list,tuple]:
            self.classifier = RidgeClassifierCV(alphas=list(alpha), cv=cv)
        else:
            self.classifier = RidgeClassifier(alpha=alpha)

    def get_best_paras(self):
        return self.classifier.alpha_

class KNNModel(Classifier):
    def __init__(self, k=9, weights='uniform'):
        super(Classifier, self).__init__()
        self.classifier = KNeighborsClassifier(n_neighbors=k, weights=weights)


