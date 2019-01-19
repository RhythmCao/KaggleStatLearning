#coding=utf8
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis 
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score

class LDAModel(object):

    def __init__(self):
        super(LDAModel, self).__init__()
        self.lda_model = None
    
    def forward(self, inputs, label=None):
        result = self.lda_model.predict(inputs)
        if label is not None:
            score = self.lda_model.score(inputs, label)
            return result, score
        return result
    
    def train(self, fit_x, fit_y):
        self.lda_model.fit(fit_x, fit_y)
        score = self.lda_model.score(fit_x, fit_y)
        return score

    def get_cv_accuracy(self, train_data, train_label, cv=5):
        return np.mean(cross_val_score(self.lda_model, train_data, train_label, cv=cv))

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def save_model(self, path):
        pickle.dump(self.lda_model, open(path, 'wb'))

    def load_model(self, path):
        self.lda_model = pickle.load(open(path, 'rb'))

class LinearDAModel(LDAModel):
    def __init__(self, solver='svd',shrinkage=None, tol=1e-4):
        super(LinearDAModel, self).__init__()
        assert solver in ['svd','lsqr','eigen']
        assert not (solver == 'svd' and shrinkage is not None)
        assert shrinkage is None or shrinkage == 'auto' or type(shrinkage) == float
        self.lda_model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, tol=tol)

class QuadraticDAModel(LDAModel):
    def __init__(self, reg_param, tol=1e-4):
        super(QuadraticDAModel, self).__init__()
        self.lda_model = QuadraticDiscriminantAnalysis(reg_param=reg_param, tol=tol)
