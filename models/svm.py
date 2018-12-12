#coding=utf8
from sklearn import svm
import pickle
import numpy as np
class SVMModel(object):
    def __init__(self):
        super(SVMModel, self).__init__()
        self.svm_model = None
        
    def forward(self, inputs, label=None):
        result = self.svm_model.predict(inputs)
        if label is not None:
            score = self.svm_model.score(inputs, label)
            return result, score
        return result
    
    def train(self, fit_x, fit_y):
        self.svm_model.fit(fit_x, fit_y)
        score = self.svm_model.score(fit_x, fit_y)
        return score

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def save_model(self, path):
        pickle.dump(self.svm_model, path)

    def load_model(self, path):
        self.svm_model = pickle.load(path)

class SVCModel(SVMModel):

    def __init__(self, kernel='rbf', C=1, degree=3, gamma='scale', coef0=0.0,
                    tol=1e-5, decision_function_shape='ovr', random_state=999):
        super(SVCModel, self).__init__()
        assert kernel in ['rbf', 'poly', 'sigmoid', 'linear'] or callable(kernel)
        assert decision_function_shape in ['ovo','ovr']
        assert gamma in ['auto', 'scale']
        self.svm_model = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=ceof0,
                    tol=tol, decision_function_shape=decision_function_shape, random_state=random_state)

class NuSVCModel(SVMModel):

    def __init__(self, kernel='rbf', nu=0.5, degree=3, gamma='scale', coef0=0.0,
                    tol=1e-5, decision_function_shape='ovr', random_state=999):
        super(NuSVCModel, self).__init__()
        assert kernel in ['rbf', 'poly', 'sigmoid', 'linear'] or callable(kernel)
        assert decision_function_shape in ['ovo','ovr']
        assert gamma in ['auto', 'scale']
        self.svm_model = svm.NuSVC(nu=nu, kernel=kernel, degree=degree, gamma=gamma, coef0=ceof0,
                    tol=tol, decision_function_shape=decision_function_shape, random_state=random_state)

class LinearSVCModel(SVMModel):

    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr', random_state=999):
        super(LinearSVCModel, self).__init__()
        assert penalty in ['l1','l2']
        assert loss in ['squared_hinge', 'hinge']
        assert multi_class in ['ovr','crammer_singer']
        self.svm_model = svm.LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class, random_state=random_state)
