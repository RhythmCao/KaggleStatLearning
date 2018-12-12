#coding=utf8
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis 
import pickle
import numpy as np

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

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def save_model(self, path):
        pickle.dump(self.lda_model, path)

    def load_model(self, path):
        self.lda_model = pickle.load(path)