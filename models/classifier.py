#coding=utf8
from sklearn.
import pickle
import numpy as np

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

    def save_model(self, path):
        pickle.dump(self.classifier, open(path, 'wb'))

    def load_model(self, path):
        self.classifier = pickle.load(open(path, 'rb'))


