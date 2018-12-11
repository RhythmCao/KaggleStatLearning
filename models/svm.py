#coding=utf8
from sklearn import svm

class SVMModel(object):
    def __init__(self, svm, kernel, C, degree=3, gamma='scale', coef0=0.0,
                    tol=1e-5, decision_function_shape='ovr', random_state=999):
        super(SVM, self).__init__()
        assert svm in ['svc','nusvc','linearsvc']
        assert kernel in ['linear','sigmoid','rbf','poly']
        assert gamma in ['auto','scale']
        assert decision_function_shape in ['ovr', 'ovo']
        if svm == 'svc':
            self.svm_model = svm.SVC(C=C,degree=3, gamma='scale', coef0=0.0,
                    tol=1e-5, decision_function_shape='ovr')
        elif svm == 'nusvc':
            self.svm_model = svm.NuSVC()
        else:
            self.svm_model = svm.LinearSVC()
        
    def forward(self, inputs, label=None):

    
    def train(self, fix_x, fit_y):
        self.svm_model.fit()


    
    def __call__(self, *input, **kwargs):
        return self.forwar(*input, **kwargs)

    def save_model(self, save_dir):
        self.svm_model = pickle.dump(save_dir)

    def load_model(self, load_dir):
        self.svm_model = pickle.load(load_dir)