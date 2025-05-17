"""
AdaBoost implementation from scratch.
"""
import numpy as np

class DecisionStump:
    """
    DecisionStump to replace dexision trees
    """
    def __init__(self):
        self.polar = 1
        self.feature_i = None
        self.threshold = None
        self.alpha = None
        
    def predict(self, X):
        n_samples = X.shape[0]
        X_col = X[:, self.feature_i]
        preds = np.ones(n_samples)
        if self.polar == 1:
            preds[X_col < self.threshold] = -1
        else:
            preds[X_col > self.threshold] = -1
        return preds

class AdaBoost:
    """
    AdaBoost model implementation.
    """
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        weights = np.full(n_samples, (1/n_samples))
        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            
            min_error = float('inf')
            for feature_i in range(n_features):
                X_col = X[:, feature_i]
                thresholds = np.unique(X_col)
                for theshold in thresholds:
                    polar = 1
                    preds = np.ones(n_samples)
                    preds[X_col < theshold] = -1
                    
                    missclass = weights[y != preds]
                    error = sum(missclass)
                    
                    if error > 0.5:
                        error = 1 - error
                        polar = -1
                        
                    if error < min_error:
                        min_error = error
                        clf.polar = polar
                        clf.threshold = theshold
                        clf.feature_i = feature_i
            eps = 1e-10
            clf.alpha = 0.5 * np.log((1-error) / (error+eps))
            
            preds = clf.predict(X)
            weights *= np.exp(-clf.alpha * y * preds)
            weights /= np.sum(weights)
            
            self.clfs.append(clf)
                    
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred