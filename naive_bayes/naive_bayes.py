"""
Naive-Bayes implementation from scratch.
"""
import numpy as np

class GaussianNaiveBayes:
    """
    Naive Bayes model implementation.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        #init mean var prior
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)
        
        for c in self._classes:
            X_c = X[c==y]
            self._mean[c:] = X_c.mean(axis=0)
            self._var[c:] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_preds = []
        for x in X:
            posteriors = []
            for i, c in enumerate(self._classes):
                prior = np.log(self.priors[i])
                class_conditional = np.sum(np.log(self.pdf(i, x)))
                posteriors.append(prior + class_conditional)
            y_preds.append(self._classes[np.argmax(posteriors)])
        return np.array(y_preds)
    
    def predict_proba(self, X):
        y_probs=[]
        for x in X:
            log_posteriors = []
            for c in self._classes:
                prior = np.log(self.priors[c])
                class_conditional = np.sum(np.log(self.pdf(c, x)))
                log_posteriors.append(prior + class_conditional)
            # convert log posteriors back to normal space
            log_posteriors = np.array(log_posteriors)
            posteriors = np.exp(log_posteriors - np.max(log_posteriors)) # softmax trick for stability
            y_probs.append(posteriors / np.sum(posteriors)) # normalize
        return np.array(y_probs)
    
    def pdf(self, i, x):
        return (np.exp(-((x - self._mean[i]) ** 2) / (2 * (self._var[i]))) / np.sqrt(2 * np.pi * self._var[i]))
