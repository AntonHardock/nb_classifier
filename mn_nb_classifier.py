import numpy as np
from collections import Counter

class NaiveBayesMN:
    '''Multinomial Naive Bayes Classifier for text data.
    For each text unit, only unique words are considered.
    The Class is instantiated with parameter "k",
    which is used to smooth zero counts.
    
    The implementation is based on Joel Grus' spam filter example 
    from his great book "Data Science From Scratch" (2015).
    Deviating from the source, this implementation...
        a) ...generalizes for n classes
        b) ...estimates priors from the training data by default
        c) ...uses numpy instead of base python
    
    For the overall code structure, I borrowed from 
    the "ML from scratch tutorials" by Python Engineer 
    (https://www.youtube.com/c/PythonEngineer).
    '''

    def __init__(self, k):
        self.k = k 

    def train(self, X, y):

        # derive classes and vocabulary from training data
        self.classes = np.unique(y)
        self.vocabulary = np.unique(np.concatenate(X))

        # calculate priors and word probabilities for each class
        n_classes = len(self.classes)
        n_samples = X.shape[0]
        self.priors = np.zeros(n_classes, dtype=np.float64)
        self.w_probs = np.zeros((len(self.vocabulary), n_classes), dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            
            # subset and count text entries
            X_sub = X[y == c]
            class_count = X_sub.shape[0]

            # calculate prior 
            self.priors[idx] = class_count / n_samples

            # initialize Counter with full vocabulary AND identical order
            w_counts = Counter({word: 0. for word in self.vocabulary})
            
            # count words in X_sub
            w_counts.update(np.concatenate(X_sub))
            self.w_probs[:, idx] = list(w_counts.values())

            # calculate w_probs with smoothing factor k
            self.w_probs[:, idx] = (
                (self.w_probs[:, idx] + self.k) /
                (class_count + 2 * self.k)
            )

    def set_priors(self, prior_list):        
        '''
        Override priors after training, if needed.
        '''
        if len(prior_list) != len(self.classes):
            raise ValueError("Number of priors in prior_list does not match number of classes")
        
        self.priors = np.array(prior_list)


    def predict(self, X, calculate_posteriors=False):
        '''For each text unit in array X, a 1d boolean array is constructed.
        It's elements indicate wether word i of the training vocabulary is present. 
        Effectively, each array corresponds to a row in a document-term-matrix.
        Instead of creating the full matrix, each row is yielded through a generator.  
        
        If word i is not present, 1 - p(word i | c) is used for prediction.        
        By creating a new axis, transposing and repeating the boolean array,
        below generator pipeline yields a mask matching the shape of "w_probs". 
        This allows a vectorized calculation of log likelihoods / posteriors across all classes.
        '''
        n_classes = len(self.classes)

        masks = (np.isin(self.vocabulary, x, assume_unique=True) for x in X)
        masks = (mask[np.newaxis].T for mask in masks)
        masks = (np.repeat(mask, n_classes, axis=1) for mask in masks)

        if calculate_posteriors:
            return np.array([self._posterior(self.priors, self.w_probs, mask) for mask in masks])
        else:
            return np.array([self._predict(self.priors, self.w_probs, mask) for mask in masks])

    def _predict(self, priors, w_probs, mask):
        
        loglikelihoods = (np.log(priors) + 
                        np.where(
                            mask,
                            np.log(w_probs),
                            np.log(1 - w_probs))
                        .sum(axis=0))
                        
        return self.classes[np.argmax(loglikelihoods)]

    def _posterior(self, priors, w_probs, mask):
        '''Calculates posteriors for each text unit and each class.
        The "column order" of resulting nd array matches 
        the class order in self.classes
        '''
        loglikelihoods = (np.log(priors) + 
                        np.where(
                            mask,
                            np.log(w_probs),
                            np.log(1 - w_probs))
                        .sum(axis=0))
                        
        likelihoods = np.exp(loglikelihoods)
        posteriors = likelihoods / likelihoods.sum()
        return posteriors










 


