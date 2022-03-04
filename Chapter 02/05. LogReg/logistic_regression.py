#import libraries
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    X = None
    y = None
    C = None
    sig = None

    def fit(self, X, y):
        """Fit the model according to the given training data"""
        one = np.ones(X.shape[0]).reshape(-1,1)
        self.X = X#np.concatenate((one, X), axis=1)
        self.y = y
        self.C = self.__getCoefficients(self.X, self.y)
        return self

    def __sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def __getCoefficients(self,X,y):
        xDotx = np.dot(X.T,X)
        xDotxInverse = np.linalg.inv(xDotx)
        xDotxInverseDotXT = np.dot(xDotxInverse,X.T)
        return np.dot(xDotxInverseDotXT,y)

    def __pred_single(self,y_score):
        if y_score <= 0.5:
            return 0
        else:
            return 1

    def predict_log_proba(self,X):
        """Get the probabilities"""
        one = np.ones(X.shape[0]).reshape(-1,1)
        X = X#np.concatenate((one, X), axis=1)
        self.sig = self.__sigmoid(np.dot(X,self.C))
        return self.sig

    def predict(self,X):
        """Predict class labels for samples in X"""
        sig = self.predict_log_proba(X)
        return np.array([self.__pred_single(i) for i in sig])

    def score(self,y_test,y_pred):
        """Return the mean accuracy on the given test data and labels"""
        return (sum(y_test == y_pred) / len(y_pred))*100

    def __separate_labels(self,labels, points):
        data = {}
        for id,l in enumerate(labels):
            if l not in data.keys():
                data[l] = [list(points[id])]
            else:
                data[l].append(list(points[id]))

        for k,v in data.items():
            data[k] = np.array(v,dtype=object)
        return data

    def make_plot(self,y,X,title='Title'):
        """makes plot of X and y

        Args:
            y (array(float)): vector of floats
            X (array(float)): vector of floats
            title (str, optional): title for plot. Defaults to 'Title'.
        """
        data = self.__separate_labels(y, X)
        for k,v in data.items():
            plt.scatter(x=v[:,0],y=v[:,1],label=str(k))
        plt.legend()
        plt.xlabel('x1');
        plt.ylabel('x2');
        plt.title(title);

