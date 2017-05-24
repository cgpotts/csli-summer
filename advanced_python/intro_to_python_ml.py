
# coding: utf-8

# <span style="font-size:180%;font-weight:bold">
# Introduction to machine learning with Python and scikit-learn
# </span>
# 
# __CSLI Summer internship Program, Summer 2017__<br />
# __Stanford__

# In[*]

__author__ = 'Will Monroe and Chris Potts'


# # Overview
# 
# This tutorial covers vectors and matrices with Numpy and Scipy and then uses supervised learning as a way to introduce you to the concepts and design patterns of [scikit-learn](http://scikit-learn.org/).
# 
# * If you've never programmed in Python before, this tutorial will be confusing and disorienting. Better to start with a more basic, less focused course!
# 
# * If you've programmed in Python before but have no experience with machine learning, this might be a confusing and disorienting experience, but it could still be valuable, so we suggest you give it a shot, accepting that you won't get all of it on this round.
# 
# * __If you're an experienced Python programmer who has taken a basic course in machine learning, this tutorial is for you!__
# 
# * If you're an expert at Python, machine learning, and scikit-learn, then stick around to help your peers!
# 
# We're not going to try to cover or motivate machine learning concepts. Our goals are more coding-oriented. However, we do think this kind of hands-on exploration is essential for achieving a deep understanding of those concepts.

# # Set-up
# 
# This tutorial requires `numpy >= 1.6.1`, `scipy >= 0.9`, and `sklearn >= 0.18.1`. If you're not already set up with these, we recommend installing them all via [Anaconda](https://www.continuum.io/downloads). This tutorial will work with Python 2 and Python 3. We recommend Python 3. (Anaconda will make it easy to switch between Python 2 and Python 3.)

# # Vectors

# In[*]

import numpy as np


# ## Vector initialization

# In[*]

np.zeros(5)


# In[*]

np.ones(5)


# In[*]

np.array([1,2,3,4,5])


# In[*]

np.ones(5).tolist()


# In[*]

np.array([1.0,2,3,4,5])


# In[*]

np.array([1,2,3,4,5], dtype='float')


# In[*]

np.array([x for x in range(20) if x % 2 == 0])


# __Exercise__: Create some vectors with [np.arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html).

# __Exercise__: Review the other [Numpy array-creation routines](https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html).

# __Exercise__: Create some random vectors with [np.random](https://docs.scipy.org/doc/numpy/reference/routines.random.html) methods.

# ## Vector indexing

# In[*]

x = np.array([10,20,30,40,50])


# In[*]

x[0]


# In[*]

x[0:2]


# In[*]

x[0:1000]


# In[*]

x[-1]


# In[*]

x[[0,2,4]]


# In[*]

x[[-1]]


# In[*]

x[-1:]


# __Exercise__: Write an expression that will get the antepenultimate and final elements of any array. (The antepenultimate element is 3 places back from the end. For our `x` above, you should get back `np.array([30, 50])`, but make sure your code would work for any `np.array` with at least three elements.)

# ## Vector assignment

# In[*]

x2 = x.copy()


# In[*]

x2[0] = 10

x2


# In[*]

x2[[1,2]] = 10

x2


# In[*]

x2[[3,4]] = [0, 1]

x2


# ## Vectorized operations

# In[*]

x.sum()


# In[*]

x.mean()


# In[*]

x.max()


# In[*]

x.argmax()


# In[*]

np.log(x)


# In[*]

np.exp(x)


# In[*]

x + x  # Try also with *, -, /, etc.


# In[*]

x + 1


# __Exercise__: complete the following sigmoid (logistic) function so that it works for both vector inputs and integer/float inputs. The function itself is defined as
# 
# $${\displaystyle f(x)={\frac {1}{1+e^{-x}}}}$$

# In[*]

def sigmoid(x):
    """The sigmoid function f(x) = 1 / (1 + e^-x)
        
    Parameters
    ----------
    x : int, float, or np.array
    
    Returns
    -------
    float or np.array
    """
    pass


# In[*]

assert sigmoid(0) == 0.5


# In[*]

assert (sigmoid(np.array([0.0])) == np.array([0.5])).all()


# ## Comparison with Python lists
# 
# Vectorizing your mathematical expressions can lead to __huge__ performance gains. The following example is meant to give you a sense for this. It compares applying `np.log` to each element of a list with 10 million values with the same operation done on a vector.

# In[*]

def listlog(vals):
    return [np.log(y) for y in vals]


# In[*]

samp = np.random.random_sample(int(1e7))+1


# In[*]

get_ipython().magic(u'time _ = np.log(samp)')


# In[*]

get_ipython().magic(u'time _ = listlog(samp)')


# # Matrices
# 
# The matrix is the core object of machine learning implementations. In `sklearn`, it's the most common input and output and thus a key to how the library's numerous methods can work together.

# ## Matrix initialization

# In[*]

np.array([[1,2,3], [4,5,6]])


# In[*]

np.array([[1,2,3], [4,5,6]], dtype='float')


# In[*]

np.zeros((3,5))


# In[*]

np.ones((3,5))


# In[*]

np.identity(3)


# In[*]

np.diag([1,2,3])


# ## Matrix indexing

# In[*]

X = np.array([[1,2,3], [4,5,6]])


# In[*]

X


# In[*]

X[0]


# In[*]

X[0,0]


# In[*]

X[0, : ]


# In[*]

X[ : , 0]


# In[*]

X[ : , [0,2]]


# __Exercise__: Write an expression to get the lowest, rightmost value of any matrix.

# ## Matrix assignment

# In[*]

X2 = X.copy()

X2


# In[*]

X2[0,0] = 20

X2


# In[*]

X2[0] = 3

X2


# In[*]

X2[: , -1] = [5, 6]

X2


# ## Matrix reshaping

# In[*]

z = np.arange(1, 7)

z


# In[*]

Z = z.reshape(2,3)

Z


# In[*]

Z.reshape(6)


# In[*]

Z.T


# __Exercise__: Reshape `Z` so that it has dimension $6 \times 1$ (6 rows of length 1).

# ## Numeric operations

# In[*]

A = np.array(range(1,7), dtype='float').reshape(2,3)

A


# In[*]

B = np.array([2, 2, 2])


# In[*]

A * B


# In[*]

A + B


# In[*]

A / B


# In[*]

A.dot(B)


# In[*]

B.dot(A.T)


# In[*]

A.dot(A.T)


# In[*]

np.outer(B, B)


# ## Sparse matrices with scipy
# 
# It is very common in machine learning to require matrices with very high dimensionality – so high that instantiating them as `np.array` objects is impossible because they require too much memory.
# 
# The one saving grace is that these large matrices tend to be __sparse__ – almost all of their elements are $0$. Scipy sparse matrices do not represent these 0s, which is often enough to address the memory problems.
# 
# For the most part, `scipy.sparse` matrices behave like `np.array` objects. The two main differences: they support fewer mathematical operations, and they always have two dimensions.
# 
# `sklearn` is pretty good at hiding the sparse/dense distinction from you.
# 
# This section explores `scipy.sparse` a bit. You'll see that we commonly use `toarray()` to see what the objects are like. (Not an option where sparsity is key, of course!)

# In[*]

import scipy.sparse


# In[*]

scipy.sparse.lil_matrix(2)


# In[*]

scipy.sparse.lil_matrix(2).toarray()


# In[*]

scipy.sparse.lil_matrix((1,2))


# In[*]

scipy.sparse.lil_matrix((1,2)).toarray()


# In[*]

scipy.sparse.lil_matrix([1,2]).toarray()


# In[*]

S = scipy.sparse.lil_matrix((2,3))

S


# In[*]

S[0,0] = 4
S[1,2] = 6

S


# In[*]

S.toarray()


# In[*]

S.nonzero()


# In[*]

for i, j in zip(*S.nonzero()):
    print(i, j, S[i, j])


# In[*]

C = scipy.sparse.csr_matrix(np.array([[1,2,3], [4,5,6]]))

C


# The following return `np.matrix` instances, which are like `np.array` instances but always have two dimensions. The gotcha here is that the results are dense, so one has to be careful if the original matrix is really big:

# In[*]

C.sum(axis=0)


# In[*]

C.sum(axis=1)


# Scipy sparse matrices can be multiplied by other sparse matrices as well as dense ones. If the argument is dense, so is the result.

# In[*]

C.dot(B)


# In[*]

B_sparse = scipy.sparse.csr_matrix([2,2,2])

C.dot(B_sparse.T)


# Rules of thumb:
#     
# * `lil_matrix` is fastest if you're adding things to the matrix
# * `csr_matrix` is fastest if you're doing row-wise operations.
# * `csc_matrix` is fastest if you're doing column-wise operations.
# 
# Other types: [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html).

# # Data splits
# 
# In supervised learning, the available data are split into a portion for training and another portion for testing. This section covers two `sklearn` methods for doing that.
# 
# We're just providing the mechanics. We won't really be able to cover the more complex scientific issues around how to divide up data for robust system development and accurate assessment. These are the building blocks for that work, though.

# ## A small classification dataset

# In[*]

from sklearn.datasets import load_iris


# In[*]

iris = load_iris()


# In[*]

X_iris = iris['data']

X_iris


# In[*]

y_iris = iris['target']

y_iris


# ## A small regression dataset

# In[*]

from sklearn.datasets import load_boston


# In[*]

boston = load_boston()


# In[*]

X_boston = boston['data']

X_boston


# In[*]

boston['feature_names']


# In[*]

y_boston = boston['target']

y_boston


# ## Simple train/test splits

# In[*]

from sklearn.model_selection import train_test_split


# In[*]

X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris, y_iris, train_size=0.7)


# In[*]

X_iris_train.shape


# In[*]

y_iris_train.shape


# In[*]

X_iris_test.shape


# In[*]

y_iris_test.shape


# ## Cross-validation
# 
# In $n$-fold cross validation, the data are divided into $n$ equal portions, and then $n$ evaluations are performed. In each, the system is trained on $n-1$ of the folds combined and tested on the remaining one. The scores from each of the $n$ evaluations can then be summarized in various ways (e.g., an average with a confidence interval).

# In[*]

from sklearn.model_selection import StratifiedKFold


# In[*]

skf = StratifiedKFold(n_splits=5)

skf


# In[*]

def iter_splits(X, y, n_splits=5):
    """Convenience wrapper around `StratifiedKFold`.
    
    Parameters
    ----------
    X : array-like -- the feature matrix, dimension m x n
    y : array-like -- the label vector, dimension m
    n_splits : int
    
    Yields
    ------
    all array-like: X_train, X_test, y_train, y_test    
    """
    skf = StratifiedKFold(n_splits=5) 
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, X_test, y_train, y_test


# In[*]

for train, test, _, _ in iter_splits(X_iris, y_iris):
    print("Train instances: {}; test instances: {}".format(
            train.shape[0], test.shape[0]))


# Other methods for cross-validation: [sklearn.model_selection](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection).

# # Basic preprecessing
# 
# Machine learning algorithms are often sensitive, to some degree or another, to the way the input matrix is scaled. If the columns (features) or rows (examples) are on very different scales, then the resulting model can be hard to interpret, or even misrepresent the underlying patterns. `sklearn` contains a variety of methods for addressing these scaling issues.

# In[*]

from sklearn.preprocessing import scale, normalize


# In[*]

P = np.array([[1.0, 4, 5], [5, 4, 1]])


# In[*]

normalize(P, norm='l1', axis=0)


# In[*]

scale(P, with_mean=True, with_std=True)


# Other methods for preprocessing: [sklearn.preprocessing](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing). 
# 
# See also [sklearn.decomposition](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition) and [sklearn.feature_extraction.text.TfidfTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer).

# # Models
# 
# The basic pattern for all these supervised models:
# 
# 1. Instantiate the model.
# 2. Call its `fit` method on the training data pair `(X, y)`.
# 3. Call its `predict` method on the test data `X_test`. This returns a list of predictions.
# 4. Compare the list of predictions to `y_test`.
# 
# We believe this holds for __all__ of the supervised models in `sklearn`.

# ## Classification

# In[*]

from sklearn.linear_model import LogisticRegression


# In[*]

maxent = LogisticRegression()

maxent.fit(X_iris_train, y_iris_train)


# In[*]

iris_predictions = maxent.predict(X_iris_test)


# In[*]

from sklearn.metrics import classification_report


# In[*]

fnames_iris = iris['feature_names']

fnames_iris


# In[*]

tnames_iris = iris['target_names']

tnames_iris


# In[*]

print(classification_report(y_iris_test, iris_predictions, target_names=tnames_iris))


# The model coefficients (weights) are an array $c \times p$, where $c$ is the number of classes and $p$ is the number of features.

# In[*]

maxent.coef_


# In[*]

maxent.intercept_


# In[*]

maxent.classes_


# In[*]

# Create a map from feature names to class names to weights:

features = {}

for fname, coefs in zip(fnames_iris, maxent.coef_.T):
    features[fname] = dict(zip(tnames_iris, coefs))
    
features


# ## Regression

# In[*]

from sklearn.linear_model import LinearRegression


# In[*]

X_boston_train, X_boston_test, y_boston_train, y_boston_test = train_test_split(
    X_boston, y_boston, train_size=0.7)


# In[*]

ols = LinearRegression()
ols.fit(X_boston_train, y_boston_train)


# In[*]

boston_predictions = ols.predict(X_boston_test)


# In[*]

from sklearn.metrics import r2_score


# In[*]

r2_score(y_boston_test, boston_predictions)


# In[*]

dict(zip(boston['feature_names'], ols.coef_))


# ## Model exercises

# __Exercise__: Pick a different classification model (any one!) from [sklearn.linear_model](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) and fit and evaluate it as above on the `iris` dataset.

# __Exercise__: Fit a [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) or a [RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor).

# __Exercise__: Use `iter_splits` above on the `iris` dataset to evaluate a [Support Vector Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) with 5-fold cross-validation. For each of the evaluations, use [metrics.f1_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score), store that value in a list, and report the mean with `np.mean`.
# 
# Important: you'll have to use `f1_score(y_test, predictions, average='macro')` since this is a multiclass problem. (`'micro'` is also allowed.)

# In[*]

from sklearn.svm import SVC
from sklearn.metrics import f1_score


# Instantiate the model:


# Add the scores to this list:
scores = []


# Iterate over the folds with `iter_split`. For each, use `predict` 
# to get predictions and `f1_score(y_test, predictions, average='macro')` 
# to get a score to add to `scores`.


# Use `np.mean()` to get the mean of `scores`.


# # Hyperparameter optimization
# 
# All `sklearn` models have knobs you can fiddle with. These are the __hyperparameters__ of the model. 
# 
# This term __hyperparameter__ contrasts with the model __parameters__, which are learned when `fit` is called. The hyperparameters are outside the scope of `fit`. 
# 
# For any given analysis, one should try to find the best setting of the hyperparameters – the setting that does the best with the data at hand. 
# 
# To see why this is important, suppose Will is a proponent of model $A$, and Chris thinks he has found a model $B$ that is better than $A$. To support this position, Chris should compare $A$ and $B$ on a dataset. Now, he could assure himself a win here by setting the hyperparameters poorly for $A$ and well for $B$, but this won't persuade Will. If he wants to persuade Will, he'll have to be able to say that he sought out the best setting for $A$. To do this, he'll have to explore the space of possible settings.
# 
# In the example below, we use a classification model, and we explore two related hyperparameters that concern how to __regularize__ the model. Regularization is a process for preventing the weights from becoming exaggerated, which can lead to overfitting and hence poor performance on unseen test data.

# In[*]

from sklearn.model_selection import GridSearchCV


# In[*]

cv = GridSearchCV(
    estimator=LogisticRegression(fit_intercept=True),
    param_grid={'C': [0.5, 1.0, 2.0], 'penalty': ['l1', 'l2']})

cv.fit(X_iris_train, y_iris_train)

cv.best_estimator_


# In[*]

cv.predict(X_iris_test)


# If the space of hyperparmeters is very large, it will be impractical to search the entire space. Taking random samples of the settings is known to be effective. See [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV).

# # Featurization with DictVectorizer
# 
# In the above examples, the datasets were already packaged up for us as vectors and matrices. We didn't have to decide how to represent the underlying data, and we didn't have to undertake the often messy steps of going from raw data to the objects that `sklearn` is expecting.
# 
# The [DictVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) is an extremely useful and fast method for handling these messier parts of machine learning. The basic pattern:
# 
# 1. Process your training data into a list of dictionaries, one per example.
# 2. Call the `fit_transform` method on that list of dicts. The return value is a matrix `X`.
# 3. The vectorizer itself stores the feature names derived from the underlying dicts. These are aligned with the columns of `X`.
# 4. Use `transform` to process new data into the feature space established by the training data.
# 
# Here's a small concrete example:

# In[*]

from collections import Counter
from sklearn.feature_extraction import DictVectorizer 


# In[*]

data = [
    ["the movie was good", "positive"],
    ["the movie was bad", "negative"],
    ["the movie was excellent", "positive"],
    ["the book was bad", "negative"]]


# In[*]

sentences, labels = zip(*data)


# In[*]

feats = [Counter(s.split()) for s in sentences]

feats


# In[*]

vec = DictVectorizer(sparse=False)
X_text = vec.fit_transform(feats)

X_text


# In[*]

vec.get_feature_names()


# In[*]

test_examples = [
    "the book was good",
    "the book was excellent"]


# In[*]

test_feats = [Counter(s.split()) for s in test_examples]


# In[*]

vec.transform(test_feats)


# # Mini-project: cheese or disease? 
# 
# The Stanford NLP group distributes a small labeled data set of cheese and diseases. How well can you do at predicting whether a given string is a cheese or a disease? The following starter code should help you with the high-level design of this experiment.

# ## Readers for the data
# 
# These should be ready to use:

# In[*]

import csv
import os

def _cheese_disease_iterator(filename):
    with open(filename) as f:
        for label, text in csv.reader(f, delimiter='\t'):
            label = 'cheese' if label == '1' else 'disease'
            yield label, text
            
def cheese_disease_train_iterator(
        filename=os.path.join('data', 'cheeseDisease.train.txt')):
    """Iterate over the cheese/disease training data, yielding
    (label, text) pairs.
    """
    return _cheese_disease_iterator(filename)
   
def cheese_disease_test_iterator(
        filename=os.path.join('data', 'cheeseDisease.test.txt')):
    """Iterate over the cheese/disease test data, yielding
    (label, text) pairs.
    """
    return _cheese_disease_iterator(filename)    


# In[*]

list(cheese_disease_test_iterator())[-5:]


# ## Featurize examples

# In[*]

def featurize(s):
    """Represent an example `s` as a count dict.
    
    Parameters
    ----------
    s : str
        The example to process.
    
    Returns
    -------
    dict 
        The keys are feature names, and the values are the feature 
        values -- int, float, or bool.    
    """
    return {}


# ## Use the featurizer on the training data

# In[*]

y_train = []

train_feats = []

for label, ex in cheese_disease_train_iterator():
    # Add `label` to `y_train`.
    
    # Apply `featurize` to `ex` and store the result in `train_feats`.


# ## Use the `DictVectorizer`
# 
# As we did above: convert `train_feats` to a matrix `X_cd` by using `fit_transform`.

# In[*]




# ## Train a classifier
# 
# You'll train it on `X_cd` and `y_train`.

# In[*]




# ## Use the featurizer on the test data
# 
# Just as you did above for the training data. Be sure to use `cheese_disease_test_iterator`.

# In[*]

y_test = []

test_feats = []


# ## Transform the test data into your representation space.
# 
# This is done with the `transform` method of your vectorizer, on `test_feats`. *Not* `fit_transform`, as that will remap the representation space to the test data.

# In[*]




# ## Assess your model
# 
# Use your model's `predict` method on the test matrix you created in the previous step, and compare the predictions to `y_test` using `classification_report` as we did [here](#Classification).

# In[*]




# ## Iterate
# 
# To improve performance, you can try different models, but the biggest changes are likely come from work on `featurize`. You can get as creative as you like there – the only requirement is that you translate your insights into feature dicts.
