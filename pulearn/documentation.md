

Features
========

* Scikit-learn compliant wrappers to prominent PU-learning methods.
* <a href="https://travis-ci.org/pulearn/pulearn" target="_blank"> Fully tested on Linux, macOS and Windows systems. </a>
* Compatible with Python 3.5+.



Installation
============

Install `pulearn` with:

```python
  pip install pulearn
```


Implemented Classifiers
=======================

Elkanoto
--------

Scikit-Learn wrappers for both the methods mentioned in the paper by Elkan and Noto, <a href="https://cseweb.ucsd.edu/~elkan/posonly.pdf" target="_blank"> "Learning classifiers from only positive and unlabeled data" </a>(published in Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, ACM, 2008).

These wrap the Python code from <a href="https://github.com/AdityaAS/pu-learning" target="_blank"> a fork by AdityaAS </a>(with implementation to both methods) to the <a href="https://github.com/aldro61/pu-learning" target="_blank"> original repository </a>by <a href="https://github.com/aldro61" target="_blank"> Alexandre Drouin </a>implementing one of the methods.


### Classic Elkanoto

To use the classic (unweighted) method, use the `ElkanotoPuClassifier` class:

```python
    from pulearn import ElkanotoPuClassifier
    from sklearn.svm import SVC
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
    pu_estimator.fit(X, y)
```

See<a href="elkanoto.html"> the documentation of the class </a>for more details.


### Weighted Elkanoto

To use the weighted method, use the `WeightedElkanotoPuClassifier` class:

```python
    from pulearn import WeightedElkanotoPuClassifier
    from sklearn.svm import SVC
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = WeightedElkanotoPuClassifier(
        estimator=svc, labeled=10, unlabeled=20, hold_out_ratio=0.2)
    pu_estimator.fit(X, y)
```


See<a href="https://cseweb.ucsd.edu/~elkan/posonly.pdf" target="_blank"> the original paper </a>for details on how the `labeled` and `unlabeled` quantities are used to weigh training examples and affect the learning process.

See<a href="elkanoto.html"> the documentation of the class </a>for more details.

## Examples

A nice code example of the classic Elkan-Noto classifier used for classification on the <a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)" target="_blank"> Wisconsin breast cancer dataset </a>, comparing it to a regular random forest classifer, can be found in the `examples` directory.

To run it, clone the repository, and run the following command from the root of the repository, with a python environment where `pulearn` is installed:

```bash
    python examples/BreastCancerElkanotoExample.py
```

You should see a nice plot, like the one below, comparing the F1 score of the PU learner versus a naive learner, demonstrating how PU learning becomes more effective - or worthwhile - the more positive examples are "hidden" from the training set. 

![alt text](https://raw.githubusercontent.com/pulearn/pulearn/master/pulearn_breast_cancer_f1_scores.png "Random forest with/without PU learning")
