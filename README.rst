pulearn ⏂
#########

|PyPI-Status| |PyPI-Versions| |Build-Status| |Codecov| |Codefactor| |LICENCE|

Positive-unlabeled learning with Python.

**Website:** `https://pulearn.github.io/pulearn/ <https://pulearn.github.io/pulearn/>`_

**Documentation:** `https://pulearn.github.io/pulearn/doc/pulearn/ <https://pulearn.github.io/pulearn/doc/pulearn/>`_


.. code-block:: python

    from pulearn import ElkanotoPuClassifier
    from sklearn.svm import SVC
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
    pu_estimator.fit(X, y)


.. contents::

.. section-numbering::


Documentation
=============

This is the repository of the ``pulearn`` package, and this readme file is aimed to help potential contributors to the project.

To learn more about how to use ``pulearn``, either `visit pulearn's homepage <https://pulearn.github.io/pulearn/>`_ or read the `online documentation of pulearn <https://pulearn.github.io/pulearn/doc/pulearn/>`_.


Installation
============

Install ``pulearn`` with:

.. code-block:: bash

  pip install pulearn


Implemented Classifiers
=======================

Elkanoto
--------

Scikit-Learn wrappers for both the methods mentioned in the paper by Elkan and Noto, `"Learning classifiers from only positive and unlabeled data" <https://cseweb.ucsd.edu/~elkan/posonly.pdf>`_ (published in Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, ACM, 2008).

These wrap the Python code from `a fork by AdityaAS <https://github.com/AdityaAS/pu-learning>`_ (with implementation to both methods) to the `original repository <https://github.com/aldro61/pu-learning>`_ by `Alexandre Drouin <https://github.com/aldro61>`_ implementing one of the methods.

Unlabeled examples are expected to be indicated by `-1`, positives by `1`.

Classic Elkanoto
~~~~~~~~~~~~~~~~

To use the classic (unweighted) method, use the ``ElkanotoPuClassifier`` class:

.. code-block:: python

    from pulearn import ElkanotoPuClassifier
    from sklearn.svm import SVC
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
    pu_estimator.fit(X, y)


Weighted Elkanoto
~~~~~~~~~~~~~~~~~

To use the weighted method, use the ``WeightedElkanotoPuClassifier`` class:

.. code-block:: python

    from pulearn import WeightedElkanotoPuClassifier
    from sklearn.svm import SVC
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = WeightedElkanotoPuClassifier(
        estimator=svc, labeled=10, unlabeled=20, hold_out_ratio=0.2)
    pu_estimator.fit(X, y)

See the original paper for details on how the ``labeled`` and ``unlabeled`` quantities are used to weigh training examples and affect the learning process: `https://cseweb.ucsd.edu/~elkan/posonly.pdf <https://cseweb.ucsd.edu/~elkan/posonly.pdf>`_.

Bagging-based PU-learning
-------------------------

Based on the paper `A bagging SVM to learn from positive and unlabeled examples (2013) <http://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Mordelet2013bagging.pdf>`_ by Mordelet and Vert. The implementation is by `Roy Wright <https://roywrightme.wordpress.com/>`__ (`roywright <https://github.com/roywright/>`_ on GitHub), and can be found in `his repository <https://github.com/roywright/pu_learning>`_.

Unlabeled examples are expected to be indicated by a number smaller than `1`, positives by `1`.

.. code-block:: python

    from pulearn import BaggingPuClassifier
    from sklearn.svm import SVC
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = BaggingPuClassifier(
        base_estimator=svc, n_estimators=15)
    pu_estimator.fit(X, y)


Examples
========

A nice code example of the classic Elkan-Noto classifier used for classification on the `Wisconsin breast cancer dataset <https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)>`_ , comparing it to a regular random forest classifer, can be found in the ``examples`` directory.

To run it, clone the repository, and run the following command from the root of the repository, with a python environment where ``pulearn`` is installed:

.. code-block:: bash

    python examples/BreastCancerElkanotoExample.py

You should see a nice plot, like the one below, comparing the F1 score of the PU learner versus a naive learner, demonstrating how PU learning becomes more effective - or worthwhile - the more positive examples are "hidden" from the training set.

.. image:: https://raw.githubusercontent.com/pulearn/pulearn/master/pulearn_breast_cancer_f1_scores.png


Contributing
============

Package author and current maintainer is Shay Palachy (shay.palachy@gmail.com); You are more than welcome to approach him for help. Contributions are very welcomed, especially since this package is very much in its infancy and many other PU Learning methods can be added.

Installing for development
--------------------------

Clone:

.. code-block:: bash

  git clone git@github.com:pulearn/pulearn.git


Install in development mode with test dependencies:

.. code-block:: bash

  cd pulearn
  pip install -e ".[test]"


Running the tests
-----------------

To run the tests, use:

.. code-block:: bash

  python -m pytest


Notice ``pytest`` runs are configured by the ``pytest.ini`` file. Read it to understand the exact ``pytest`` arguments used.


Adding tests
------------

At the time of writing, ``pulearn`` is maintained with a test coverage of 100%. Although challenging, I hope to maintain this status. If you add code to the package, please make sure you thoroughly test it. Codecov automatically reports changes in coverage on each PR, and so PR reducing test coverage will not be examined before that is fixed.

Tests reside under the ``tests`` directory in the root of the repository. Each model has a separate test folder, with each class - usually a pipeline stage - having a dedicated file (always starting with the string "test") containing several tests (each a global function starting with the string "test"). Please adhere to this structure, and try to separate tests cases to different test functions; this allows us to quickly focus on problem areas and use cases. Thank you! :)

Code style
----------

``pulearn`` code is written to adhere to the coding style dictated by `flake8 <http://flake8.pycqa.org/en/latest/>`_. Practically, this means that one of the jobs that runs on `the project's Travis <https://travis-ci.org/pulearn/pulearn>`_ for each commit and pull request checks for a successfull run of the ``flake8`` CLI command in the repository's root. Which means pull requests will be flagged red by the Travis bot if non-flake8-compliant code was added.

To solve this, please run ``flake8`` on your code (whether through your text editor/IDE or using the command line) and fix all resulting errors. Thank you! :)


Adding documentation
--------------------

This project is documented using the `numpy docstring conventions`_, which were chosen as they are perhaps the most widely-spread conventions that are both supported by common tools such as Sphinx and result in human-readable docstrings (in my personal opinion, of course). When documenting code you add to this project, please follow `these conventions`_.

.. _`numpy docstring conventions`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
.. _`these conventions`: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

Additionally, if you update this ``README.rst`` file,  use ``python setup.py checkdocs`` to validate it compiles.


License
=======

This package is released as open-source software under the `BSD 3-clause license <https://opensource.org/licenses/BSD-3-Clause>`_. See ``LICENSE_NOTICE.md`` for the different copyright holders of different parts of the code.


Credits
=======

Implementations code by:

* Elkan & Noto - Alexandre Drouin and `AditraAS <https://github.com/AdityaAS>`_.
* Bagging PU Classifier - `Roy Wright <https://github.com/roywright/>`_.

Packaging, testing and documentation by `Shay Palachy <http://www.shaypalachy.com/>`_.

Fixes and feature contributions by:

* `kjappelbaum <https://github.com/kjappelbaum>`_
* `mepland <https://github.com/mepland>`_
* `TEGELB <https://github.com/TEGELB>`_


.. alternative:
.. https://badge.fury.io/py/yellowbrick.svg

.. |PyPI-Status| image:: https://img.shields.io/pypi/v/pulearn.svg
  :target: https://pypi.org/project/pulearn

.. |PyPI-Versions| image:: https://img.shields.io/pypi/pyversions/pulearn.svg
   :target: https://pypi.org/project/pulearn

.. |Build-Status| image:: https://github.com/pulearn/pulearn/actions/workflows/test.yml/badge.svg
  :target: https://github.com/pulearn/pulearn/actions/workflows/test.yml

.. |LICENCE| image:: https://img.shields.io/badge/License-BSD%203--clause-ff69b4.svg
  :target: https://pypi.python.org/pypi/pulearn

.. .. |LICENCE| image:: https://github.com/pulearn/pulearn/blob/master/mit_license_badge.svg
  :target: https://pypi.python.org/pypi/pulearn

.. https://img.shields.io/pypi/l/pulearn.svg

.. |Codecov| image:: https://codecov.io/github/pulearn/pulearn/coverage.svg?branch=master
   :target: https://codecov.io/github/pulearn/pulearn?branch=master


.. |Codacy|  image:: https://api.codacy.com/project/badge/Grade/7d605e063f114ecdb5569266bd0226cd
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/pulearn/pulearn?utm_source=github.com&utm_medium=referral&utm_content=pulearn/pulearn&utm_campaign=Badge_Grade_Dashboard

.. |Requirements| image:: https://requires.io/github/pulearn/pulearn/requirements.svg?branch=master
     :target: https://requires.io/github/pulearn/pulearn/requirements/?branch=master
     :alt: Requirements Status

.. |Downloads| image:: https://pepy.tech/badge/pulearn
     :target: https://pepy.tech/project/pulearn
     :alt: PePy stats

.. |Codefactor| image:: https://www.codefactor.io/repository/github/pulearn/pulearn/badge?style=plastic
     :target: https://www.codefactor.io/repository/github/pulearn/pulearn
     :alt: Codefactor code quality
