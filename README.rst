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

This is the repository for the ``pulearn`` package. The readme file is aimed at helping contributors to the project.

To learn more about how to use ``pulearn``, either `visit pulearn's homepage <https://pulearn.github.io/pulearn/>`_ or read the documentation at <https://pulearn.github.io/pulearn/doc/pulearn/>`_.


Installation
============

Install ``pulearn`` with:

.. code-block:: bash

  pip install pulearn


Extending ``pulearn``
=====================

New learner work now goes through a small registry and contributor scaffold:

- ``pulearn.get_algorithm_registry()`` exposes discoverable metadata for the
  built-in learners.
- ``doc/new_algorithm_checklist.md`` defines the required contributor steps.
- ``doc/templates/new_algorithm_doc_stub.md`` is the docs page starting point.
- ``tests/templates/test_new_algorithm_template.py.tmpl`` and
  ``tests/templates/test_api_contract_template.py.tmpl`` provide regression
  and shared API contract scaffolds.
- ``benchmarks/templates/benchmark_entry_template.py.tmpl`` is the benchmark stub
  until the benchmark harness lands in the dedicated roadmap milestone.
- ``pulearn.get_scaffold_templates()`` resolves those scaffold files only
  from a repository checkout and fails clearly when they are unavailable.

At minimum, every new learner should register metadata, add focused tests,
run the shared API contract checks when it inherits from
``BasePUClassifier``, and add docs plus a benchmark placeholder in the same
PR.


Implemented Classifiers
=======================

Elkanoto
--------

Scikit-Learn wrappers for both the methods mentioned in the paper by Elkan and Noto, `"Learning classifiers from only positive and unlabeled data" <https://cseweb.ucsd.edu/~elkan/posonly.pdf>`_ (published in Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, ACM, 2008).

These wrap the Python code from `a fork by AdityaAS <https://github.com/AdityaAS/pu-learning>`_ (with implementation to both methods) to the `original repository <https://github.com/aldro61/pu-learning>`_ by `Alexandre Drouin <https://github.com/aldro61>`_ implementing one of the methods.

PU labels are normalized to a canonical internal representation
(``1`` = labeled positive, ``0`` = unlabeled). Accepted input conventions
include ``{1, -1}``, ``{1, 0}``, and ``{True, False}``.
Use ``pulearn.normalize_pu_labels(...)`` to normalize labels immediately at
ingest or estimator/metric boundaries.

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

Accepted PU label conventions match the package-wide contract:
``1``/``True`` for labeled positives and ``0``/``-1``/``False`` for
unlabeled examples.

.. code-block:: python

    from pulearn import BaggingPuClassifier
    from sklearn.svm import SVC
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = BaggingPuClassifier(
        estimator=svc, n_estimators=15)
    pu_estimator.fit(X, y)


Bayesian PU Classifiers
-----------------------

Bayesian classifiers for PU learning based on the MIT-licensed
`Bayesian Classifiers for PU Learning <https://github.com/chengning-zhang/Bayesian-Classifers-for-PU_learning>`_
project by Chengning Zhang.

All four classifiers accept the same package-wide PU label conventions:
``{1, 0}``, ``{1, -1}``, and ``{True, False}``. Continuous features are
automatically discretized into equal-width bins.

Positive Naive Bayes (PNB)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest Bayesian PU classifier.  Class-conditional distributions
:math:`P(x|y=1)` and :math:`P(x|y=0)` are estimated from the labeled positives
and the unlabeled set (treated as approximate negatives) respectively, with
Laplace smoothing controlled by ``alpha``.

.. code-block:: python

    from pulearn import PositiveNaiveBayesClassifier

    clf = PositiveNaiveBayesClassifier(alpha=1.0, n_bins=10)
    clf.fit(X_train, y_pu)           # y_pu: 1 = labeled positive, 0 = unlabeled
    proba = clf.predict_proba(X_test)  # shape (n_samples, 2): [P(y=0|x), P(y=1|x)]
    labels = clf.predict(X_test)

Weighted Naive Bayes (WNB)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Extends PNB by weighting each feature's log-likelihood contribution by its
empirical mutual information with the PU label.  Features that are more
informative receive higher weights; all weights are non-negative and sum to 1.

.. code-block:: python

    from pulearn import WeightedNaiveBayesClassifier

    clf = WeightedNaiveBayesClassifier(alpha=1.0, n_bins=10)
    clf.fit(X_train, y_pu)
    print(clf.feature_weights_)      # normalized MI weight per feature (sums to 1)
    proba = clf.predict_proba(X_test)

Positive Tree-Augmented Naive Bayes (PTAN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extends PNB by replacing the naive feature-independence assumption with a
tree structure learned via the Chow-Liu algorithm.  Pairwise conditional
mutual information :math:`I(X_i; X_j \mid S)` is computed for all feature
pairs, and a maximum spanning tree is built with Prim's algorithm.  Each
non-root feature depends on exactly one parent feature in addition to the
class label.

.. code-block:: python

    from pulearn import PositiveTANClassifier

    clf = PositiveTANClassifier(alpha=1.0, n_bins=10)
    clf.fit(X_train, y_pu)
    print(clf.tan_parents_)          # parent index per feature; -1 for the root
    proba = clf.predict_proba(X_test)

Weighted Tree-Augmented Naive Bayes (WTAN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combines PTAN's tree structure with WNB's per-feature MI weighting.

.. code-block:: python

    from pulearn import WeightedTANClassifier

    clf = WeightedTANClassifier(alpha=1.0, n_bins=10)
    clf.fit(X_train, y_pu)
    print(clf.feature_weights_)      # normalized MI weight per feature
    print(clf.tan_parents_)          # learned tree structure
    proba = clf.predict_proba(X_test)

A complete end-to-end example on the Wisconsin breast cancer dataset can be
found in the ``examples`` directory:

.. code-block:: bash

    python examples/BayesianPULearnersExample.py


Prior Estimation
----------------

``pulearn.priors`` now exposes a small, unified class-prior estimation API
for SCAR workflows:

- ``LabelFrequencyPriorEstimator`` returns the observed labeled-positive rate
  as a naive lower bound for :math:`\pi`.
- ``HistogramMatchPriorEstimator`` fits a probabilistic scorer and matches
  labeled-positive vs. unlabeled score histograms.
- ``ScarEMPriorEstimator`` runs a soft-label EM refinement loop over latent
  positives in the unlabeled pool.

All three implement ``fit(X, y)`` and ``estimate(X, y)`` and return a
``PriorEstimateResult`` with the estimated prior, sample counts, and
method-specific metadata.

.. code-block:: python

    from pulearn import (
        HistogramMatchPriorEstimator,
        LabelFrequencyPriorEstimator,
        ScarEMPriorEstimator,
    )

    baseline = LabelFrequencyPriorEstimator().estimate(X_train, y_pu)
    histogram = HistogramMatchPriorEstimator().estimate(X_train, y_pu)
    scar_em = ScarEMPriorEstimator().estimate(X_train, y_pu)

    print(baseline.pi, histogram.pi, scar_em.pi)
    print(scar_em.metadata["c_estimate"])

Bootstrap confidence intervals are available when you need uncertainty
estimates or reproducible sensitivity checks:

.. code-block:: python

    estimator = ScarEMPriorEstimator().fit(X_train, y_pu)
    result = estimator.bootstrap(
        X_train,
        y_pu,
        n_resamples=200,
        confidence_level=0.95,
        random_state=7,
    )

    print(result.confidence_interval.lower, result.confidence_interval.upper)

Diagnostics helpers can summarize estimator stability across a parameter
sweep and optionally drive sensitivity plots:

.. code-block:: python

    from pulearn import (
        HistogramMatchPriorEstimator,
        diagnose_prior_estimator,
    )

    diagnostics = diagnose_prior_estimator(
        HistogramMatchPriorEstimator(),
        X_train,
        y_pu,
        parameter_grid={"n_bins": [8, 12, 20], "smoothing": [0.5, 1.0]},
    )

    print(diagnostics.unstable, diagnostics.warnings)
    print(diagnostics.range_pi, diagnostics.std_pi)

    # Optional: requires matplotlib
    # from pulearn import plot_prior_sensitivity
    # plot_prior_sensitivity(diagnostics)


Evaluation Metrics
==================

Standard binary classification metrics (precision, recall, F1) are systematically
biased in PU settings: a trivial classifier that predicts *positive for every sample*
achieves recall = 1.0 with no penalty for false positives in the unlabeled pool.
``pulearn.metrics`` provides unbiased estimators that cover the full evaluation
lifecycle under the **SCAR** (Selected Completely At Random) assumption.

.. note::

   All PU metrics assume SCAR as their baseline.  If selection bias is present
   (SAR / SNAR settings) you will need inverse propensity weighting on top of
   these estimators.

Calibration
-----------

Before computing confusion-matrix metrics you must map the model's observed output
:math:`P(s=1|x)` to the calibrated posterior :math:`P(y=1|x)`.

.. code-block:: python

   from pulearn.metrics import estimate_label_frequency_c, calibrate_posterior_p_y1

   # 1. Estimate the propensity score c = P(s=1 | y=1)
   c_hat = estimate_label_frequency_c(y_pu, s_proba)

   # 2. Calibrate: P(y=1|x) ≈ P(s=1|x) / c
   p_y1 = calibrate_posterior_p_y1(s_proba, c_hat)

``estimate_label_frequency_c`` implements the Elkan-Noto estimator
:math:`\hat{c} \approx \mathbb{E}[P(s=1|x) \mid s=1]`.
``calibrate_posterior_p_y1`` clips :math:`P(s=1|x) / \hat{c}` to :math:`[0, 1]`.

Expected-Confusion Metrics
---------------------------

These metrics reconstruct the confusion matrix from calibrated posteriors rather
than treating unlabeled data as confirmed negatives.

.. code-block:: python

   from pulearn.metrics import (
       pu_recall_score,
       pu_precision_score,
       pu_f1_score,
       pu_specificity_score,
   )

   # Recall on labeled positives (no class prior needed)
   rec  = pu_recall_score(y_pu, y_pred)

   # Unbiased precision and F1 require the class prior π
   prec = pu_precision_score(y_pu, y_pred, pi=0.3)
   f1   = pu_f1_score(y_pu, y_pred, pi=0.3)

   # Expected specificity — returns 0.0 for any all-positive classifier
   spec = pu_specificity_score(y_pu, y_score)

``pu_specificity_score`` is particularly useful as a sanity-check guard: a
degenerate classifier that assigns every sample to the positive class obtains
:math:`\text{spec} = 0`, immediately flagging the model as trivial.

Ranking Metrics
---------------

Two AUC-based metrics correct for the absence of ground-truth negatives.

.. code-block:: python

   from pulearn.metrics import pu_roc_auc_score, pu_average_precision_score

   # Sakai (2018) adjustment: AUC_pn = (AUC_pu − 0.5π) / (1 − π)
   auc = pu_roc_auc_score(y_pu, y_score, pi=0.3)

   # Area Under Lift: AUL = 0.5π + (1 − π) · AUC_pu
   aul = pu_average_precision_score(y_pu, y_score, pi=0.3)

``pu_roc_auc_score`` maps the biased :math:`AUC_{pu}` (computed against PU labels)
to an unbiased estimator of the true positive-vs-negative AUC.
``pu_average_precision_score`` returns the Area Under Lift (AUL), which is more
robust to severe class imbalance.

Risk Estimators
---------------

For flexible models such as deep networks, raw risk estimators are suitable for
early stopping and model selection in lieu of black-box accuracy metrics.

.. code-block:: python

   from pulearn.metrics import pu_unbiased_risk, pu_non_negative_risk

   # uPU: pi * R_p+ + R_u- - pi * R_p-  (du Plessis et al., 2015)
   risk_upu  = pu_unbiased_risk(y_pu, y_score, pi=0.3)

   # nnPU: clamps negative component to zero to prevent over-fitting
   #        (Kiryo et al., 2017)
   risk_nnpu = pu_non_negative_risk(y_pu, y_score, pi=0.3)

Both functions accept a ``loss`` argument (currently ``"logistic"``).

Diagnostics
-----------

Two utility functions help detect *why* a model may be performing poorly.

.. code-block:: python

   from pulearn.metrics import pu_distribution_diagnostics, homogeneity_metrics

   # KL divergence between labeled and unlabeled score distributions
   # Near-zero divergence → model cannot separate positives from unlabeled
   diag = pu_distribution_diagnostics(y_pu, y_score)
   print(diag["kl_divergence"])

   # STD and IQR of predicted-negative scores
   # Low STD/IQR → model may be over-relying on trivial features
   hom_metrics = homogeneity_metrics(y_pu, y_score)
   print(hom_metrics["std"], hom_metrics["iqr"])

Scikit-learn Integration
------------------------

``make_pu_scorer`` wraps any PU metric as a ``make_scorer``-compatible callable,
enabling direct use with ``GridSearchCV`` and ``RandomizedSearchCV``.

.. code-block:: python

   from sklearn.model_selection import GridSearchCV
   from pulearn.metrics import make_pu_scorer

   scorer = make_pu_scorer("pu_f1", pi=0.3)

   gs = GridSearchCV(estimator, param_grid, scoring=scorer)
   gs.fit(X_train, y_pu_train)

Supported metric names for ``make_pu_scorer``:

==========================================  ============================================
``"lee_liu"``                               Lee & Liu score (no ``pi`` required)
``"pu_recall"``                             PU recall (no ``pi`` required)
``"pu_precision"``                          Unbiased PU precision
``"pu_f1"``                                 Unbiased PU F1
``"pu_specificity"``                        Expected specificity
``"pu_roc_auc"``                            Adjusted ROC-AUC (Sakai 2018)
``"pu_average_precision"``                  Area Under Lift (AUL)
``"pu_unbiased_risk"``                      uPU risk (lower is better)
``"pu_non_negative_risk"``                  nnPU risk (lower is better)
==========================================  ============================================

Risk metrics are wrapped with ``greater_is_better=False`` so that
``GridSearchCV`` correctly minimises them.

Complete Example
----------------

An end-to-end demo comparing naive F1 inflation vs. corrected metrics on
synthetic SCAR data can be found in the ``examples`` directory:

.. code-block:: bash

   python examples/PUMetricsEvaluationExample.py


Examples
========

A nice code example of the classic Elkan-Noto classifier used for classification on the `Wisconsin breast cancer dataset <https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)>`_ , comparing it to a regular random forest classifier, can be found in the ``examples`` directory.

To run it, clone the repository, and run the following command from the root of the repository, with a python environment where ``pulearn`` is installed:

.. code-block:: bash

    python examples/BreastCancerElkanotoExample.py

You should see a nice plot like the one below, comparing the F1 score of the PU learner versus a naive learner, demonstrating how PU learning becomes more effective - or worthwhile - the more positive examples are "hidden" from the training set.

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

``pulearn`` code is written to adhere to the coding style dictated by `flake8 <http://flake8.pycqa.org/en/latest/>`_. Practically, this means that one of the jobs that runs on `the project's Travis <https://travis-ci.org/pulearn/pulearn>`_ for each commit and pull request checks for a successful run of the ``flake8`` CLI command in the repository's root. Which means pull requests will be flagged red by the Travis bot if non-flake8-compliant code was added.

To solve this, please run ``flake8`` on your code (whether through your text editor/IDE or using the command line) and fix all resulting errors. Thank you! :)


Adding documentation
--------------------

This project is documented using the `numpy docstring conventions`_, which were chosen as perhaps the most widelspread conventions both supported by common tools such as Sphinx and resulting in human-readable docstrings (in my personal opinion, of course). When documenting code you add to this project, please follow `these conventions`_.

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
* Bayesian PU Classifiers (PNB, WNB, PTAN, WTAN) - ported from
  `Bayesian Classifiers for PU Learning <https://github.com/chengning-zhang/Bayesian-Classifers-for-PU_learning>`_
  by `Chengning Zhang <https://github.com/chengning-zhang>`_ (MIT License).

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

.. |Build-Status| image:: https://github.com/pulearn/pulearn/actions/workflows/ci-test.yml/badge.svg
  :target: https://github.com/pulearn/pulearn/actions/workflows/ci-test.yml

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
