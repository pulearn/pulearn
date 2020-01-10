"""
The `pulearn` Python package provide a collection of scikit-learn wrappers to
several positive-unlabled learning (PU-learning) methods.

.. include:: ./documentation.md
"""

from .elkanoto import (  # noqa: F401
    ElkanotoPuClassifier,
    WeightedElkanotoPuClassifier,
)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
