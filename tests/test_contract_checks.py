import numpy as np
import pytest

from pulearn.base import BasePUClassifier
from tests.contract_checks import assert_base_pu_estimator_contract


class _ContractEstimator(BasePUClassifier):
    def __init__(self, proba):
        self._proba = proba

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self._proba


def test_contract_helper_rejects_non_numeric_predict_proba():
    estimator = _ContractEstimator([["bad", 1.0], [0.2, 0.8]])
    with pytest.raises(AssertionError, match="converted to float"):
        assert_base_pu_estimator_contract(
            estimator,
            np.zeros((2, 1)),
            np.array([1, 0]),
        )


def test_contract_helper_enforces_row_sum_via_base_validator():
    estimator = _ContractEstimator([[0.2, 0.2], [0.1, 0.9]])
    with pytest.raises(AssertionError, match="rows must sum to 1"):
        assert_base_pu_estimator_contract(
            estimator,
            np.zeros((2, 1)),
            np.array([1, 0]),
        )
