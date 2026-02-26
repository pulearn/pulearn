import importlib
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pulearn._version as version_module


def test_version_is_set():
    assert isinstance(version_module.__version__, str)
    assert version_module.__version__ != "unknown"


def test_version_fallback_on_package_not_found():
    with patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError("pulearn"),
    ):
        importlib.reload(version_module)
        assert version_module.__version__ == "unknown"
    # Restore to the real version after the test
    importlib.reload(version_module)
