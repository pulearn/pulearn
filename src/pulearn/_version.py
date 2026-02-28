# This file is part of pulearn.
# https://github.com/pulearn/pulearn

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pulearn")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
