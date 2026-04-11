"""Tests for the decision-oriented documentation guides in doc/.

These checks verify that the four required guide files exist and contain
the expected section headings.  They do not test prose quality, but they
ensure no guide is accidentally deleted and that the top-level structure
remains coherent.
"""

import pathlib
import re

_DOC_DIR = pathlib.Path(__file__).parent.parent / "doc"

_GUIDE_FILES = {
    "guide_pu_fundamentals.md",
    "guide_learner_selection.md",
    "guide_evaluation.md",
    "guide_failure_modes.md",
}

# Required H2 headings (## ...) that must appear in each guide.
_REQUIRED_HEADINGS = {
    "guide_pu_fundamentals.md": [
        "What Is PU Learning?",
        "Core Assumptions",
        "Key Quantities",
        "Label Conventions",
    ],
    "guide_learner_selection.md": [
        "Quick Decision Matrix",
        "Available Classifiers",
        "Feature Comparison Table",
    ],
    "guide_evaluation.md": [
        "Why Standard Metrics Are Wrong",
        "What You Need",
        "Available Corrected Metrics",
        "Sensitivity Analysis",
    ],
    "guide_failure_modes.md": [
        "Prior / Propensity Warnings",
        "Classifier Warnings",
        "Metric Warnings",
        "Calibration Warnings",
        "Prediction Failure Modes",
        "Debug Checklist",
    ],
}

# Internal cross-links that must appear in each guide.
_REQUIRED_CROSSLINKS = {
    "guide_pu_fundamentals.md": [
        "guide_learner_selection.md",
        "guide_evaluation.md",
        "guide_failure_modes.md",
    ],
    "guide_evaluation.md": [
        "guide_pu_fundamentals.md",
        "guide_failure_modes.md",
    ],
    "guide_failure_modes.md": [
        "guide_pu_fundamentals.md",
        "guide_learner_selection.md",
        "guide_evaluation.md",
    ],
    "guide_learner_selection.md": [
        "guide_failure_modes.md",
    ],
}


def test_guide_files_exist():
    """All four required guide files must be present in doc/."""
    for name in _GUIDE_FILES:
        path = _DOC_DIR / name
        assert path.exists(), f"Missing guide file: {path}"
        assert path.stat().st_size > 0, f"Guide file is empty: {path}"


def test_guide_files_are_markdown():
    """Guide files must have a top-level H1 heading."""
    for name in _GUIDE_FILES:
        text = (_DOC_DIR / name).read_text(encoding="utf-8")
        assert re.search(r"^#\s+\S", text, re.MULTILINE), (
            f"{name}: no top-level H1 heading found"
        )


def test_required_headings_present():
    """Each guide must contain its required section headings as H2 (##)."""
    for name, headings in _REQUIRED_HEADINGS.items():
        text = (_DOC_DIR / name).read_text(encoding="utf-8")
        for heading in headings:
            pattern = re.compile(
                r"^##\s+" + re.escape(heading) + r"\s*$",
                re.MULTILINE,
            )
            assert pattern.search(text), (
                f"{name}: required H2 heading '## {heading}' not found"
            )


def test_cross_links_present():
    """Each guide must reference the other related guides."""
    for name, links in _REQUIRED_CROSSLINKS.items():
        text = (_DOC_DIR / name).read_text(encoding="utf-8")
        for link_target in links:
            assert link_target in text, (
                f"{name}: expected cross-link to '{link_target}' not found"
            )


def test_documentation_md_references_guides():
    """src/pulearn/documentation.md must reference all four guide modules."""
    doc_md = (
        pathlib.Path(__file__).parent.parent
        / "src"
        / "pulearn"
        / "documentation.md"
    )
    assert doc_md.exists(), f"Missing file: {doc_md}"
    text = doc_md.read_text(encoding="utf-8")
    # Links now point to the pdoc3-generated docsite pages.
    guide_links = [
        "guides/pu_fundamentals.html",
        "guides/learner_selection.html",
        "guides/evaluation.html",
        "guides/failure_modes.html",
    ]
    for link in guide_links:
        assert link in text, (
            f"documentation.md does not reference docsite guide link '{link}'"
        )


def test_readme_rst_references_guides():
    """README.rst must reference all four guide files."""
    readme = pathlib.Path(__file__).parent.parent / "README.rst"
    assert readme.exists(), f"Missing file: {readme}"
    text = readme.read_text(encoding="utf-8")
    for name in _GUIDE_FILES:
        assert name in text, (
            f"README.rst does not reference guide file '{name}'"
        )
