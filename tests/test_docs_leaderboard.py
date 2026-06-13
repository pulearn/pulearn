"""Tests for the static benchmark leaderboard documentation page."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_leaderboard_page_fetches_public_latest_json():
    html = (ROOT / "doc" / "leaderboard.html").read_text(encoding="utf-8")

    assert "PUlearn Benchmark Leaderboard" in html
    assert 'const DATA_URL = "/pulearn/leaderboard/latest.json";' in html
    assert 'href="/pulearn/leaderboard/latest.csv"' in html
    assert "pu_average_precision" in html
    assert "oracle_roc_auc" in html


def test_docs_build_copies_leaderboard_page():
    build_script = (ROOT / "doc" / "build.sh").read_text(encoding="utf-8")

    assert 'mkdir -p "$BUILDROOT/pulearn/leaderboard"' in build_script
    assert (
        'cp "$DOCROOT/leaderboard.html" '
        '"$BUILDROOT/pulearn/leaderboard/index.html"'
    ) in build_script


def test_nightly_workflow_publishes_leaderboard_data_after_matrix_success():
    workflow = (
        ROOT / ".github" / "workflows" / "benchmark-nightly.yml"
    ).read_text(encoding="utf-8")

    assert "publish-leaderboard:" in workflow
    assert "needs: benchmark-nightly" in workflow
    assert "benchmark-results-py3.12" in workflow
    assert "destination_dir: leaderboard" in workflow
    assert "publish_dir: public-leaderboard/leaderboard" in workflow
