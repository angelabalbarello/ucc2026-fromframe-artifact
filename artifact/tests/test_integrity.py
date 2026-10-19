"""Integrity of the frozen artifact (CRLF-normalised manifests)."""
import hashlib
import json
from pathlib import Path
import pytest

ART = Path(__file__).resolve().parents[1]
OUT = ART / "experiments" / "e10_eaml" / "outputs"


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha_lf(p):
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _manifest(p):
    rows = []
    for line in p.read_text(encoding="utf-8").replace("\r", "").splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            rows.append((h, name.strip().lstrip("*")))
    return rows


def test_root_manifest_all_files_match():
    bad = [n for h, n in _manifest(ART / "MANIFEST.sha256")
           if h not in (_sha(ART / n), _sha_lf(ART / n))]
    assert bad == []


def test_freeze_manifest_core_protocol_files_match():
    m = dict((n, h) for h, n in _manifest(OUT / "MANIFEST.sha256"))
    base = ART / "experiments" / "e10_eaml"
    assert m["spec_e10.json"] in (_sha(OUT / "spec_e10.json"), _sha_lf(OUT / "spec_e10.json"))
    for f in ("arms.py", "metrics.py", "scenarios.py"):
        assert m[f] in (_sha(base / f), _sha_lf(base / f)), f


def test_run_dev_divergence_from_freeze_manifest_is_recorded():
    """run_dev.py does not match the freeze-time manifest under either line
    ending (documented in docs/camera_ready/00; cause not presumed). This test
    pins the current state so any further change is detected."""
    m = dict((n, h) for h, n in _manifest(OUT / "MANIFEST.sha256"))
    p = ART / "experiments" / "e10_eaml" / "run_dev.py"
    assert m["run_dev.py"] not in (_sha(p), _sha_lf(p))
    assert _sha_lf(p).startswith("4a0dd71488f7e461")


def test_frozen_spec_values():
    s = json.loads((OUT / "spec_e10.json").read_text(encoding="utf-8"))
    assert s["specification_version"] == 3 and s["sar_window"] == 12
    assert s["a2_config"]["window_w"] == 8 and s["a0_config"]["cooldown_frames"] == 24
    assert s["holdout_seeds"] == list(range(3001, 3021))
