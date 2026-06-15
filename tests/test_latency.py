"""
tests/test_latency.py — Real Estate Fraud Detection
p95 inference latency assertion — standalone file.

Run:
  pytest tests/test_latency.py -v -s
"""

import os
import sys
import time
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


@pytest.fixture(scope="module")
def loaded_bundle():
    from src.inference import load_bundle
    return load_bundle()


def test_p95_latency_under_500ms(loaded_bundle):
    """
    Plan requirement: p95 inference < 500ms.
    Tests 20 predictions after warmup — p95 must be under limit.
    """
    from src.inference import predict_fraud

    listing = {
        "price": 350000.0, "bed": 3.0, "bath": 2.0,
        "house_size": 1800.0, "city": "Austin", "state": "TX",
        "zip_code": "78701", "status": "for_sale",
    }

    # Warmup call
    predict_fraud(listing, loaded_bundle)

    # Measure
    latencies = []
    N = 20
    for _ in range(N):
        t0 = time.perf_counter()
        predict_fraud(listing, loaded_bundle)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = latencies[N // 2]
    p95 = latencies[int(N * 0.95)]
    p99 = latencies[int(N * 0.99)] if N >= 100 else latencies[-1]

    print(f"\n{'─'*40}")
    print(f"  Latency Report ({N} calls, post-warmup)")
    print(f"  p50 : {p50:.1f}ms")
    print(f"  p95 : {p95:.1f}ms  ← must be < 500ms")
    print(f"  max : {p99:.1f}ms")
    print(f"{'─'*40}")

    assert p95 < 500, (
        f"p95 latency {p95:.1f}ms EXCEEDS 500ms limit!\n"
        f"Consider: reducing SHAP sample size or caching explainer."
    )
    print(f"✅ p95 = {p95:.1f}ms < 500ms — PASSED")
