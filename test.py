import AI_stats_lab as amt
import numpy as np


# simple runner helper
def run_test(name, fn):
    try:
        fn()
        print(f"✅ {name} PASSED")
    except AssertionError as e:
        print(f"❌ {name} FAILED")
    except Exception as e:
        print(f"💥 {name} ERROR -> {e}")


# =========================
# Probability Tests
# =========================

def test_probability_union():
    assert abs(amt.probability_union(0.5, 0.4, 0.2) - 0.7) < 1e-6


def test_conditional_probability():
    assert abs(amt.conditional_probability(0.2, 0.4) - 0.5) < 1e-6


def test_are_independent():
    assert amt.are_independent(0.5, 0.4, 0.2)


def test_bayes_rule():
    assert abs(amt.bayes_rule(0.9, 0.01, 0.05) - 0.18) < 1e-6


# =========================
# Bernoulli Tests
# =========================

def test_bernoulli_pmf():
    assert abs(amt.bernoulli_pmf(1, 0.7) - 0.7) < 1e-6
    assert abs(amt.bernoulli_pmf(0, 0.7) - 0.3) < 1e-6


def test_bernoulli_symmetry():
    result = amt.bernoulli_theta_analysis([0.5])
    assert result[0][3] is True


# =========================
# Uniform Distribution Tests
# =========================

def test_uniform_statistics():
    result = amt.uniform_histogram_analysis([0], [1])
    a, b, sm, tm, me, sv, tv, ve = result[0]

    assert abs(tm - 0.5) < 1e-6
    assert abs(tv - (1/12)) < 1e-6
    assert me < 0.05
    assert ve < 0.05


# =========================
# Normal Distribution Tests
# =========================

def test_normal_statistics():
    result = amt.normal_histogram_analysis([0], [1])
    mu, sigma, sm, tm, me, sv, tv, ve = result[0]

    assert abs(tm - 0) < 1e-6
    assert abs(tv - 1) < 1e-6
    assert me < 0.05
    assert ve < 0.1


# =========================
# Run everything manually
# =========================

if __name__ == "__main__":
    tests = [
        test_probability_union,
        test_conditional_probability,
        test_are_independent,
        test_bayes_rule,
        test_bernoulli_pmf,
        test_bernoulli_symmetry,
        test_uniform_statistics,
        test_normal_statistics,
    ]

    print("\nRunning tests...\n")

    passed = 0
    for t in tests:
        run_test(t.__name__, t)
        try:
            t()
            passed += 1
        except:
            pass

    print(f"\nSummary: {passed}/{len(tests)} passed\n")
