"""Offline tests for bayesian_tracker.py -- no network/data dependency."""
from bayesian_tracker import BetaBinomialPosterior


def test_default_prior_is_uninformative():
    print("--- test_default_prior_is_uninformative ---")
    p = BetaBinomialPosterior()
    assert p.alpha == 1.0 and p.beta == 1.0
    assert abs(p.mean - 0.5) < 1e-9
    print("  OK: default Beta(1,1), mean=0.5")


def test_mean_converges_to_true_rate():
    print("--- test_mean_converges_to_true_rate ---")
    p = BetaBinomialPosterior()
    wins = 70
    losses = 30
    for _ in range(wins):
        p.update(win=True)
    for _ in range(losses):
        p.update(win=False)
    expected = (1 + wins) / (1 + wins + 1 + losses)
    assert abs(p.mean - expected) < 1e-9
    assert p.mean > 0.6  # pulled well toward the 70% true rate from the 50% prior
    print(f"  OK: mean={p.mean:.3f} after {wins}W/{losses}L, pulled toward true rate")


def test_credible_interval_narrows_with_more_data():
    print("--- test_credible_interval_narrows_with_more_data ---")
    p_small = BetaBinomialPosterior()
    for i in range(10):
        p_small.update(win=(i % 2 == 0))  # 5W/5L

    p_large = BetaBinomialPosterior()
    for i in range(200):
        p_large.update(win=(i % 2 == 0))  # 100W/100L

    lo_small, hi_small = p_small.credible_interval(0.80)
    lo_large, hi_large = p_large.credible_interval(0.80)
    width_small = hi_small - lo_small
    width_large = hi_large - lo_large
    assert width_large < width_small, f"expected narrower interval with more data: {width_large} vs {width_small}"
    print(f"  OK: 80% CI width shrinks from {width_small:.3f} (n=10) to {width_large:.3f} (n=200)")


def test_credible_lower_median_near_mean_for_symmetric_posterior():
    print("--- test_credible_lower_median_near_mean_for_symmetric_posterior ---")
    p = BetaBinomialPosterior()
    for i in range(40):
        p.update(win=(i % 2 == 0))  # symmetric 20W/20L -> Beta(21,21)
    median = p.credible_lower(0.5)
    assert abs(median - p.mean) < 0.02, f"median {median} should be close to mean {p.mean} for a symmetric posterior"
    print(f"  OK: median={median:.3f} close to mean={p.mean:.3f}")


def test_reset_restores_exact_prior():
    print("--- test_reset_restores_exact_prior ---")
    p = BetaBinomialPosterior(alpha=3.0, beta=2.0)
    for _ in range(15):
        p.update(win=True)
    assert p.n_updates == 15
    p.reset()
    assert p.alpha == 3.0 and p.beta == 2.0 and p.n_updates == 0
    print("  OK: reset() restores exact prior state")


if __name__ == "__main__":
    test_default_prior_is_uninformative()
    test_mean_converges_to_true_rate()
    test_credible_interval_narrows_with_more_data()
    test_credible_lower_median_near_mean_for_symmetric_posterior()
    test_reset_restores_exact_prior()
    print("\nALL BAYESIAN TRACKER TESTS PASSED")
