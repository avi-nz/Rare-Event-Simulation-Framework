"""
Tests for rareevent.monte_carlo.crude_mc.CrudeMonteCarlo

These are correctness checks with known right answers (per the book's
Sec 1.1 derivations), not just "does it run without crashing." Randomized
tests use a fixed seed so failures are reproducible, and use tolerances
wide enough to not be flaky, but tight enough to catch real bugs.
"""

import numpy as np
import pytest
from scipy import stats

from rare_event_sim.estimators.crude_mc import CrudeMonteCarlo


def gaussian_sampler(n, rng):
    return rng.standard_normal(n)


def make_estimator(threshold, seed=0):
    return CrudeMonteCarlo(
        sampler=gaussian_sampler,
        event_fn=lambda x: x > threshold,
        seed=seed,
    )


class TestPointEstimate:
    def test_symmetric_case_converges_to_half(self):
        """P(X > 0) for X ~ N(0,1) is exactly 0.5 — a good sanity anchor
        because it's a large probability, so a modest n should nail it
        tightly (no rare-event issues to muddy the check)."""
        est = make_estimator(threshold=0.0, seed=1).run(200_000)
        assert est.p_hat == pytest.approx(0.5, abs=0.01)

    def test_converges_to_known_tail_probability(self):
        """P(X > 3) for X ~ N(0,1) has a known closed-form value.
        Run at the estimator's own required_n for 10% RE and check we're
        actually within a reasonable multiple of that RE."""
        threshold = 3.0
        true_p = stats.norm.sf(threshold)
        n = int(CrudeMonteCarlo.required_n(target_re=0.10, gamma=true_p))

        est = make_estimator(threshold, seed=2).run(n)
        rel_err = est.relative_error(true_p)

        # A single run at the "designed for 10% RE" sample size won't hit
        # exactly 10% every time (it's a statistical guarantee, not a
        # hard cap) — check it's in a sane ballpark, not blown up.
        assert rel_err < 0.30

    def test_reproducible_with_same_seed(self):
        est1 = make_estimator(threshold=2.0, seed=42).run(50_000)
        est2 = make_estimator(threshold=2.0, seed=42).run(50_000)
        assert est1.p_hat == est2.p_hat
        assert est1.n_hits == est2.n_hits

    def test_different_seeds_give_different_runs(self):
        est1 = make_estimator(threshold=2.0, seed=1).run(50_000)
        est2 = make_estimator(threshold=2.0, seed=2).run(50_000)
        assert est1.p_hat != est2.p_hat


class TestVariance:
    def test_variance_matches_theoretical_bernoulli_variance(self):
        """Empirical variance of p_hat across many independent runs should
        match the theoretical gamma(1-gamma)/n — this is the check that
        would catch a wrong formula (e.g. forgetting the n-1 correction,
        or using n instead of n-1) even though a single run's .variance
        can't be checked against anything on its own."""
        threshold = 1.0
        true_p = stats.norm.sf(threshold)
        n = 5_000
        n_trials = 2_000

        rng = np.random.default_rng(7)
        p_hats = np.empty(n_trials)
        for t in range(n_trials):
            est = CrudeMonteCarlo(
                sampler=gaussian_sampler,
                event_fn=lambda x: x > threshold,
                seed=int(rng.integers(1, 10**9)),
            )
            est.run(n)
            p_hats[t] = est.p_hat

        empirical_var = p_hats.var(ddof=1)
        theoretical_var = true_p * (1 - true_p) / n

        # Ratio check rather than absolute, since these are small numbers.
        assert empirical_var == pytest.approx(theoretical_var, rel=0.15)

    def test_variance_decreases_with_n(self):
        est_small = make_estimator(threshold=1.0, seed=3).run(1_000)
        est_large = make_estimator(threshold=1.0, seed=3).run(100_000)
        assert est_large.variance < est_small.variance

    def test_variance_requires_run_first(self):
        est = make_estimator(threshold=1.0)
        with pytest.raises(RuntimeError):
            _ = est.variance


class TestConfidenceInterval:
    def test_ci_is_ordered_and_contains_point_estimate(self):
        est = make_estimator(threshold=1.0, seed=4).run(20_000)
        ci_low, ci_high = est.confidence_interval()
        assert ci_low <= est.p_hat <= ci_high

    def test_wider_level_gives_wider_interval(self):
        est = make_estimator(threshold=1.0, seed=5).run(20_000)
        low_95, high_95 = est.confidence_interval(level=0.95)
        low_99, high_99 = est.confidence_interval(level=0.99)
        assert (high_99 - low_99) > (high_95 - low_95)

    def test_coverage_near_nominal_when_n_hits_is_not_tiny(self):
        """The book flags that CLT-based CI coverage is unreliable when
        n*p_hat is small (this is exactly what Model 0's diagnostics
        plot showed). Here we deliberately pick a regime where n*gamma
        is comfortably large (~100 expected hits) and check coverage is
        close to nominal — this test would fail if run in the tiny-n
        regime, which is the point: it documents where the guarantee
        actually holds."""
        threshold = 2.0
        true_p = stats.norm.sf(threshold)
        n = int(100 / true_p)  # ~100 expected hits
        n_trials = 500
        level = 0.95

        rng = np.random.default_rng(9)
        contains = 0
        for _ in range(n_trials):
            est = CrudeMonteCarlo(
                sampler=gaussian_sampler,
                event_fn=lambda x: x > threshold,
                seed=int(rng.integers(1, 10**9)),
            )
            est.run(n)
            ci_low, ci_high = est.confidence_interval(level=level)
            if ci_low <= true_p <= ci_high:
                contains += 1

        coverage = contains / n_trials
        assert coverage == pytest.approx(level, abs=0.05)


class TestRequiredN:
    def test_required_n_scales_inversely_with_gamma(self):
        n_moderate = CrudeMonteCarlo.required_n(target_re=0.1, gamma=1e-3)
        n_rare = CrudeMonteCarlo.required_n(target_re=0.1, gamma=1e-6)
        # gamma shrinks by 1000x -> required n grows by ~1000x
        assert n_rare / n_moderate == pytest.approx(1000, rel=0.01)

    def test_required_n_scales_inversely_with_re_squared(self):
        n_loose = CrudeMonteCarlo.required_n(target_re=0.2, gamma=1e-4)
        n_tight = CrudeMonteCarlo.required_n(target_re=0.1, gamma=1e-4)
        # RE halved -> required n quadruples
        assert n_tight / n_loose == pytest.approx(4, rel=0.01)

    def test_matches_book_example(self):
        """Book Sec 1.1 worked example: gamma=1e-9, target RE=0.1
        requires n >= 3.84e11."""
        n = CrudeMonteCarlo.required_n(target_re=0.1, gamma=1e-9)
        assert n == pytest.approx(3.84e11, rel=0.01)


class TestInputValidation:
    def test_raises_on_n_too_small(self):
        est = make_estimator(threshold=1.0)
        with pytest.raises(ValueError):
            est.run(1)

    def test_raises_when_accessing_results_before_run(self):
        est = make_estimator(threshold=1.0)
        with pytest.raises(RuntimeError):
            _ = est.p_hat

    def test_relative_error_raises_on_zero_true_p(self):
        est = make_estimator(threshold=1.0, seed=1).run(1000)
        with pytest.raises(ValueError):
            est.relative_error(0.0)