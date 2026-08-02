"""
Crude (naive / standard) Monte Carlo estimator.

Reference: Rubino & Tuffin (eds.), "Rare Event Simulation using Monte Carlo
Methods", Chapter 1, Section 1.1.

The method: to estimate gamma = P(A) for some event A, draw n independent
copies of the system, record the Bernoulli indicator X_i = 1(A occurs in
draw i), and estimate gamma by the sample proportion

    gamma_hat = (X_1 + ... + X_n) / n

This is deliberately generic — it takes a `sampler` (draws raw system
outcomes) and an `event_fn` (decides which outcomes count as the rare
event) rather than being hardcoded to the Gaussian tail example, so the
same class works for the finance (Model 5) and reliability applications
later without rewriting the estimator itself.
"""

import numpy as np
from scipy import stats


class CrudeMonteCarlo:
    """
    Crude Monte Carlo estimator for a rare-event probability.

    Parameters
    ----------
    sampler : callable(n, rng) -> array-like of length n
        Draws n independent realizations of the underlying system.
        e.g. lambda n, rng: rng.standard_normal(n)
    event_fn : callable(samples) -> boolean array-like of length n
        Returns True where the rare event occurred for each sample.
        e.g. lambda x: x > 3.0
    seed : int, optional
        Seed for the internal RNG. Two estimators with the same seed
        and same n produce identical results (useful for tests and for
        fair comparisons against later estimators in the benchmark).
    """

    def __init__(self, sampler, event_fn, seed=None):
        self.sampler = sampler
        self.event_fn = event_fn
        self.rng = np.random.default_rng(seed)
        self._n = None
        self._n_hits = None

    def run(self, n):
        """Draw n samples and record the estimator. Returns self (chainable)."""
        if n < 2:
            raise ValueError("n must be >= 2 (variance estimate needs n-1 in the denominator)")

        samples = self.sampler(n, self.rng)
        hits = np.asarray(self.event_fn(samples), dtype=bool)

        self._n = n
        self._n_hits = int(np.sum(hits))
        return self

    def _check_has_run(self):
        if self._n is None:
            raise RuntimeError("Call .run(n) before accessing results.")

    @property
    def n(self):
        self._check_has_run()
        return self._n

    @property
    def n_hits(self):
        self._check_has_run()
        return self._n_hits

    @property
    def p_hat(self):
        """Point estimate gamma_hat = n_hits / n."""
        self._check_has_run()
        return self._n_hits / self._n

    @property
    def variance(self):
        """
        Unbiased estimator of Var(gamma_hat), per book eq. in Sec 1.1:
        sigma_hat^2 = n * p_hat * (1 - p_hat) / (n - 1), then
        Var(gamma_hat) = sigma_hat^2 / n.

        Note: uses (n-1) in the denominator for the unbiased variance of
        the underlying Bernoulli, not the biased p_hat*(1-p_hat)/n used
        in the original prototype script.
        """
        self._check_has_run()
        p = self.p_hat
        sigma_sq_hat = self._n * p * (1 - p) / (self._n - 1)
        return sigma_sq_hat / self._n

    @property
    def std_error(self):
        return np.sqrt(self.variance)

    def confidence_interval(self, level=0.95):
        """
        CLT-based (Wald) confidence interval at the given level, per book
        Sec 1.1: (p_hat +/- z * std_error), z = Phi^-1((1+level)/2).

        Known to undercover badly when n * p_hat is small (few or zero
        hits) — see Model 0's diagnostics. Reported as-is; a Wilson or
        Clopper-Pearson interval (book footnote, Sec 1.1) would be a
        Model 0 follow-up, not fixed here.
        """
        self._check_has_run()
        z = stats.norm.ppf((1 + level) / 2)
        se = self.std_error
        p = self.p_hat
        return p - z * se, p + z * se

    def relative_error(self, true_p):
        """abs(p_hat - true_p) / true_p, requires knowing the ground truth."""
        self._check_has_run()
        if true_p == 0:
            raise ValueError("relative_error is undefined when true_p == 0")
        return abs(self.p_hat - true_p) / true_p

    @staticmethod
    def required_n(target_re, gamma, level=0.95):
        """
        Minimum n to achieve relative error <= target_re at the given
        confidence level, per book Sec 1.1:

            n >= z^2 / (target_re^2 * gamma)

        This is the headline number for comparing crude MC against every
        later method: "how many samples does each technique need to hit
        a 10% RE on this gamma?"
        """
        z = stats.norm.ppf((1 + level) / 2)
        return z**2 / (target_re**2 * gamma)


def main():
    # Quick sanity check against the known Gaussian tail case.
    def gaussian_sampler(n, rng):
        return rng.standard_normal(n)

    for threshold in (3.0, 4.5):
        true_p = stats.norm.sf(threshold)
        est = CrudeMonteCarlo(
            sampler=gaussian_sampler,
            event_fn=lambda x, t=threshold: x > t,
            seed=42,
        )
        n_needed = CrudeMonteCarlo.required_n(target_re=0.10, gamma=true_p)
        n_run = min(int(n_needed), 2_000_000)  # cap for a quick sanity run
        est.run(n_run)
        ci_low, ci_high = est.confidence_interval()

        print(f"threshold={threshold}  true_p={true_p:.4e}")
        print(f"  required n for 10% RE: {n_needed:,.0f}  (ran {n_run:,d})")
        print(f"  p_hat={est.p_hat:.4e}  95% CI=({ci_low:.4e}, {ci_high:.4e})")
        if true_p > 0 and est.n_hits > 0:
            print(f"  relative_error={est.relative_error(true_p):.3f}")
        print()


if __name__ == "__main__":
    main()