"""
main.py — interactive CLI for the rare-event-simulation estimators.

pick an estimator, pick the problem parameters, run it, see the results.
"""

from scipy import stats
from src.rare_event_sim.estimators.crude_mc import CrudeMonteCarlo


def gaussian_sampler(n, rng):
    return rng.standard_normal(n)


def choose_estimator():
    """
    Prompt the user to choose an estimator.
    Returns:
        tuple[str, type]: The estimator name and corresponding class.
    """
    estimators = {
        "0": ("Model 0 — Crude Monte Carlo", CrudeMonteCarlo),
        # Model 1 - Variance Reduction, Model 2 - Importance Sampling, etc.
        # get added here as they're built.
    }
    print("\nAvailable estimators:")
    for key, (name, _) in estimators.items():
        print(f"  [{key}] {name}")
    choice = input("\nSelect an estimator: ").strip()
    while choice not in estimators:
        print(f"  Invalid choice '{choice}'. Please enter one of: {', '.join(estimators.keys())}")
        choice = input("  Select an estimator: ").strip()
    estimator_name, estimator_class = estimators[choice]
    return estimator_name, estimator_class


def choose_threshold():
    """
    Prompt the user for the tail threshold t, estimating P(X > t) for X ~ N(0,1).
    Returns:
        float: The threshold.
    """
    raw = input("\nEnter threshold t, for P(X > t) with X ~ N(0,1) (e.g. 3.0): ").strip()
    while True:
        try:
            return float(raw)
        except ValueError:
            raw = input(f"  Invalid threshold '{raw}'. Enter a number: ").strip()


def choose_n():
    """
    Prompt the user for the number of samples to draw.
    Returns:
        int: n, must be >= 2.
    """
    raw = input("\nEnter number of samples n (e.g. 100000): ").strip()
    while True:
        if raw.isdigit() and int(raw) >= 2:
            return int(raw)
        raw = input("  n must be a whole number >= 2. Enter n: ").strip()


def choose_target_re():
    """
    Prompt the user for an optional target relative error.
    Returns:
        float | None: target RE, or None if skipped.
    """
    raw = input("\nTarget relative error, e.g. 0.1 (press Enter to skip): ").strip()
    if raw == "":
        return None
    while True:
        try:
            val = float(raw)
            if 0 < val < 1:
                return val
        except ValueError:
            pass
        raw = input("  Must be a number between 0 and 1, or blank to skip: ").strip()


def choose_seed():
    """
    Prompt the user for an optional RNG seed.
    Returns:
        int | None: seed, or None for a random run.
    """
    raw = input("\nRNG seed (press Enter for random): ").strip()
    if raw == "":
        return None
    while not raw.isdigit():
        raw = input("  Must be a whole number, or blank for random: ").strip()
    return int(raw)


def run_crude_mc(threshold, n, seed, target_re, level=0.95):
    true_p = stats.norm.sf(threshold)

    est = CrudeMonteCarlo(
        sampler=gaussian_sampler,
        event_fn=lambda x: x > threshold,
        seed=seed,
    )
    est.run(n)
    ci_low, ci_high = est.confidence_interval(level=level)

    print(f"\nthreshold        : {threshold}")
    print(f"true P(X>t)      : {true_p:.6e}")
    print(f"n                : {n:,}")
    print(f"n_hits           : {est.n_hits:,}")
    print(f"p_hat            : {est.p_hat:.6e}")
    print(f"std_error        : {est.std_error:.6e}")
    print(f"{int(level*100)}% CI          : ({ci_low:.6e}, {ci_high:.6e})")

    if est.n_hits > 0:
        print(f"relative_error   : {est.relative_error(true_p):.4f}")
    else:
        print("relative_error   : undefined (zero hits)")

    if target_re is not None:
        n_needed = CrudeMonteCarlo.required_n(target_re=target_re, gamma=true_p)
        print(f"\nrequired n for {target_re:.0%} RE at this threshold: {n_needed:,.0f}")


def main():
    print("=== Rare Event Simulation Framework ===")
    estimator_name, estimator_class = choose_estimator()
    threshold = choose_threshold()
    n = choose_n()
    target_re = choose_target_re()
    seed = choose_seed()

    print(f"\nRunning {estimator_name}...")

    # Dispatch by class — right now there's only one, but this is the
    # seam where Model 1/2/etc. branch to their own run_* function once
    # they exist (their choose_* prompts may differ, e.g. no closed-form
    # true_p for a finance problem).
    if estimator_class is CrudeMonteCarlo:
        run_crude_mc(threshold, n, seed, target_re)


if __name__ == "__main__":
    main()