"""
Serve Like DeepSeek: A Disaggregated Inference Simulator

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_workload
import math
import numpy as np

CLASS_SPEC = {
    "chat": (2000, 0.6, 16000, 300, 0.3),
    "agent": (40000, 0.4, 64000, 150, 0.8),
    "doc": (150000, 0.5, 400000, 500, 0.1),
}


def make_workload(n, seed, rate=4.0, mix=(0.7, 0.2, 0.1)):
    # Create a reproducible random number generator.
    rng = np.random.default_rng(seed)

    # 1. Generate inter-arrival gaps and calculate cumulative arrival times.
    gaps = rng.exponential(1 / rate, size=n)
    t_arrivals = np.cumsum(gaps)

    # 2. Assign a request class to each request.
    classes = rng.choice(
        ["chat", "agent", "doc"],
        size=n,
        p=list(mix)
    )

    # 3. Generate context length, cached prefix length, and output length
    #    for each request, in that order.
    workload = []

    for i in range(n):
        cls = str(classes[i])

        # Retrieve the constants for the selected request class.
        median, sigma, cap, out_mean, cached_frac = CLASS_SPEC[cls]

        # Generate context length using a log-normal distribution.
        ctx = int(
            min(
                cap,
                max(128, rng.lognormal(math.log(median), sigma))
            )
        )

        # Generate the cached prefix length, rounded down to a
        # multiple of 128 tokens.
        cached = (
            int(ctx * cached_frac * rng.uniform(0.5, 1.0))
            // 128
            * 128
        )

        # Generate output length using a Poisson distribution.
        out = int(max(1, rng.poisson(out_mean)))

        # Store the request using the required dictionary structure.
        workload.append({
            "id": i,
            "t_arrive": float(t_arrivals[i]),
            "cls": cls,
            "ctx": ctx,
            "cached": cached,
            "out": out,
        })

    return workload

# Step 2 - fit_cost_model
def prefill_cost(q, c, coeffs):
    # Unpack the fitted coefficients: quadratic, cross-term, and linear.
    a, b, d = coeffs

    # Estimate prefill execution time in nanoseconds.
    return float(a * q**2 + b * q * c + d * q)


def fit_cost_model(samples):
    # Build the design matrix with columns [q^2, q*c, q].
    X = np.array(
        [[q**2, q * c, q] for q, c, t in samples],
        dtype=float
    )

    # Observed prefill times, in nanoseconds.
    y = np.array(
        [t for q, c, t in samples],
        dtype=float
    )

    # Solve the least-squares problem X @ [a, b, d] ~= y.
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Return Python floats rounded to six decimal places.
    return tuple(float(round(value, 6)) for value in coeffs)

