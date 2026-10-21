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

# Step 3 - tier_demand
def context_tier(ctx, tiers=(16000, 64000, 256000)):
    # Return the number of tier boundaries that ctx meets or exceeds.
    # For the default tiers, the resulting indices are:
    # 0: below 16K, 1: 16K–64K, 2: 64K–256K, 3: 256K and above.
    return sum(1 for boundary in tiers if ctx >= boundary)


def tier_demand(requests, coeffs, tiers=(16000, 64000, 256000)):
    # There is one more tier than the number of boundaries.
    num_tiers = len(tiers) + 1

    # Initialize per-request tier assignments and per-tier statistics.
    tier_per_request = []
    counts = [0] * num_tiers
    demand_totals = [0.0] * num_tiers

    # Assign each request to a tier and accumulate its prefill demand.
    for request in requests:
        ctx = request["ctx"]
        cached = request["cached"]

        # Classify the request using its full context length.
        tier = context_tier(ctx, tiers)
        tier_per_request.append(tier)
        counts[tier] += 1

        # Only uncached tokens require new prefill computation.
        q = ctx - cached

        # Convert the estimated prefill cost from nanoseconds to seconds.
        demand_seconds = prefill_cost(q, cached, coeffs) / 1e9

        # Accumulate demand for this context-length tier.
        demand_totals[tier] += demand_seconds

    # Calculate total prefill demand across all tiers.
    total_demand = sum(demand_totals)

    # Calculate each tier's share of total demand.
    # If the total demand is zero, all shares must be 0.0.
    if total_demand == 0:
        shares = [0.0] * num_tiers
    else:
        shares = [
            round(demand / total_demand, 4)
            for demand in demand_totals
        ]

    # Return the required dictionary with the specified rounding.
    return {
        "tier": tier_per_request,
        "counts": counts,
        "demand_s": [round(demand, 3) for demand in demand_totals],
        "shares": shares,
    }

# Step 4 - decode_step_time
def ep_params(ep, expert_bytes_total, hbm_bw, fixed_ms=0.6, alpha=0.05, beta=1e-5):
    # Calculate the expert-weight read time in milliseconds.
    # expert_bytes_total / ep / hbm_bw gives the time in seconds.
    expert_read_ms = (expert_bytes_total / ep / hbm_bw) * 1000.0

    # Add the fixed time for dense-weight processing.
    t0 = expert_read_ms + fixed_ms

    # Return the decode-step parameters, including the default
    # MoE load-balancing multiplier of 1.0.
    return {
        "t0": t0,
        "alpha": alpha,
        "beta": beta,
        "moe_lb": 1.0,
    }


def decode_step_time(B, K, params):
    # Calculate the decode step time in milliseconds:
    # base expert-weight cost, batch-size cost, and KV-token cost.
    return (
        params["t0"] * params.get("moe_lb", 1.0)
        + params["alpha"] * B
        + params["beta"] * K
    )

# Step 5 - chunk_for
def chunk_for(remaining_ns, c, coeffs, cap=32768):
    # No prefill work can be scheduled when the time budget is non-positive.
    if remaining_ns <= 0:
        return 0

    # Unpack the quadratic cost-model coefficients.
    a, b, d = coeffs
    lin = b * c + d
    R = remaining_ns

    # Solve a*q^2 + lin*q - R = 0 for the positive boundary.
    if a > 0:
        discriminant = lin**2 + 4 * a * R
        root = (-lin + math.sqrt(discriminant)) / (2 * a)
    else:
        # When the quadratic coefficient is zero or negative,
        # use the linear solution when the linear coefficient is positive.
        if lin <= 0:
            return cap
        root = R / lin

    # Floor the boundary, adding a tiny epsilon to absorb floating-point
    # rounding errors near integer values.
    q = max(0, math.floor(root + 1e-9))

    # Correct any rounding error that would make the chunk exceed the budget.
    while q > 0 and prefill_cost(q, c, coeffs) > R:
        q -= 1

    # Enforce the maximum permitted chunk size.
    return min(q, cap)

# Step 6 - plan_prefill_step
def plan_prefill_step(queue, budget_ns, coeffs, cap=32768):
    # Track the remaining time budget without modifying the input queue.
    remaining_ns = budget_ns
    chunks = []

    # Process requests in FIFO order.
    for index, item in enumerate(queue):
        # Stop if the remaining time budget is exhausted.
        if remaining_ns <= 0:
            break

        # Determine the largest chunk that fits the remaining budget,
        # while ensuring we do not process more than the request needs.
        q = min(
            chunk_for(remaining_ns, item["prefix"], coeffs, cap),
            item["left"],
        )

        # Stop at the first request that cannot receive a positive chunk.
        if q <= 0:
            break

        # Record the request index and the number of tokens to prefill.
        chunks.append((index, q))

        # Calculate the time consumed by this chunk and update the budget.
        remaining_ns -= prefill_cost(q, item["prefix"], coeffs)

    # Report the total time used and the budget remaining, rounded to
    # integer nanoseconds as required.
    return {
        "chunks": chunks,
        "used_ns": int(round(budget_ns - remaining_ns)),
        "left_ns": int(round(remaining_ns)),
    }

# Step 7 - PrefillEngine
import collections


class PrefillEngine:
    def __init__(self, coeffs, budget_ns, cap=32768):
        # Store the prefill cost model and scheduling constraints.
        self.coeffs = coeffs
        self.budget_ns = budget_ns
        self.cap = cap

        # FIFO queue containing requests awaiting prefill completion.
        self.queue = collections.deque()

        # Cumulative simulated GPU busy time, measured in milliseconds.
        self.busy_ms = 0.0

    def submit(self, req):
        # Store the uncached tokens still requiring prefill, the cached
        # prefix length, and a reference to the original request.
        item = {
            "id": req["id"],
            "left": req["ctx"] - req["cached"],
            "prefix": req["cached"],
            "t_arrive": req["t_arrive"],
            "req": req,
        }

        self.queue.append(item)

    def pending_tokens(self):
        # Return the total number of tokens still requiring prefill.
        return sum(item["left"] for item in self.queue)

    def step(self, now_ms):
        # Take a snapshot so chunk indices refer to a stable queue order.
        items = list(self.queue)

        # Plan one FIFO scheduling step within the configured time budget.
        plan = plan_prefill_step(
            items,
            self.budget_ns,
            self.coeffs,
            self.cap,
        )

        chunks = plan["chunks"]

        # Nothing can execute if no positive-sized chunks were scheduled.
        if not chunks:
            return []

        # Convert the consumed time from nanoseconds to milliseconds.
        step_ms = plan["used_ns"] / 1e6

        # Accumulate simulated GPU utilization time.
        self.busy_ms += step_ms

        # Apply the scheduled prefill chunks to their corresponding requests.
        for index, q in chunks:
            item = items[index]
            item["left"] -= q
            item["prefix"] += q

        # Remove completed requests while preserving the FIFO order of
        # unfinished requests and the completion order of finished requests.
        completed = []

        for _ in range(len(self.queue)):
            item = self.queue.popleft()

            if item["left"] <= 0:
                req = item["req"]

                # Record completion time and time to first token in milliseconds.
                req["t_prefill_done"] = now_ms + step_ms
                req["ttft_ms"] = (
                    now_ms + step_ms - item["t_arrive"] * 1000
                )

                completed.append(req)
            else:
                # Unfinished requests remain in their original relative order.
                self.queue.append(item)

        return completed

