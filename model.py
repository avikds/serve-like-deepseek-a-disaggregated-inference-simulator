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

# Step 8 - run_prefill_pool
def allocate_engines(shares, n_engines):
    # Convert each tier's demand share into its raw engine allocation.
    raw = [float(share) * n_engines for share in shares]

    # Initially assign the floor of each raw allocation.
    counts = [int(math.floor(value)) for value in raw]

    # Give every tier with positive demand at least one engine.
    for i, share in enumerate(shares):
        if share > 0:
            counts[i] = max(1, counts[i])

    # If the minimum-one-engine rule over-allocates the pool, remove
    # engines from the tier with the largest count-minus-raw difference.
    # Ties are resolved in favor of the lower tier index.
    while sum(counts) > n_engines:
        candidates = [i for i, count in enumerate(counts) if count > 0]

        if not candidates:
            break

        i = max(
            candidates,
            key=lambda j: (counts[j] - raw[j], -j)
        )
        counts[i] -= 1

    # If engines remain unallocated, give each next engine to the tier
    # with the largest raw-minus-count difference.
    # Ties are resolved in favor of the lower tier index.
    while sum(counts) < n_engines:
        if not counts:
            break

        i = max(
            range(len(counts)),
            key=lambda j: (raw[j] - counts[j], -j)
        )
        counts[i] += 1

    return counts


def run_prefill_pool(
    requests,
    n_engines,
    coeffs,
    budget_ns,
    tiered=True,
    tiers=(16000, 64000, 256000),
):
    # Keep the original request dictionaries but process them in
    # deterministic arrival-time and request-ID order.
    request_list = list(requests)
    ordered = sorted(
        request_list,
        key=lambda req: (req["t_arrive"], req["id"])
    )

    # Handle an empty workload without running the simulation.
    if not ordered:
        return {
            "by_tier": {},
            "makespan_ms": 0.0,
            "utilization": 0.0,
            "ttft_ms": {},
        }

    # A nonempty workload requires at least one engine and a positive
    # budget so that simulation time can advance.
    if n_engines <= 0:
        raise ValueError("n_engines must be positive")

    if budget_ns <= 0:
        raise ValueError("budget_ns must be positive")

    # Create the prefill engines.
    engines = [
        PrefillEngine(coeffs, budget_ns)
        for _ in range(n_engines)
    ]
    all_engine_ids = list(range(n_engines))

    # There is one context tier for each interval defined by the
    # boundaries, plus one tier for contexts above the last boundary.
    num_tiers = len(tiers) + 1

    if tiered:
        # Estimate the compute-demand shares for the context tiers.
        shares = tier_demand(
            request_list,
            coeffs,
            tiers
        )["shares"]

        # Allocate engines among tiers based on their demand shares.
        engine_counts = allocate_engines(shares, n_engines)

        # Give each tier a consecutive block of engines.
        # For example, allocations [1, 2, 1] give:
        # tier 0 -> [0], tier 1 -> [1, 2], tier 2 -> [3].
        groups = []
        start = 0

        for count in engine_counts:
            groups.append(list(range(start, start + count)))
            start += count

    else:
        # Without tiering, every context tier can use every engine.
        groups = [
            all_engine_ids[:]
            for _ in range(num_tiers)
        ]

    # Simulation time is maintained in milliseconds.
    tick_ms = budget_ns / 1e6
    now = 0.0

    # Track which requests have been submitted and completed.
    next_request = 0
    completed = []

    # Advance the simulation one scheduling tick at a time.
    while len(completed) < len(ordered):

        # First, submit every request that has arrived by this tick.
        while (
            next_request < len(ordered)
            and ordered[next_request]["t_arrive"] * 1000 <= now
        ):
            req = ordered[next_request]

            # Determine the request's context-length tier.
            tier = context_tier(req["ctx"], tiers)

            # Prefer engines assigned to this tier. If the tier has
            # no engines, fall back to the entire pool.
            candidates = groups[tier]

            if not candidates:
                candidates = all_engine_ids

            # Choose the engine with the fewest pending tokens.
            # The engine index breaks ties in favor of the lower index.
            engine_idx = min(
                candidates,
                key=lambda j: (
                    engines[j].pending_tokens(),
                    j,
                )
            )

            engines[engine_idx].submit(req)
            next_request += 1

        # Next, execute one budgeted prefill step on every engine.
        for engine in engines:
            finished = engine.step(now)
            completed.extend(finished)

        # Advance to the next scheduling tick. This also makes the
        # final makespan reflect the discrete scheduling interval.
        now += tick_ms

    # Group the original requests by their actual context-length tier.
    # Only tiers containing requests are included in the summary.
    requests_by_tier = {}

    for req in request_list:
        tier = context_tier(req["ctx"], tiers)
        requests_by_tier.setdefault(tier, []).append(req)

    # Calculate TTFT percentiles for every populated tier.
    by_tier = {}

    for tier in sorted(requests_by_tier):
        tier_requests = requests_by_tier[tier]
        ttfts = [
            float(req["ttft_ms"])
            for req in tier_requests
        ]

        by_tier[tier] = {
            "count": len(tier_requests),
            "p50_ms": round(
                float(np.percentile(ttfts, 50)), 1
            ),
            "p99_ms": round(
                float(np.percentile(ttfts, 99)), 1
            ),
        }

    # Calculate total engine busy time and overall pool utilization.
    total_busy_ms = sum(engine.busy_ms for engine in engines)
    total_capacity_ms = n_engines * now

    utilization = (
        round(total_busy_ms / total_capacity_ms, 4)
        if total_capacity_ms > 0
        else 0.0
    )

    # Return TTFT by request ID, rounded to one decimal place.
    ttft_by_id = {
        req["id"]: round(float(req["ttft_ms"]), 1)
        for req in request_list
    }

    return {
        "by_tier": by_tier,
        "makespan_ms": round(float(now), 1),
        "utilization": utilization,
        "ttft_ms": ttft_by_id,
    }

# Step 9 - eplb_assign
def moe_lb(rank_loads):
    # Calculate the mean load across all ranks.
    if not rank_loads:
        return 1.0

    mean_load = sum(rank_loads) / len(rank_loads)

    # Avoid division by zero when all ranks have zero load.
    if mean_load == 0:
        return 1.0

    # Ratio of the slowest (most heavily loaded) rank to the mean load.
    return round(max(rank_loads) / mean_load, 4)


def eplb_assign(loads, n_ranks, redundant_per_rank):
    # Each expert's load is initially assigned to one copy.
    loads = list(loads)
    n_experts = len(loads)

    if n_ranks <= 0:
        raise ValueError("n_ranks must be positive")

    copies = [1] * n_experts

    # Each rank has its base number of expert slots plus redundant slots.
    base_slots = n_experts // n_ranks
    slots_per_rank = base_slots + redundant_per_rank

    # Allocate redundant copies one slot at a time.
    spare_slots = n_ranks * redundant_per_rank

    for _ in range(spare_slots):
        # An expert cannot have more copies than there are ranks.
        eligible = [
            i for i in range(n_experts)
            if copies[i] < n_ranks
        ]

        # Stop if every expert already has the maximum number of copies.
        if not eligible:
            break

        # Give the next copy to the expert with the greatest load per copy.
        # Ties go to the expert with the lower index.
        expert = max(
            eligible,
            key=lambda i: (loads[i] / copies[i], -i)
        )

        copies[expert] += 1

    # Initialize per-rank placement, loads, and remaining slot capacity.
    placement = [[] for _ in range(n_ranks)]
    rank_loads = [0.0] * n_ranks
    free_slots = [slots_per_rank] * n_ranks

    # Place the most heavily loaded experts per copy first.
    # Ties are broken by the lower expert index.
    expert_order = sorted(
        range(n_experts),
        key=lambda i: (-loads[i] / copies[i], i)
    )

    for expert in expert_order:
        # Find ranks with available capacity, ordered by current load.
        available_ranks = sorted(
            (
                rank for rank in range(n_ranks)
                if free_slots[rank] > 0
            ),
            key=lambda rank: (rank_loads[rank], rank)
        )

        # Use distinct ranks for an expert's replicas. If fewer ranks
        # have available slots than requested, reduce its copy count.
        actual_copies = min(copies[expert], len(available_ranks))
        copies[expert] = actual_copies

        # No placement is possible if no rank has an available slot.
        if actual_copies == 0:
            continue

        # Divide the expert's load equally among its actual copies.
        load_per_copy = loads[expert] / actual_copies

        # Place copies on the least-loaded available ranks.
        for rank in available_ranks[:actual_copies]:
            placement[rank].append(expert)
            free_slots[rank] -= 1
            rank_loads[rank] += load_per_copy

    # Sort expert IDs within each rank for deterministic output.
    for rank in placement:
        rank.sort()

    # Return rank loads rounded to four decimal places and the
    # expert-parallel load imbalance calculated from the actual loads.
    return {
        "replicas": copies,
        "placement": placement,
        "rank_loads": [round(load, 4) for load in rank_loads],
        "moe_lb": moe_lb(rank_loads),
    }

# Step 10 - expert_loads
def expert_loads(n_tokens, n_experts, k, skew, rng, perm=None):
    # Assign each expert a power-law probability.
    # Lower-indexed experts receive higher probabilities when skew > 0.
    p = np.arange(1, n_experts + 1, dtype=float) ** (-skew)

    # Normalize probabilities so they sum to one.
    p = p / p.sum()

    # Optionally reorder probabilities according to the supplied permutation.
    # This changes which expert positions receive the highest traffic.
    if perm is not None:
        p = p[np.asarray(perm)]

        # Normalize again to ensure a valid probability distribution.
        p = p / p.sum()

    # Model n_tokens * k independent expert selections using a multinomial
    # distribution and return the integer token count for each expert.
    return rng.multinomial(n_tokens * k, p)

# Step 11 - moe_lb_trace
def moe_lb_trace(step_loads, n_ranks, redundant_per_rank, window, period):
    # Convert the input into a NumPy array.
    step_loads = np.asarray(step_loads)

    # Get the number of decode steps and experts.
    n_steps, n_experts = step_loads.shape

    if n_ranks <= 0:
        raise ValueError("n_ranks must be positive")

    if period <= 0:
        raise ValueError("period must be positive")

    if n_experts % n_ranks != 0:
        raise ValueError("n_experts must be divisible by n_ranks")

    # Initially distribute experts contiguously across ranks, with one
    # copy of every expert.
    experts_per_rank = n_experts // n_ranks

    placement = [
        list(range(
            rank * experts_per_rank,
            (rank + 1) * experts_per_rank
        ))
        for rank in range(n_ranks)
    ]

    replicas = [1] * n_experts

    # Store the load-imbalance metric for each decode step.
    imbalance_trace = []
    reassignments = 0

    for t in range(n_steps):
        # Reassign experts periodically, before evaluating this step.
        # Do not reassign at t = 0 because the initial placement is used.
        if t > 0 and t % period == 0:
            # Aggregate expert loads over the preceding observation window.
            start = max(0, t - window)
            recent_loads = step_loads[start:t].sum(axis=0)

            # Rebalance experts using their recent aggregate loads.
            result = eplb_assign(
                recent_loads,
                n_ranks,
                redundant_per_rank
            )

            placement = result["placement"]
            replicas = result["replicas"]
            reassignments += 1

        # Calculate each rank's load for the current step.
        # Each expert's load is divided equally among its replicas.
        rank_loads = []

        for rank in range(n_ranks):
            rank_load = sum(
                float(step_loads[t, expert]) / replicas[expert]
                for expert in placement[rank]
            )
            rank_loads.append(rank_load)

        # Record the current step's expert-parallel load imbalance.
        imbalance_trace.append(moe_lb(rank_loads))

    # Calculate the average imbalance across all steps.
    mean_imbalance = (
        round(float(np.mean(imbalance_trace)), 4)
        if imbalance_trace
        else 0.0
    )

    return {
        "moe_lb": imbalance_trace,
        "mean": mean_imbalance,
        "reassignments": reassignments,
    }

# Step 12 - route_decode
def route_decode(need_tokens, gpus, bytes_per_token):
    # Calculate the KV-cache memory required by this request.
    need = need_tokens * bytes_per_token

    best_gpu = -1
    best_headroom = -1

    # Examine GPUs in index order so ties favor the lower index.
    for i, gpu in enumerate(gpus):
        # Calculate the remaining batch slots and KV-cache memory.
        free_slots = gpu["slots"] - gpu["used_slots"]
        free_kv = gpu["kv_bytes"] - gpu["used_kv"]

        # A GPU is feasible only if it has a free slot and enough memory.
        if free_slots <= 0 or free_kv < need:
            continue

        # Calculate how many additional copies of this request could fit.
        # When the request requires zero KV memory, available slots are
        # the only limiting factor.
        if need == 0:
            headroom = free_slots
        else:
            headroom = min(
                free_slots,
                int(free_kv // need)
            )

        # Update only when headroom is strictly greater, preserving
        # the lower GPU index when multiple GPUs tie.
        if headroom > best_headroom:
            best_headroom = headroom
            best_gpu = i

    # Return -1 if no GPU can accommodate the request.
    return best_gpu

# Step 13 - DecodePool
class DecodePool:
    def __init__(
        self,
        n_gpus,
        slots,
        kv_bytes,
        bytes_per_token,
        params,
        router="headroom",
        tokens_per_step=1.0,
    ):
        self.bytes_per_token = bytes_per_token
        self.params = params
        self.router = router
        self.tokens_per_step = tokens_per_step

        # Initialize GPU capacity, active sequences, and independent clocks.
        self.gpus = [
            {
                "slots": slots,
                "kv_bytes": float(kv_bytes),
                "used_slots": 0,
                "used_kv": 0.0,
                "seqs": [],
                "clock": 0.0,
                "busy_ms": 0.0,
            }
            for _ in range(n_gpus)
        ]

        # FIFO queue of (request, admission_time_ms).
        self.waiting = collections.deque()
        self.finished = []

    def _choose_gpu(self, req):
        # Reserve memory for the request's full context and output.
        need_tokens = req["ctx"] + req["out"]
        need_bytes = need_tokens * self.bytes_per_token

        if self.router == "headroom":
            # Prefer the feasible GPU with the greatest request headroom.
            return route_decode(
                need_tokens,
                self.gpus,
                self.bytes_per_token,
            )

        if self.router == "count":
            # Choose the feasible GPU with the fewest occupied slots.
            # The GPU index resolves ties.
            candidates = []

            for i, gpu in enumerate(self.gpus):
                free_slots = gpu["slots"] - gpu["used_slots"]
                free_kv = gpu["kv_bytes"] - gpu["used_kv"]

                if free_slots > 0 and free_kv >= need_bytes:
                    candidates.append((gpu["used_slots"], i))

            return min(candidates)[1] if candidates else -1

        raise ValueError("router must be 'headroom' or 'count'")

    def _drain_waiting(self):
        # Admit requests in FIFO order until the head cannot fit.
        while self.waiting:
            req, t_in = self.waiting[0]
            gpu_idx = self._choose_gpu(req)

            if gpu_idx < 0:
                break

            self.waiting.popleft()
            gpu = self.gpus[gpu_idx]

            reserved_tokens = req["ctx"] + req["out"]
            reserved_bytes = reserved_tokens * self.bytes_per_token

            # Reserve the batch slot and KV memory.
            gpu["used_slots"] += 1
            gpu["used_kv"] += reserved_bytes

            # The GPU cannot start before the request's admission time.
            gpu["clock"] = max(gpu["clock"], t_in)

            gpu["seqs"].append({
                "req": req,
                "ctx": req["ctx"],
                "produced": 0.0,
                "t_in": t_in,
                "steps": 0,
                "first": None,
            })

    def admit(self, req, now_ms):
        # Queue the request, then admit as many FIFO requests as can fit.
        self.waiting.append((req, now_ms))
        self._drain_waiting()

    def run_until(self, now_ms):
        # Advance every GPU using its own clock.
        for gpu_idx, gpu in enumerate(self.gpus):
            while gpu["seqs"] and gpu["clock"] < now_ms:
                seqs = gpu["seqs"]

                # Calculate the duration of one decode step for this batch.
                batch_size = len(seqs)
                total_context = sum(seq["ctx"] for seq in seqs)

                dt = decode_step_time(
                    batch_size,
                    total_context,
                    self.params,
                )

                gpu["clock"] += dt
                gpu["busy_ms"] += dt

                remaining = []

                for seq in seqs:
                    # Generate tokens and advance the sequence context.
                    seq["produced"] += self.tokens_per_step
                    seq["ctx"] += self.tokens_per_step
                    seq["steps"] += 1

                    # Record the first-token time at the end of step one.
                    if seq["first"] is None:
                        seq["first"] = gpu["clock"]

                    if seq["produced"] >= seq["req"]["out"]:
                        req = seq["req"]

                        req["t_done"] = gpu["clock"]
                        req["t_first_token"] = seq["first"]

                        # Exclude the first-token step when measuring
                        # average time between subsequent generated tokens.
                        if seq["steps"] > 1:
                            req["tbt_ms"] = round(
                                (gpu["clock"] - seq["first"])
                                / (seq["steps"] - 1),
                                3,
                            )
                        else:
                            req["tbt_ms"] = 0.0

                        req["gpu"] = gpu_idx
                        self.finished.append(req)

                        # Release the originally reserved resources.
                        gpu["used_slots"] -= 1
                        reserved_bytes = (
                            req["ctx"] + req["out"]
                        ) * self.bytes_per_token

                        gpu["used_kv"] = max(
                            0.0,
                            gpu["used_kv"] - reserved_bytes,
                        )
                    else:
                        remaining.append(seq)

                gpu["seqs"] = remaining

        # Do not admit newly unblocked requests inside this call.
        # The next admission or drain_all iteration handles the queue.
        return len(self.finished)

    def drain_all(self):
        # Continue until no runnable sequences or admissible waiting
        # requests remain.
        while self.waiting or any(gpu["seqs"] for gpu in self.gpus):
            self._drain_waiting()

            # Requests may remain queued if none can fit on any GPU.
            if not any(gpu["seqs"] for gpu in self.gpus):
                break

            latest_clock = max(
                gpu["clock"] for gpu in self.gpus
            )

            # Snapshot clocks and sequence progress. A step can make
            # progress even when no request has finished yet.
            before = (
                tuple(gpu["clock"] for gpu in self.gpus),
                tuple(
                    tuple(seq["steps"] for seq in gpu["seqs"])
                    for gpu in self.gpus
                ),
                len(self.finished),
                len(self.waiting),
            )

            # Advance past the latest clock; slow GPUs may need multiple
            # steps to reach this horizon.
            self.run_until(latest_clock + 1.0)

            after = (
                tuple(gpu["clock"] for gpu in self.gpus),
                tuple(
                    tuple(seq["steps"] for seq in gpu["seqs"])
                    for gpu in self.gpus
                ),
                len(self.finished),
                len(self.waiting),
            )

            # Stop only if the simulation genuinely made no progress.
            if after == before:
                break

        return self.finished

# Step 14 - block_hashes
MOD = 2 ** 61 - 1


def block_hashes(tokens, block=128):
    # Start the rolling hash at zero.
    h = 0
    hashes = []

    # Process only complete blocks; ignore any trailing partial block.
    n_full_blocks = len(tokens) // block

    for i in range(n_full_blocks):
        start = i * block
        end = start + block

        # Fold each token into the hash, preserving the entire prefix.
        for token in tokens[start:end]:
            h = (h * 131 + int(token) + 1) % MOD

        # Record the cumulative hash for this full block.
        hashes.append(h)

    return hashes

# Step 15 - prefix_hit
def prefix_hit(blocks, swa, hashes, n_tokens, block=128, window=128, backtrack=6):
    # Maximum number of prompt tokens eligible for prefix caching.
    cap = max(0, n_tokens - (window + backtrack))

    # Count the longest leading run of cached, complete blocks,
    # limited by the cap.
    max_blocks = min(cap // block, len(hashes))
    n_hit_blocks = 0

    for i in range(max_blocks):
        if hashes[i] not in blocks:
            break
        n_hit_blocks += 1

    block_hit = n_hit_blocks * block

    # No leading cached blocks means no cache hit.
    if n_hit_blocks == 0:
        return {
            "cap": cap,
            "block_hit": 0,
            "hit": 0,
        }

    # A single cached block needs no sliding-window state.
    if n_hit_blocks == 1:
        hit = block_hit
    else:
        # Check the last and second-to-last hashes in the cached run.
        last_hash = hashes[n_hit_blocks - 1]
        previous_hash = hashes[n_hit_blocks - 2]

        # Either state validates the entire cached prefix.
        if last_hash in swa or previous_hash in swa:
            hit = block_hit
        else:
            # Otherwise, back off by one complete block.
            hit = block_hit - block

    return {
        "cap": cap,
        "block_hit": block_hit,
        "hit": max(0, min(hit, cap)),
    }

# Step 16 - KVStore
class KVStore:
    def __init__(self, capacity_bytes, ttl_s, bytes_per_block, watermark=0.9):
        self.capacity_bytes = capacity_bytes
        self.ttl_s = ttl_s
        self.bytes_per_block = float(bytes_per_block)
        self.watermark = watermark

        # Entries are ordered from least recently used to most recently used.
        self.entries = collections.OrderedDict()

        self.requested = 0
        self.hit = 0

    def __contains__(self, h):
        return h in self.entries

    def size_bytes(self):
        # Return a float, including when the block count is an integer.
        return len(self.entries) * self.bytes_per_block

    def lookup(self, hashes, now):
        # Every supplied block counts as a request.
        self.requested += len(hashes)

        run_length = 0

        # Count only the leading consecutive run of cached hashes.
        for h in hashes:
            if h not in self.entries:
                break

            # Refresh its last-use time and move it to the MRU end.
            self.entries[h] = now
            self.entries.move_to_end(h)
            run_length += 1

        self.hit += run_length
        return run_length

    def insert(self, hashes, now):
        # Insert or refresh keys, moving each to the MRU end.
        for h in hashes:
            self.entries[h] = now
            self.entries.move_to_end(h)

        return self.gc(now)

    def gc(self, now):
        evicted = 0
        max_size = self.watermark * self.capacity_bytes

        # Remove expired or over-capacity entries from the LRU end.
        while self.entries:
            oldest_hash, last_use = next(iter(self.entries.items()))

            expired = now - last_use > self.ttl_s
            over_capacity = self.size_bytes() > max_size

            if not expired and not over_capacity:
                break

            self.entries.popitem(last=False)
            evicted += 1

        return evicted

    def hit_rate(self):
        if self.requested == 0:
            return 0.0

        return self.hit / self.requested

# Step 17 - make_reuse_stream
def make_reuse_stream(n, seed, cls, days=14.0):
    # Create a reproducible random number generator.
    rng = np.random.default_rng(seed)

    if cls == "chat":
        # Chat traffic reuses prefixes drawn from a Zipf distribution.
        prefix_ids = rng.zipf(1.5, size=n) % 20000

        # Spread requests uniformly across the specified time period.
        t = rng.uniform(0.0, days, size=n)

        # Each chat request uses 16 blocks.
        n_blocks = np.full(n, 16, dtype=int)

    else:
        # Agent traffic is organized into sessions, with at least one
        # session even when n is small.
        n_sessions = max(1, n // 8)

        # Generate session start times.
        start = rng.uniform(0.0, days, size=n_sessions)

        # Assign each request to a session.
        session = rng.integers(0, n_sessions, size=n)

        # Requests in the same session share a prefix ID.
        prefix_ids = session + 100000

        # Generate request times relative to their session start times.
        t = start[session] + rng.exponential(0.05, size=n)

        # Each agent request uses 300 blocks.
        n_blocks = np.full(n, 300, dtype=int)

    # Sort requests chronologically while preserving their original
    # order when timestamps are equal.
    order = np.argsort(t, kind="stable")

    # Return (time in days, prefix ID, block count), using Python
    # floats and ints rather than NumPy scalar types.
    return [
        (
            float(t[i]),
            int(prefix_ids[i]),
            int(n_blocks[i]),
        )
        for i in order
    ]

# Step 18 - hit_rate_vs_ttl
def hit_rate_vs_ttl(stream, ttls_days, capacity_blocks):
    # Store the measured hit rate for each requested TTL.
    results = {}

    # Evaluate the same request stream independently for every TTL.
    for ttl in ttls_days:
        store = KVStore(
            capacity_blocks,
            ttl,
            1.0,
            watermark=1.0,
        )

        # Replay requests in their original chronological order.
        for t, pid, nb in stream:
            # Construct the block keys for this request's prefix.
            keys = [(pid, j) for j in range(nb)]

            # Look up the cached prefix before inserting its blocks.
            store.lookup(keys, t)

            # Insert or refresh the request's blocks at the current time.
            store.insert(keys, t)

        # Record the hit rate rounded to four decimal places.
        results[ttl] = round(store.hit_rate(), 4)

    return results

# Step 19 - plan_verification
def plan_verification(conf, budget):
    # Convert confidence values to a NumPy array.
    conf = np.asarray(conf, dtype=float)

    R, D = conf.shape

    # Compute cumulative acceptance probabilities for each sequence.
    # Verification stops at the first rejected draft.
    cumulative = np.cumprod(conf, axis=1)

    # Create a global list of (expected value, sequence index, draft index).
    candidates = [
        (float(cumulative[r, j]), r, j)
        for r in range(R)
        for j in range(D)
    ]

    # Rank by expected value descending, then sequence index and
    # draft index ascending to resolve ties deterministically.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    # Select the highest-value drafts within the global verification budget.
    n_verified = min(max(0, int(budget)), R * D)
    selected = candidates[:n_verified]

    # Count selected drafts for each sequence.
    lengths = [0] * R
    expected_accepted = 0.0

    for value, r, j in selected:
        lengths[r] += 1
        expected_accepted += value

    return {
        "lengths": lengths,
        "expected_accepted": round(expected_accepted, 4),
        "verified": n_verified,
    }

# Step 20 - verification_budget
def verification_budget(alpha, conc, t0, kappa):
    # Convert acceptance probabilities to a NumPy array.
    alpha = np.asarray(alpha, dtype=float)

    table = []
    throughputs = []
    accepted_by_length = []

    # Evaluate every possible verification length, including m = 0
    # (no speculative draft tokens).
    for m in range(len(alpha) + 1):
        # Expected accepted tokens per sequence per step.
        accepted = 1.0 + float(np.sum(alpha[:m]))
        accepted_by_length.append(accepted)

        # Time required for a decode step, in milliseconds.
        step_ms = t0 + kappa * conc * (1 + m)

        # Throughput in tokens per second.
        if step_ms > 0:
            throughput = conc * accepted / step_ms * 1000.0
        else:
            throughput = 0.0

        throughputs.append(throughput)

        # Store the reported throughput rounded to one decimal place.
        table.append((m, round(throughput, 1)))

    # Choose the smallest verification length attaining maximum throughput.
    best_m = max(
        range(len(throughputs)),
        key=lambda m: (throughputs[m], -m)
    )

    # Calculate expected tokens generated per sequence per step
    # at the optimal verification length.
    tokens_per_step = round(accepted_by_length[best_m], 4)

    return {
        "table": table,
        "best_m": best_m,
        "tokens_per_step": tokens_per_step,
    }

# Step 21 - async_step_times
def async_step_times(n_steps, t_sched, t_fwd, t_sample, t_post, t_launch, mode):
    # GPU execution time for one decode step.
    gpu_step_ms = t_fwd + t_sample

    # Serial execution: scheduling, forward pass, sampling, and
    # post-processing occur consecutively for every step.
    if mode == "serial":
        step_ms = t_sched + t_fwd + t_sample + t_post
        emit_times = [
            (i + 1) * step_ms
            for i in range(n_steps)
        ]

        makespan = emit_times[-1] if emit_times else 0.0

    elif mode == "async":
        # Validate the execution mode above; model independent CPU and GPU
        # availability times for the overlapped schedule.
        cpu_free = 0.0
        gpu_free = 0.0

        # Store launch completion, GPU completion, and emission times
        # for each decode step.
        launch_times = [0.0] * n_steps
        gpu_finish_times = [0.0] * n_steps
        emit_times = [0.0] * n_steps

        if n_steps > 0:
            # The CPU schedules and launches the first step.
            cpu_free += t_sched + t_launch
            launch_times[0] = cpu_free

            for n in range(n_steps):
                # The GPU starts this step when both it and the step
                # are available.
                gpu_start = max(gpu_free, launch_times[n])
                gpu_finish = gpu_start + gpu_step_ms

                gpu_finish_times[n] = gpu_finish
                gpu_free = gpu_finish

                # Schedule and launch the next step while GPU work runs,
                # if another step exists.
                if n + 1 < n_steps:
                    cpu_free += t_sched + t_launch
                    launch_times[n + 1] = cpu_free

                # Post-process step n-1 once the CPU is free and that
                # step's GPU execution has completed.
                if n - 1 >= 0:
                    post_start = max(
                        cpu_free,
                        gpu_finish_times[n - 1],
                    )
                    emit_times[n - 1] = post_start + t_post
                    cpu_free = emit_times[n - 1]

            # Post-process the final step after the loop.
            last = n_steps - 1
            post_start = max(cpu_free, gpu_finish_times[last])
            emit_times[last] = post_start + t_post

        makespan = emit_times[-1] if emit_times else 0.0

    else:
        raise ValueError("mode must be 'serial' or 'async'")

    # Calculate the fraction of the makespan during which the GPU
    # was not performing forward-pass or sampling work.
    if makespan > 0:
        gpu_idle_frac = 1.0 - (n_steps * gpu_step_ms) / makespan
    else:
        gpu_idle_frac = 0.0

    # Mean interval between emissions, or None for zero/one emission.
    if len(emit_times) > 1:
        mean_step = sum(
            emit_times[i] - emit_times[i - 1]
            for i in range(1, len(emit_times))
        ) / (len(emit_times) - 1)
    else:
        mean_step = None

    # Return all numeric values rounded to four decimal places.
    return {
        "emit_times": [round(float(t), 4) for t in emit_times],
        "makespan": round(float(makespan), 4),
        "gpu_idle_frac": round(float(gpu_idle_frac), 4),
        "mean_step": (
            round(float(mean_step), 4)
            if mean_step is not None
            else None
        ),
    }

# Step 22 - decode_throughput_curve
def decode_throughput_curve(conc_list, alpha, t0, kappa, fixed_m=5):
    results = []

    # Compare fixed-length and dynamically optimized verification
    # at each requested concurrency level.
    for conc in conc_list:
        profile = verification_budget(alpha, conc, t0, kappa)

        # Fixed-length verification uses the throughput at fixed_m,
        # capped at the number of available speculative positions.
        m_fixed = min(fixed_m, len(alpha))
        fixed = profile["table"][m_fixed][1]

        # Dynamic verification chooses the length with maximum throughput.
        best_m = profile["best_m"]
        dynamic = profile["table"][best_m][1]

        # Calculate the percentage throughput improvement.
        gain_pct = (
            round((dynamic / fixed - 1.0) * 100.0, 2)
            if fixed != 0
            else 0.0
        )

        results.append({
            "conc": conc,
            "fixed": fixed,
            "best_m": best_m,
            "dynamic": dynamic,
            "gain_pct": gain_pct,
        })

    return results

# Step 23 - simulate_cluster
def simulate_cluster(workload, n_prefill, n_decode, cfg):
    # Copy every request so the caller's workload remains unchanged.
    requests = [dict(req) for req in workload]
    n_requests = len(requests)

    # Handle an empty workload without running either pool.
    if n_requests == 0:
        return {
            "tokens_per_s": 0.0,
            "ttft_p50_ms": 0.0,
            "ttft_p99_ms": 0.0,
            "tbt_p50_ms": 0.0,
            "tbt_p99_ms": 0.0,
            "makespan_ms": 0.0,
            "prefill_utilization": 0.0,
            "decode_utilization": 0.0,
            "n_requests": 0,
        }

    # Run the prefill pool. It stamps t_prefill_done on each
    # completed request, with timestamps expressed in milliseconds.
    prefill_stats = run_prefill_pool(
        requests,
        n_prefill,
        cfg["coeffs"],
        cfg["budget_ns"],
        tiered=cfg.get("tiered", True),
    )

    # Construct the decode pool using the supplied configuration.
    pool = DecodePool(
        n_decode,
        cfg["slots"],
        cfg["kv_bytes"],
        cfg["bytes_per_token"],
        cfg["decode"],
        router=cfg.get("router", "headroom"),
        tokens_per_step=cfg.get("tokens_per_step", 1.0),
    )

    # Transfer requests in prefill-completion order, with request ID
    # as the deterministic tie-breaker.
    completed_prefill = sorted(
        requests,
        key=lambda r: (r["t_prefill_done"], r["id"]),
    )

    link_bytes_per_ms = cfg.get("link_bytes_per_ms", 6.25e6)

    if link_bytes_per_ms <= 0:
        raise ValueError("link_bytes_per_ms must be positive")

    for req in completed_prefill:
        # Model transfer time across the interconnect in milliseconds.
        transfer_ms = (
            req["ctx"] * cfg["bytes_per_token"] / link_bytes_per_ms
        )
        t_in = req["t_prefill_done"] + transfer_ms

        # Advance the decode GPUs to the transfer-completion time,
        # then admit the request.
        pool.run_until(t_in)
        pool.admit(req, t_in)

    # Drain all remaining decode work.
    pool.drain_all()

    # Collect end-to-end latency, first-token latency, and inter-token
    # latency from the original request copies, now updated by DecodePool.
    done_requests = pool.finished

    if not done_requests:
        return {
            "tokens_per_s": 0.0,
            "ttft_p50_ms": 0.0,
            "ttft_p99_ms": 0.0,
            "tbt_p50_ms": 0.0,
            "tbt_p99_ms": 0.0,
            "makespan_ms": 0.0,
            "prefill_utilization": round(
                float(prefill_stats["utilization"]), 4
            ),
            "decode_utilization": 0.0,
            "n_requests": n_requests,
        }

    # The makespan is the latest decode completion time, in milliseconds.
    makespan_ms = max(req["t_done"] for req in done_requests)

    # Total generated output tokens divided by elapsed simulation time.
    total_output_tokens = sum(req["out"] for req in done_requests)
    tokens_per_s = (
        total_output_tokens * 1000.0 / makespan_ms
        if makespan_ms > 0
        else 0.0
    )

    # TTFT is measured from the original request arrival time.
    ttft_values = [
        req["t_first_token"] - req["t_arrive"] * 1000.0
        for req in done_requests
    ]

    ttft_p50_ms = float(np.percentile(ttft_values, 50))
    ttft_p99_ms = float(np.percentile(ttft_values, 99))

    # TBT percentiles include only requests with output length > 1.
    tbt_values = [
        req["tbt_ms"]
        for req in done_requests
        if req["out"] > 1
    ]

    if tbt_values:
        tbt_p50_ms = float(np.percentile(tbt_values, 50))
        tbt_p99_ms = float(np.percentile(tbt_values, 99))
    else:
        tbt_p50_ms = 0.0
        tbt_p99_ms = 0.0

    # Decode utilization is total GPU busy time divided by total
    # available decode-GPU time over the makespan.
    total_decode_busy_ms = sum(
        gpu["busy_ms"] for gpu in pool.gpus
    )
    total_decode_capacity_ms = n_decode * makespan_ms

    decode_utilization = (
        total_decode_busy_ms / total_decode_capacity_ms
        if total_decode_capacity_ms > 0
        else 0.0
    )

    return {
        "tokens_per_s": round(tokens_per_s, 1),
        "ttft_p50_ms": round(ttft_p50_ms, 1),
        "ttft_p99_ms": round(ttft_p99_ms, 1),
        "tbt_p50_ms": round(tbt_p50_ms, 2),
        "tbt_p99_ms": round(tbt_p99_ms, 2),
        "makespan_ms": round(makespan_ms, 1),
        "prefill_utilization": round(
            float(prefill_stats["utilization"]), 4
        ),
        "decode_utilization": round(decode_utilization, 4),
        "n_requests": n_requests,
    }

# Step 24 - sweep_pd_ratio
def sweep_pd_ratio(workload, total_gpus, cfg, prefill_fracs=(0.25, 0.5, 0.75)):
    results = []

    # Evaluate each requested prefill fraction in the supplied order.
    for f in prefill_fracs:
        # Allocate at least one GPU to prefill and reserve the remaining
        # GPUs for decode, following the specified rounding rule.
        n_prefill = max(
            1,
            min(
                total_gpus - 1,
                int(round(f * total_gpus)),
            ),
        )
        n_decode = total_gpus - n_prefill

        # Simulate the cluster for this GPU allocation.
        metrics = simulate_cluster(
            workload,
            n_prefill,
            n_decode,
            cfg,
        )

        # Collect the requested performance and utilization metrics.
        results.append({
            "n_prefill": n_prefill,
            "n_decode": n_decode,
            "tokens_per_s": metrics["tokens_per_s"],
            "ttft_p99_ms": metrics["ttft_p99_ms"],
            "tbt_p99_ms": metrics["tbt_p99_ms"],
            "prefill_utilization": metrics["prefill_utilization"],
            "decode_utilization": metrics["decode_utilization"],
        })

    return results

