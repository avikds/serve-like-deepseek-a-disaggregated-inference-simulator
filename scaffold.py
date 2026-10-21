"""
Serve Like DeepSeek: A Disaggregated Inference Simulator scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Serve Like DeepSeek: a disaggregated inference simulator.

Story: a cluster that splits prefill from decode is a set of scheduling decisions, and each
one changes a number you can measure. Context tiers take head-of-line blocking out of the
chat tier. Redundant experts take the imbalance out of expert parallelism. A TTL on the
prefix store trades capacity for hit rate, and the trade differs by traffic class.
Verification budgets chosen from the throughput curve pay at high concurrency. Overlapping
the scheduler with the forward pass makes the step as long as the GPU, not the sum of both.
And the split of GPUs between prefill and decode traces a frontier of throughput against
latency with a price on every point.
"""
import numpy as np


def main() -> None:
    rep = serving_report(seed=0, n_requests=160, total_gpus=8)

    print("1. The workload and its demand")
    print(f"   requests by class: {rep['workload']}")
    print(f"   tier counts {rep['tiers']['counts']} | demand shares {rep['tiers']['shares']}")
    print("   The top tiers hold a few percent of the requests and most of the prefill demand.")

    print("\n2. Context tiers versus a single FIFO (time to first token, ms)")
    for k in sorted(rep["prefill_tiered"]):
        t, f = rep["prefill_tiered"][k], rep["prefill_fifo"].get(k, {"p50_ms": float("nan"), "p99_ms": float("nan")})
        print(f"   tier {k}: tiered p50 {t['p50_ms']:>9.1f} p99 {t['p99_ms']:>9.1f} | fifo p50 {f['p50_ms']:>9.1f} p99 {f['p99_ms']:>9.1f}")
    print("   Tiers move waiting time from the many short requests to the few long ones.")

    print("\n3. Expert-parallel load balancing (mean moe_lb over 60 steps, 64 experts on 8 ranks)")
    print(f"   no redundancy {rep['eplb']['none']:.3f} | two redundant experts per rank {rep['eplb']['redundant']:.3f}")
    print("   Replicas of the hot experts bring the slowest rank down toward the mean.")

    print("\n4. Prefix-store hit rate against time-to-live (days)")
    print(f"   chatbot traffic: {rep['hit_rate']['chat']}")
    print(f"   agent traffic:   {rep['hit_rate']['agent']}")
    print("   The long tail of chat prefixes pays for a shorter TTL; bursty agent sessions do not.")

    print("\n5. Dynamic verification against fixed-5 (tokens per second per rank)")
    for r in rep["dspark"]:
        print(f"   {r['conc']:>3} seqs/rank: fixed {r['fixed']:>8.1f} | best length {r['best_m']} {r['dynamic']:>8.1f} | gain {r['gain_pct']:+.2f}%")
    print(f"   asynchronous scheduling: serial step {rep['async']['serial']} ms, overlapped step {rep['async']['async']} ms")

    print("\n6. Splitting 8 GPUs between prefill and decode")
    for r in rep["sweep"]:
        key = f"{r['n_prefill']}P{r['n_decode']}D"
        print(f"   {key}: {r['tokens_per_s']:>7.1f} tok/s | TTFT p99 {r['ttft_p99_ms']:>9.1f} ms | TBT p99 {r['tbt_p99_ms']:>6.2f} ms | util prefill {r['prefill_utilization']:.2f} decode {r['decode_utilization']:.2f} | ${rep['cost'][key]:.2f} per million tokens")
    front = [str(r["n_prefill"]) + "P" + str(r["n_decode"]) + "D" for r in rep["frontier"]]
    print(f"   non-dominated operating points: {front}")
    print("   Every point on the frontier is a different cluster; the price says what each one charges.")


if __name__ == "__main__":
    main()

