# Serve Like DeepSeek: A Disaggregated Inference Simulator

DeepSeek serves V4.1 Flash from a cluster that splits prefill from decode, runs decode at EP64 with a load balancer that replicates hot experts, schedules prefill by a time budget solved from a fitted cost model, routes decode requests on both batch slots and KV memory, keeps the prefix cache on a shared storage cluster with a TTL, verifies speculative drafts with a budget chosen from the throughput curve, and hides the scheduler behind the forward pass. Each of those is a scheduling decision, and the only way to see how they interact is to run the cluster. This project builds that cluster as a discrete-event simulator in plain Python and numpy. You generate a realistic request mix, fit the quadratic prefill cost model and solve chunk sizes from a budget, build a tiered prefill pool and watch head-of-line blocking disappear, build an expert-parallel balancer with redundant experts and a two-dimensional decode router, model a tiered KV store with hash-chained prefix hits and measure hit rate against TTL, choose the speculative verification budget from a profiled throughput curve and overlap scheduling with execution, and finally assemble the whole system to sweep the prefill-to-decode GPU ratio, draw the throughput-versus-latency frontier and price an operating point in dollars per million tokens. Every step prints a number you can hold against DeepSeek's published deployment notes.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_workload
- [x] **2.** fit_cost_model
- [x] **3.** tier_demand
- [x] **4.** decode_step_time
- [x] **5.** chunk_for

---

Built on Deep-ML.
