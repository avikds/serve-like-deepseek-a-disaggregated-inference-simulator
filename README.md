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
- [x] **6.** plan_prefill_step
- [x] **7.** PrefillEngine
- [x] **8.** run_prefill_pool
- [x] **9.** eplb_assign
- [x] **10.** expert_loads
- [x] **11.** moe_lb_trace
- [x] **12.** route_decode
- [x] **13.** DecodePool
- [x] **14.** block_hashes
- [x] **15.** prefix_hit
- [x] **16.** KVStore
- [x] **17.** make_reuse_stream
- [x] **18.** hit_rate_vs_ttl
- [x] **19.** plan_verification
- [x] **20.** verification_budget
- [x] **21.** async_step_times
- [x] **22.** decode_throughput_curve
- [x] **23.** simulate_cluster
- [x] **24.** sweep_pd_ratio
- [x] **25.** pareto_frontier
- [x] **26.** serving_report

## Results

```
1. The workload and its demand
   requests by class: {'chat': 99, 'agent': 42, 'doc': 19}
   tier counts [99, 40, 18, 3] | demand shares [0.0254, 0.1214, 0.5361, 0.3171]
   The top tiers hold a few percent of the requests and most of the prefill demand.

2. Context tiers versus a single FIFO (time to first token, ms)
   tier 0: tiered p50     370.2 p99    2308.0 | fifo p50    3083.9 p99   10680.9
   tier 1: tiered p50     984.8 p99    2358.9 | fifo p50    4177.0 p99   10060.3
   tier 2: tiered p50    6721.8 p99   17801.4 | fifo p50    6865.7 p99   17133.1
   tier 3: tiered p50   32691.8 p99   45183.1 | fifo p50   14063.9 p99   20112.5
   Tiers move waiting time from the many short requests to the few long ones.

3. Expert-parallel load balancing (mean moe_lb over 60 steps, 64 experts on 8 ranks)
   no redundancy 1.992 | two redundant experts per rank 1.413
   Replicas of the hot experts bring the slowest rank down toward the mean.

4. Prefix-store hit rate against time-to-live (days)
   chatbot traffic: {1: 0.8712, 3: 0.9052, 7: 0.917}
   agent traffic:   {1: 0.875, 3: 0.875, 7: 0.875}
   The long tail of chat prefixes pays for a shorter TTL; bursty agent sessions do not.

5. Dynamic verification against fixed-5 (tokens per second per rank)
    16 seqs/rank: fixed   4668.8 | best length 4   4739.8 | gain +1.52%
    32 seqs/rank: fixed   8382.0 | best length 4   8642.7 | gain +3.11%
    60 seqs/rank: fixed  13329.3 | best length 3  14283.9 | gain +7.16%
    96 seqs/rank: fixed  17842.8 | best length 3  19942.3 | gain +11.77%
   asynchronous scheduling: serial step 16.0 ms, overlapped step 12.5 ms

6. Splitting 8 GPUs between prefill and decode
   2P6D:   456.4 tok/s | TTFT p99   47768.3 ms | TBT p99  13.35 ms | util prefill 0.85 decode 0.25 | $9.74 per million tokens
   4P4D:   762.6 tok/s | TTFT p99   25953.0 ms | TBT p99  13.57 ms | util prefill 0.73 decode 0.56 | $5.83 per million tokens
   6P2D:   799.4 tok/s | TTFT p99   25953.3 ms | TBT p99  15.01 ms | util prefill 0.51 decode 0.76 | $5.56 per million tokens
   non-dominated operating points: ['6P2D', '4P4D']
   Every point on the frontier is a different cluster; the price says what each one charges.
```
