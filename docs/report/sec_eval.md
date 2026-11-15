# Evaluation & Experimental Benchmarks

## 1. Experimental Methodology and Setup

### 1.1 Testbed Environment
All experiments were conducted on a native bare-metal workstation with hardware performance counters enabled:
- **CPU**: AMD/Intel x86_64 multi-core processor running with the Linux `performance` frequency governor.
- **Kernel**: Ubuntu 22.04 LTS booted natively on Linux `6.8.0-138-generic` with kernel BTF enabled.
- **Storage Subsystem**: NVMe Solid State Drive (`/dev/nvme0n1p6`) formatted as `ext4`.
- **Toolchain**: Clang 14.0.0, static `libbpf v1.4.0`, and `bpftool v7.4.0`.
- **Clock Synchronization**: All ground truth intervals and in-kernel telemetry strictly align to `CLOCK_MONOTONIC` in nanoseconds.

### 1.2 Evaluation Workloads
Controlled workloads were orchestrated by [`workloads/harness.c`](file:///home/priyansu/ebpf-stalls/workloads/harness.c):
1. **`CPU_SYSCALL` (Pure Compute)**: Active arithmetic spin loop executing on-CPU without issuing blocking storage system calls.
2. **`IO_WAIT` (Cold Synchronous I/O)**: Block reads using `O_DIRECT` and `posix_fadvise(POSIX_FADV_DONTNEED)` ensuring physical block traversal.
3. **`WARM_READ` (Page-Cache Resident Hit)**: Buffered reads satisfied entirely within kernel page cache without block-layer dispatches.
4. **`SCHED_WAIT` (CPU Oversubscription)**: Pinned main thread contending on core 0 with multiple pinned background CPU worker threads via tight `sched_yield` loops.
5. **`NEGATIVE_CONTROL` / Background Noise**: Tracked CPU spin loop executing concurrently with un-tracked background `fio` storage traffic to test cross-layer attribution ambiguity.

---

## 2. Quantitative Evaluation: Validating Hypotheses H1 & H2

### 2.1 Multi-Layer Fusion vs. Single-Layer Baselines & WoSC'25 Method (Testing H1)

**Hypothesis H1** posits that fusing syscall, scheduler, and block-layer evidence classifies stall types more accurately than any isolated single-layer probe.

To test H1, our fused models are benchmarked on identical runs against three canonical single-layer baselines and two variants of the correlation-based tail latency diagnosis method from Darzi et al. (WoSC'25, `B_paper`):
1. **Syscall-Duration-Only**: Classifies operations based solely on duration thresholding.
2. **Block-Metrics-Only**: Classifies based strictly on physical block requests.
3. **Scheduler-Only**: Classifies based on task off-CPU duration and runqueue waiting, blind to physical block I/O.
4. **Correlation Baseline (`B_paper`, Darzi et al. WoSC'25)**: Evaluates spike scores combined with signal alignment ($\alpha = 0.5$) with forced single-label output.
5. **Correlation Baseline w/ Abstain (`B_paper_abstain`)**: Introduces an abstention threshold ($\tau = 0.30$) tuned strictly on `data/tune` to abstain when alignment scores are weak.
6. **Fused Decision Tree (Ours)**: Trained on `data/tune/` features and evaluated on `data/final/`.
7. **Fused Heuristic Rules (Ours)**: Rule engine utilizing domain invariants frozen from tune data (`analysis/thresholds.json`).

The benchmark was executed across all 24 evaluation runs in `data/final/` ($N = 960$ operations, 40 operations per run $\times$ 3 repeat runs). 

#### Formal Scoring Rules (Defined in `docs/taxonomy.md`)
1. **Negative Control (`final_neg_ctrl`)**: Accepts either `CPU_SYSCALL` (application thread was purely on-CPU) or `INDETERMINATE` (cross-layer evidence was conflicted by background `fio` storage noise).
2. **Mixed Contention (`final_mixed_1mb`)**: Ground truth is strictly `INDETERMINATE`. Any concrete label (`CPU_SYSCALL`, `IO_WAIT`, `SCHED_WAIT`, `WARM_READ`) is an error.
3. **Pure Workloads**: Ground truth is the nominal scenario label.

95% confidence intervals were computed via cluster bootstrap resampling over independent run clusters ($B = 1000$ iterations).

| Method | Observability Layers Used | 5-Class Macro-F1 (All 960 Ops) | 95% Cluster Bootstrap CI | 4-Class Macro-F1 (All Ops) | Coverage (%) | Accuracy on Answered (%) |
|---|---|---|---|---|---|---|
| **Syscall Duration Only** | Layer 1 | 0.3473 | [0.2755, 0.3931] | 0.4341 | 100.0% | 51.46% (494/960) |
| **Block Metrics Only** | Layer 3 | 0.1979 | [0.1071, 0.2683] | 0.2474 | 100.0% | 37.81% (363/960) |
| **Scheduler Only** | Layer 2 | 0.4650 | [0.3325, 0.5416] | 0.5812 | 100.0% | 62.08% (596/960) |
| **Correlation Baseline (Darzi et al.)** | Layers 1 + 2 + 3 | 0.6297 | [0.5003, 0.7148] | 0.7871 | 100.0% | 75.31% (723/960) |
| **Correlation Baseline w/ Abstain** | Layers 1 + 2 + 3 | 0.6073 | [0.4762, 0.6974] | 0.8093 | 90.4% | 78.23% (679/868) |
| **Fused Heuristic Rules (Ours)** | **Layers 1 + 2 + 3** | **0.8562** | **[0.7001, 0.9861]** | **0.9107** | **87.8%** | **85.77% (723/843)** |
| **Fused Decision Tree (Ours)** | Layers 1 + 2 + 3 | 0.8777 | [0.6472, 0.9877] | 0.8654 | 85.5% | 86.97% (714/821) |

#### Confidence Interval Analysis & Assessment of Overlap
The 95% cluster bootstrap confidence intervals indicate that while multi-layer fusion models clearly separate from isolated single-layer probes (Scheduler Only CI [0.3325, 0.5416] does not overlap with Fused Rules [0.7001, 0.9861]), there is non-trivial confidence interval overlap between the correlation baselines (`B_paper` forced [0.5003, 0.7148], `B_paper` abstain [0.4762, 0.6974]) and the fused models (Fused Decision Tree [0.6472, 0.9877] overlaps extensively; Fused Rules [0.7001, 0.9861] touches at 0.7001). Because the bootstrap intervals overlap across clusters, the data does not support a claim of statistically significant superiority over Darzi et al. Instead, the experimental evidence demonstrates an empirical performance advantage in handling multi-bottleneck mixed contention and block-size transfer robustness.

#### Reconciliation of Prior Discrepant Metrics (0.9167 vs. 0.8051)
In preliminary draft reporting, two conflicting F1 scores were presented for Fused Heuristic Rules:
1. **The 0.9167 metric was flawed**: It was computed as a conditional Macro-F1 restricted strictly to the 723 decided instances, completely omitting the 120 mixed contention operations and discarding all 117 abstentions on the negative control. Conditioning solely on decided instances artificially inflated the metric by evaluating only "easy" unimodal cases.
2. **The 0.8051 metric was also flawed**: It was an ad-hoc 4-class average over 840 operations that penalized negative control abstentions but still completely omitted the 120 mixed contention operations.
3. **Rigorous Recomputed Metric**: Evaluating over all 960 operations with formal scoring rules yields a **5-Class Macro-F1 of 0.8562** (95% CI: [0.7001, 0.9861]), **87.8% coverage**, and **85.77% accuracy** on answered operations.

#### Unified Scoring Implementation
All metrics above were computed directly from `pred_*.jsonl` files using this unified evaluation function:
```python
def load_and_score_all(runs_dir="data/final"):
    runs = sorted(glob.glob(f"{runs_dir}/*"))
    for m_key in ["syscall_only", "block_only", "sched_only", "paper", "rules", "tree"]:
        records = []
        for r in runs:
            rname = Path(r).name
            truths = [json.loads(l) for l in open(f"{r}/truth.jsonl") if l.strip()]
            preds = [json.loads(l) for l in open(f"{r}/pred_{m_key}.jsonl") if l.strip()]
            for t, p in zip(truths, preds):
                records.append({
                    "run": rname,
                    "true": t.get("true_label", t.get("label")),
                    "pred": p.get("pred_label", p.get("pred")),
                    "is_neg_ctrl": "neg_ctrl" in rname,
                    "is_mixed": "mixed" in rname,
                })
        df = pd.DataFrame(records)

        # Ground truth evaluation semantics
        eval_true = []
        for _, row in df.iterrows():
            if row["is_neg_ctrl"]:
                eval_true.append(row["pred"] if row["pred"] in ["CPU_SYSCALL", "INDETERMINATE"] else "CPU_SYSCALL")
            elif row["is_mixed"]:
                eval_true.append("INDETERMINATE")
            else:
                eval_true.append(row["true"])
        df["eval_true"] = eval_true

        # (a) Macro-F1 over all 960 operations
        f1_5 = f1_score(df["eval_true"], df["pred"], labels=ALL_CLASSES, average="macro", zero_division=0)
        # (b) Coverage & Answered Accuracy
        answered = df[df["pred"] != "INDETERMINATE"]
        coverage = len(answered) / len(df) * 100.0
        acc_answered = accuracy_score(answered["eval_true"], answered["pred"]) if len(answered) > 0 else 0.0
```

The baseline comparison confirms Hypothesis H1: isolated single-layer probes fail to separate root causes across the full workload suite (5-class all-operations Macro-F1 of 0.1979 to 0.4650), whereas fusing syscall, scheduler, and block-layer signals achieves an all-operations Macro-F1 of 0.8562 (Fused Heuristics) and 0.8777 (Fused Tree) with 85.5% to 87.8% decision coverage. The visual comparison with 95% cluster bootstrap confidence intervals is illustrated in [`eval/plots/baseline_comparison.png`](file:///home/priyansu/ebpf-stalls/eval/plots/baseline_comparison.png).

---

### 2.2 Full Confusion Matrices and Per-Scenario Breakdown (Testing H2)

**Hypothesis H2** states that providing an explicit **`INDETERMINATE`** abstention class prevents high-confidence misclassifications when cross-layer evidence is weak, conflicted, or multi-bottlenecked.

#### Per-Scenario Evaluation Across All 960 Operations
The table below reports operation attribution for every evaluation scenario in `data/final` across all 6 methods. Note specifically the **Mixed Contention** scenario where ground truth is `INDETERMINATE`:

| Scenario | Workload | Background Noise | Ground Truth | N | Syscall Only | Block Only | Sched Only | B_paper (Darzi) | Fused Rules | Decision Tree |
|---|---|---|---|---|---|---|---|---|---|---|
| **Warm Read 1MB** | `warm` | `none` | `WARM_READ` | 120 | WARM_READ: 120 | CPU_SYSCALL: 120 | CPU_SYSCALL: 120 | WARM_READ: 120 | WARM_READ: 120 | WARM_READ: 120 |
| **Cold Direct I/O 1MB** | `io` | `none` | `IO_WAIT` | 120 | WARM_READ: 108, IO_WAIT: 12 | IO_WAIT: 120 | IO_WAIT: 119, WARM_READ: 1 | IO_WAIT: 64, CPU_SYSCALL: 56 | IO_WAIT: 120 | IO_WAIT: 114, INDETERMINATE: 6 |
| **CPU Spin 50ms** | `cpu` | `none` | `CPU_SYSCALL` | 120 | CPU_SYSCALL: 120 | CPU_SYSCALL: 114, IO_WAIT: 6 | CPU_SYSCALL: 120 | CPU_SYSCALL: 120 | CPU_SYSCALL: 114, INDETERMINATE: 6 | CPU_SYSCALL: 120 |
| **Sched Contention (4 comp)** | `sched` | `none` | `SCHED_WAIT` | 120 | IO_WAIT: 120 | CPU_SYSCALL: 63, IO_WAIT: 57 | SCHED_WAIT: 120 | SCHED_WAIT: 120 | SCHED_WAIT: 120 | SCHED_WAIT: 120 |
| **Mixed Contention (I/O + Sched)** | `mixed` | `none` | `INDETERMINATE` | 120 | IO_WAIT: 120 | IO_WAIT: 120 | SCHED_WAIT: 120 | SCHED_WAIT: 120 | SCHED_WAIT: 120 | INDETERMINATE: 120 |
| **Negative Control (CPU + fio)** | `cpu` | `fio` | `CPU_SYSCALL` | 120 | CPU_SYSCALL: 120 | IO_WAIT: 111, CPU_SYSCALL: 9 | CPU_SYSCALL: 120 | CPU_SYSCALL: 120 | INDETERMINATE: 111, CPU_SYSCALL: 9 | CPU_SYSCALL: 120 |
| **Transfer Cold I/O (4MB/256K)** | `io` | `none` | `IO_WAIT` | 120 | WARM_READ: 118, IO_WAIT: 2 | IO_WAIT: 120 | IO_WAIT: 117, WARM_READ: 3 | CPU_SYSCALL: 61, IO_WAIT: 59 | IO_WAIT: 120 | WARM_READ: 107, INDETERMINATE: 13 |
| **Transfer Warm Read (4MB/256K)** | `warm` | `none` | `WARM_READ` | 120 | WARM_READ: 120 | CPU_SYSCALL: 120 | CPU_SYSCALL: 120 | WARM_READ: 120 | WARM_READ: 120 | WARM_READ: 120 |

---

#### 5x5 Confusion Matrices Over All 960 Operations (Including `INDETERMINATE` Ground Truth)

##### 1. Syscall Duration Only ($N = 960$)
| True Class \ Predicted | `Pred_CPU_SYSCALL` | `Pred_IO_WAIT` | `Pred_SCHED_WAIT` | `Pred_WARM_READ` | `Pred_INDETERMINATE` |
|---|---|---|---|---|---|
| **`True_CPU_SYSCALL`** | 240 | 0 | 0 | 0 | 0 |
| **`True_IO_WAIT`** | 0 | 14 | 0 | 226 | 0 |
| **`True_SCHED_WAIT`** | 0 | 120 | 0 | 0 | 0 |
| **`True_WARM_READ`** | 0 | 0 | 0 | 240 | 0 |
| **`True_INDETERMINATE`** | 0 | 120 | 0 | 0 | 0 |

##### 2. Block Metrics Only ($N = 960$)
| True Class \ Predicted | `Pred_CPU_SYSCALL` | `Pred_IO_WAIT` | `Pred_SCHED_WAIT` | `Pred_WARM_READ` | `Pred_INDETERMINATE` |
|---|---|---|---|---|---|
| **`True_CPU_SYSCALL`** | 123 | 117 | 0 | 0 | 0 |
| **`True_IO_WAIT`** | 0 | 240 | 0 | 0 | 0 |
| **`True_SCHED_WAIT`** | 63 | 57 | 0 | 0 | 0 |
| **`True_WARM_READ`** | 240 | 0 | 0 | 0 | 0 |
| **`True_INDETERMINATE`** | 0 | 120 | 0 | 0 | 0 |

##### 3. Scheduler Only ($N = 960$)
| True Class \ Predicted | `Pred_CPU_SYSCALL` | `Pred_IO_WAIT` | `Pred_SCHED_WAIT` | `Pred_WARM_READ` | `Pred_INDETERMINATE` |
|---|---|---|---|---|---|
| **`True_CPU_SYSCALL`** | 240 | 0 | 0 | 0 | 0 |
| **`True_IO_WAIT`** | 0 | 236 | 0 | 4 | 0 |
| **`True_SCHED_WAIT`** | 0 | 0 | 120 | 0 | 0 |
| **`True_WARM_READ`** | 240 | 0 | 0 | 0 | 0 |
| **`True_INDETERMINATE`** | 0 | 0 | 120 | 0 | 0 |

##### 4. Correlation Baseline (Darzi et al. WoSC'25, $N = 960$)
| True Class \ Predicted | `Pred_CPU_SYSCALL` | `Pred_IO_WAIT` | `Pred_SCHED_WAIT` | `Pred_WARM_READ` | `Pred_INDETERMINATE` |
|---|---|---|---|---|---|
| **`True_CPU_SYSCALL`** | 240 | 0 | 0 | 0 | 0 |
| **`True_IO_WAIT`** | 117 | 123 | 0 | 0 | 0 |
| **`True_SCHED_WAIT`** | 0 | 0 | 120 | 0 | 0 |
| **`True_WARM_READ`** | 0 | 0 | 0 | 240 | 0 |
| **`True_INDETERMINATE`** | 0 | 0 | 120 | 0 | 0 |

##### 5. Fused Heuristic Rules (Ours, $N = 960$)
| True Class \ Predicted | `Pred_CPU_SYSCALL` | `Pred_IO_WAIT` | `Pred_SCHED_WAIT` | `Pred_WARM_READ` | `Pred_INDETERMINATE` |
|---|---|---|---|---|---|
| **`True_CPU_SYSCALL`** | 123 | 0 | 0 | 0 | 117 |
| **`True_IO_WAIT`** | 0 | 240 | 0 | 0 | 0 |
| **`True_SCHED_WAIT`** | 0 | 0 | 120 | 0 | 0 |
| **`True_WARM_READ`** | 0 | 0 | 0 | 240 | 0 |
| **`True_INDETERMINATE`** | 0 | 0 | 120 | 0 | 0 |

##### 6. Fused Decision Tree (Ours, $N = 960$)
| True Class \ Predicted | `Pred_CPU_SYSCALL` | `Pred_IO_WAIT` | `Pred_SCHED_WAIT` | `Pred_WARM_READ` | `Pred_INDETERMINATE` |
|---|---|---|---|---|---|
| **`True_CPU_SYSCALL`** | 240 | 0 | 0 | 0 | 0 |
| **`True_IO_WAIT`** | 0 | 114 | 0 | 107 | 19 |
| **`True_SCHED_WAIT`** | 0 | 0 | 120 | 0 | 0 |
| **`True_WARM_READ`** | 0 | 0 | 0 | 240 | 0 |
| **`True_INDETERMINATE`** | 0 | 0 | 0 | 0 | 120 |

#### Empirical Finding on Negative Control and Ambiguous Telemetry
In the negative control workload (`final_neg_ctrl_r1`, `r2`, `r3`), the tracked application process executes pure CPU arithmetic (`on_cpu_ratio >= 0.80`) while an asynchronous background `fio` process generates global storage traffic (`block_issues > 0`).

The experimental results across models reveal distinct behaviors under background noise:
1. **Naive Block Counter Failure**: The isolated block-metrics baseline observed active block requests and misclassified 111 out of 120 negative control operations (92.5%) as `IO_WAIT`, demonstrating that counting physical block requests without verifying thread sleep state produces severe attribution errors.
2. **`B_paper` Behavior on Negative Control**: As shown in its confusion matrix, `B_paper` (Darzi et al.) correctly predicted `CPU_SYSCALL` on all 120 negative control operations (yielding 240/240 total on `True_CPU_SYSCALL` with 0 false `IO_WAIT` attributions). Because its CPU alignment proxy relies on `on_cpu_ratio` ($\approx 1.0$) while the application thread exhibited zero D-state sleep ($\text{io\_sleep\_ratio} = 0.0$), the compute score $s_{cpu}$ dominated $s_{io}$ by a wide margin ($s_{cpu} \approx s_{io} + 0.50$), preventing false storage attribution.
3. **Fused Heuristic Rules (Conflict Rule 2)**: The rule engine activated Conflict Rule 2 (`abstain_bg_noise_on_cpu_min = 0.80`), detecting that high on-CPU utilization directly conflicted with the presence of un-attributed block requests. The model abstained on 111 out of 120 operations (92.5%), classifying them as `INDETERMINATE`, while predicting `CPU_SYSCALL` on the remaining 9. Under our formal scoring semantics (which accept either `CPU_SYSCALL` or `INDETERMINATE`), both `B_paper` and Fused Rules correctly handle the negative control.
4. **Where `B_paper` Actually Fails**: As shown in the per-scenario table and confusion matrices, `B_paper`'s attribution errors occur in two distinct scenarios:
   - **Multi-Bottleneck Mixed Contention**: In `final_mixed_1mb`, `B_paper` forced a single label (`SCHED_WAIT` on all 120 operations), failing to recognize that concurrent direct I/O and CPU contention cannot be attributed to a single cause.
   - **Cold Direct I/O Misclassification**: In both 1 MB and 4 MB cold I/O runs (`final_io_1mb` and `final_io_4mb_256k`), `B_paper` misclassified 117 out of 240 storage operations (48.8%) as `CPU_SYSCALL` because operations whose durations did not produce a 3-sigma spike above the run mean were overwhelmed by the baseline on-CPU term.

---

## 3. Ablation Studies

### 3.1 Layer Ablation
To quantify each telemetry layer's marginal contribution, feature subsets were systematically ablated on `data/final/` ($N = 960$ operations):

| Ablation Configuration | Retained Features | Macro-F1 (Decided) | `IO_WAIT` Recall | `SCHED_WAIT` Recall | Abstention Rate (%) | Decided Samples |
|---|---|---|---|---|---|---|
| **Full Fused Model** | All features across 3 layers | 0.9167 | 1.0000 | 1.0000 | 12.2% | 843 |
| **No Block Layer** | Syscall + Scheduler only | 0.5952 | 0.0000 | 1.0000 | 5.0% | 912 |
| **No Scheduler Layer** | Syscall + Block only | 0.3892 | 0.2000 | 0.0000 | 6.6% | 897 |
| **Syscall Boundary Only**| `duration_ms`, `sys_enters` | 0.4740 | 0.2000 | 0.0000 | 0.0% | 960 |

#### Directional Movement Analysis
1. **Collapsing `IO_WAIT` Recall Without Block Telemetry**:
   Removing the block layer (`block_rq_issue`, `block_rq_complete`, `block_bio_queue`) causes `IO_WAIT` recall to collapse from 1.0000 to **0.0000**. Because the scheduler layer only reports off-CPU D-state sleep duration without confirming whether physical I/O requests were dispatched or completed, the classifier cannot verify storage stalls, leading to 100% false negatives on cold direct I/O.
2. **Collapsing `SCHED_WAIT` Recall Without Scheduler Telemetry**:
   Removing the scheduler layer (`sched_switch`, `sched_wakeup`) collapses `SCHED_WAIT` recall from 1.0000 to **0.0000**. Without direct measurement of runnable wait time (`sched_wait_ratio`), CPU oversubscription and preemption delays become completely invisible.
3. **Insufficiency of Syscall Duration**:
   Boundary-only instrumentation achieves a Macro-F1 of only 0.4740, demonstrating that wall-clock stall duration alone cannot differentiate compute loops, runqueue delays, and blocking disk operations. The visual ablation study is saved to [`eval/plots/ablation_study.png`](file:///home/priyansu/ebpf-stalls/eval/plots/ablation_study.png).

---

### 3.2 Transfer Test: Robustness Across Transfer Sizes (4 MB / 256 KB vs. Matched 1 MB)
To evaluate model generalization beyond training conditions, `data/final/` included transfer test scenarios (`final_io_4mb_r1..r3` and `final_warm_4mb_r1..r3`) operating on 4 MB files with 256 KB block sizes, compared against the matched 1 MB file / 1 MB block configurations present in `data/tune/`:

| Workload Split | Model | Decided Samples | Coverage (%) | Accuracy (%) | Tested Classes Macro-F1 | Overall 4-Class Macro-F1 |
|---|---|---|---|---|---|---|
| **Matched Configs (1 MB / 1 MB)** | Fused Heuristic Rules | 360 / 360 | 100.0% | 100.0% | 1.0000 | 0.5000 |
| **Matched Configs (1 MB / 1 MB)** | Fused Decision Tree | 234 / 360 | 65.0% | 100.0% | 1.0000 | 0.5000 |
| **Transfer Test (4 MB / 256 KB)** | **Fused Heuristic Rules** | **240 / 240** | **100.0%** | **100.0%** | **1.0000** | **0.5000** |
| **Transfer Test (4 MB / 256 KB)** | Fused Decision Tree | 227 / 240 | 94.6% | 52.9% | 0.3458 | 0.1729 |

#### Transfer Test Findings
The transfer test reveals a fundamental distinction between purely data-driven ML models and domain-informed physical invariants:
- **Decision Tree Overfitting**: The shallow decision tree trained on 1 MB tune data learned a split on raw request count (`block_issues > 4.0`). When tested on 256 KB block sizes where direct I/O chunking issued fewer requests per operation, the tree misclassified 107 out of 120 cold direct I/O operations (89.2%) as `WARM_READ`, dropping its accuracy to 52.9% and tested-class Macro-F1 to 0.3458.
- **Physical Invariant Generalization**: In contrast, the Fused Heuristics engine classifies storage stalls using the ratio of off-CPU D-state sleep (`io_sleep_ratio >= 0.18`) coupled with non-zero block request issues (`block_issues > 0`). Because physical sleep ratios remain invariant across block sizes, the rule engine achieved 100.0% accuracy and 1.0000 Macro-F1 across both matched and transfer test workloads.

---

## 4. Instrumentation Overhead & Safety

### 4.1 Probe Latency Measurement Limitations
In earlier benchmark iterations, uninstrumented baseline trials discarded operation ground truth logs to `/dev/null`. Measuring overhead by comparing total process wrapper wall-clock execution time conflates the one-time user-space BPF initialization, ELF verification, and ring-buffer setup of the `collector` binary (~350–370 ms) with actual in-kernel probe execution overhead. 

Because per-trial untraced operation timestamps (`truth.jsonl`) are not preserved for uninstrumented trials, in-kernel probe execution latency cannot be reliably measured from outer process execution times. In accordance with strict experimental standards, probe latency overhead is not reliably measured and no numerical latency penalty is reported.

### 4.2 System Safety and Lossless Telemetry
1. **Zero Ring-Buffer Event Loss**:
   Events are emitted as 72-byte fixed-size structs via `bpf_ringbuf_reserve` and `bpf_ringbuf_submit`. Across all 24 evaluation matrix runs in `data/final/` ($N = 960$ operations) and 10 alternating trial executions, the 64 MB ring buffer sustained **0 dropped events (0.00% loss rate)**.
2. **Early In-Kernel PID Filtering**:
   All attached tracepoints query `bpf_get_current_pid_tgid()` at the entry instruction and return immediately if the TGID does not match the monitored target process, minimizing disturbance to concurrent host workloads.

---

## 5. Threats to Validity, Portability Boundary & Limitations

### 5.1 Threats to Validity
1. **Decoupled Block Dispatch (`kworker` & Interrupt Context)**:
   In Linux, asynchronous writeback and block completions occur in kernel worker threads or hardware interrupt / softirq contexts. Application TIDs often do not appear on raw `block_rq_issue` tracepoints. Our system resolves this using the in-kernel `bio_submitters` map, capturing the submitter TID at `block_bio_queue` and propagating it to block request completions.
2. **Page Cache Decoupling**:
   Warm reads satisfied from page cache do not trigger block-layer tracepoints. While physically accurate (they do not represent disk storage stalls), memory reclamation under heavy host memory pressure can introduce unexpected allocation latency.
3. **Storage Virtualization**:
   In virtualized environments (e.g., KVM/virtio), block completions may reflect host memory caching rather than physical NAND flash write cycles. All evaluation benchmarks were conducted on native bare-metal NVMe storage to preserve physical realism.

### 5.2 Portability Boundary
The tracer was developed and evaluated natively on Ubuntu 22.04 LTS running Linux kernel `6.8.0-138-generic` (x86_64) with Clang 14.0.0 and libbpf 1.4.0. The core tracepoints (`raw_syscalls`, `sched_switch`, `sched_wakeup`, `block_rq_issue`, `block_rq_complete`, `block_bio_queue`, `softirq_entry`, `softirq_exit`) are standard across Linux v5.8+, but rely on BTF (`vmlinux.h`). Kernels lacking `CONFIG_DEBUG_INFO_BTF` or non-x86 architectures require recompiling BPF skeletons with appropriate architecture headers.

### 5.3 Limitations & Documented Scope Cuts

While the evaluation demonstrates the efficacy of multi-layer eBPF fusion, several fundamental limitations characterize the experimental design and model architectures:

#### 1. Synthetic and Unimodal Benchmark Scenarios ("Easy Scenarios")
The benchmark suite evaluated in this study consists of controlled, synthetically isolated micro-workloads where root-cause bottlenecks are predominantly unimodal:
- Pure on-CPU arithmetic loops (`CPU_SYSCALL`),
- Clean synchronous direct NVMe reads (`IO_WAIT`),
- Pinned runqueue preemption loops (`SCHED_WAIT`),
- In-memory cache hits (`WARM_READ`),
- An explicitly constructed bivariate contention case (`INDETERMINATE`).

In production enterprise systems, application stalls are substantially more complex and "messy." Real-world stalls frequently stem from cascading multi-tier dependencies, including user-space pthread mutex and futex lock contention, kernel memory compaction during page allocation storms, asynchronous network RPC timeouts, socket buffer saturation, and distributed backpressure. Such compound failure modes present subtle cross-layer signatures that are significantly more difficult to diagnose than the clean, non-overlapping bottlenecks evaluated here.

#### 2. Abstention Heuristics Designed Around the Negative Control
The Fused Heuristics classifier's abstention mechanism incorporates an explicit rule—Conflict Rule 2 in `analysis/rules.py`—which forces an `INDETERMINATE` prediction whenever an operation exhibits high on-CPU execution (`on_cpu_ratio >= 0.80`) while simultaneously recording non-zero block request issues (`block_issues > 0`).

While this rule is physically grounded (a thread executing active computation on a CPU core cannot simultaneously be blocked in synchronous storage I/O, indicating that recorded block events belong to background noise rather than the thread's stall), this specific conflict invariant was deliberately informed by and engineered around the failure mode observed in the Negative Control scenario (a monitored CPU spin loop executing concurrently with un-tracked background `fio` disk traffic). Designing heuristic conflict rules to handle anticipated experimental conditions risks overfitting the domain logic to known benchmark artifacts. If a real-world application were legitimately performing asynchronous overlapped computation alongside non-blocking I/O polling, this rule would aggressively abstain rather than providing a nuanced breakdown.

#### 3. Workload Overlap Between Tune and Final Splits (Limited Transfer Diversity)
The factorial benchmark configurations evaluated in the `final` test split (`final_warm_1mb`, `final_io_1mb`, `final_cpu_50ms`, `final_sched_4comp`, `final_mixed_1mb`, `final_neg_ctrl`) shared the identical C harness generators (`workloads/harness.c`), operational semantics, and background stress models as the `tune` split.

The only dimension varied to evaluate out-of-distribution transfer was the storage transfer configuration: 4 MB file reads issued in 256 KB blocks (`final_io_4mb`, `final_warm_4mb`) versus the matched 1 MB file / 1 MB block configurations present in `tune`. While this size transfer successfully exposed the vulnerability of raw feature counts in data-driven decision trees (causing the tree's test-class Macro-F1 to plunge from 1.0000 to 0.3458), the evaluation did not assess model transfer across:
- Entirely distinct application binaries (e.g., PostgreSQL, Redis, RocksDB),
- Variable filesystem layouts (e.g., XFS, Btrfs, ZFS vs. ext4),
- Mixed concurrent read/write ratios, or
- Non-synchronous I/O mechanisms (e.g., `io_uring`, POSIX AIO, `epoll`).

Consequently, high performance on `data/final/` reflects robustness across request chunk sizes within a known generator family, rather than arbitrary cross-workload generality.

#### 4. Documented Scope Cuts
In accordance with course assignment boundaries and project scope agreements, five planned technical capabilities were explicitly cut and are documented as limitations:
1. **Multi-Threaded Application Concurrency (Scope Cut 1)**:
   All monitored benchmarks were executed with a single tracked application thread (`concurrency = 1`). Cross-thread lock contention, futex sleep tracking, thread-pool worker handoffs, and attribution among multiple concurrent threads within the same process thread group were not evaluated.
2. **Buffered Read and Asynchronous Writeback Scenarios (Scope Cut 2)**:
   Attribution under asynchronous readahead, delayed allocation, and background page-cache writeback (`fsync`, `fdatasync`, `sync_file_range`) was not evaluated. Benchmark storage scenarios were intentionally restricted to synchronous cold direct I/O (`O_DIRECT` + `POSIX_FADV_DONTNEED`) and memory-resident page cache hits (`WARM_READ`).
3. **Window-Margin Temporal Sweeping (Scope Cut 3)**:
   Systematic sweeping of loose temporal window margins ($\pm 1\text{ ms}$ to $\pm 5\text{ s}$) was not conducted across the final benchmark matrix. Operations were evaluated within ground-truth synchronized syscall intervals ($[t_{start}, t_{end}]$). In production without application-level boundaries, loose time windows capture extraneous concurrent events and degrade precision.
4. **Pseudo-Random Run Seeds (Scope Cut 4)**:
   Experimental variability was evaluated across factorial repeat runs (`_r1`, `_r2`, `_r3`) rather than varying pseudo-random generator seeds. Stochastic variation was driven by system-level scheduling and background noise rather than synthetic PRNG seed sequences.
5. **Secondary Hardware and Kernel Architecture Setup (Scope Cut 5)**:
   Evaluation was conducted on a single native bare-metal workstation running Linux kernel `6.8.0-138-generic` (x86_64) on an NVMe SSD. Cross-platform validation across alternative kernel versions (e.g., Linux 5.15, 6.1 LTS), non-x86 architectures (e.g., ARM64), or secondary storage controller hardware (e.g., SATA SSD, HDD, networked iSCSI/NFS) was not performed.
