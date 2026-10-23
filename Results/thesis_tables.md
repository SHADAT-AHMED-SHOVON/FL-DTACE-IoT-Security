# Tables | tag=Big seed=42 w1=0.6 w2=0.4 | P_th=0.4760 T_th=0.7745 | rows=2,219,201 | holdout=()

## Detection (streaming final decisions)

| accuracy | precision | recall | f1 | fpr | balanced_acc | mcc | precision_at_50pct_prevalence |
|---|---|---|---|---|---|---|---|
| 0.9596 | 0.9026 | 0.9545 | 0.9278 | 0.0385 | 0.9580 | 0.9005 | 0.9612 |

## Ablation (full test set)

| variant | auc | tpr@fpr1% | tpr@fpr2% | tpr@fpr5% |
|---|---|---|---|---|
| Classifier only (FL int8) | 0.9931 | 0.9300 | 0.9383 | 0.9596 |
| B_norm only (standard ZTA rules) | 0.7319 | 0.2334 | 0.2563 | 0.3250 |
| FL-DTACE: classifier + B_norm (proposed) | 0.9928 | 0.9303 | 0.9387 | 0.9597 |
| NEG. RESULT: + per-device trust | 0.9436 | 0.5698 | 0.6093 | 0.6925 |
| NEG. RESULT control: shuffled device ids | 0.7849 | 0.1333 | 0.1805 | 0.2876 |

## FL vs centralised vs local-only

| method | auc | tpr@fpr1% | tpr@fpr2% | tpr@fpr5% | worst | best |
|---|---|---|---|---|---|---|
| Centralised (no privacy) | 0.9942 | 0.9414 | 0.9490 | 0.9612 | nan | nan |
| FedAvg float32 | 0.9929 | 0.9376 | 0.9435 | 0.9564 | nan | nan |
| FL-DTACE int8 | 0.9931 | 0.9300 | 0.9383 | 0.9596 | nan | nan |
| Local-only (mean of sub-stations) | 0.9920 | nan | nan | nan | 0.9908 | 0.9927 |

## Latency (ms)

| System | mean_ms | p50_ms | p95_ms | p99_ms | pct_under_10ms |
|---|---|---|---|---|---|
| FL-DTACE (fast path) | 4.5847 | 4.5463 | 7.3073 | 8.5455 | 99.9683 |
| FL-DTACE (no fast path) | 4.7811 | 4.5924 | 8.0957 | 11.9397 | 97.5892 |
| FL-DTACE (feedback off) | 4.5560 | 4.5186 | 7.2536 | 8.5097 | 99.9800 |
| Cloud ZTA (simulated) | 51.5841 | 50.1266 | 73.3830 | 86.5086 | 0.0000 |

## Feedback loop ON/OFF

| Feedback loop | recall | fpr | mcc | balanced_acc | mean_latency_ms |
|---|---|---|---|---|---|
| ON | 0.9545 | 0.0385 | 0.9005 | 0.9580 | 4.5847 |
| OFF | 0.9542 | 0.0383 | 0.9007 | 0.9580 | 4.5560 |

## Communication cost

| update | uplink_bytes |
|---|---|
| int8 dW | 224550 |
| float32 dW | 883800 |

## Per-attack detection

| atype | packets | detection_rate | mean_latency_ms |
|---|---|---|---|
| Backdoor | 7473 | 0.9451 | 3.4059 |
| DDoS_HTTP | 14813 | 0.8919 | 3.3751 |
| DDoS_ICMP | 35220 | 1.0000 | 3.0204 |
| DDoS_TCP | 15023 | 0.9117 | 3.3021 |
| DDoS_UDP | 36360 | 1.0000 | 3.0335 |
| Fingerprinting | 325 | 0.8031 | 3.6537 |
| MITM | 373 | 0.9812 | 4.3286 |
| Normal | 484693 | 0.0385 | 5.0972 |
| Password | 15083 | 0.8731 | 3.4042 |
| Port_Scanning | 6776 | 0.9594 | 3.2440 |
| Ransomware | 3278 | 0.9954 | 3.2535 |
| SQL_injection | 15468 | 0.9999 | 3.1145 |
| Uploading | 11202 | 0.8625 | 3.5227 |
| Vulnerability_scanner | 15015 | 0.9659 | 3.2095 |
| XSS | 4659 | 0.8789 | 3.7376 |

## w1/w2 sensitivity

| w1 | w2 | AUC (classifier + B_norm) |
|---|---|---|
| 0.0000 | 1.0000 | 0.7319 |
| 0.1000 | 0.9000 | 0.8968 |
| 0.2000 | 0.8000 | 0.9797 |
| 0.3000 | 0.7000 | 0.9847 |
| 0.4000 | 0.6000 | 0.9884 |
| 0.5000 | 0.5000 | 0.9912 |
| 0.6000 | 0.4000 | 0.9928 |
| 0.7000 | 0.3000 | 0.9931 |
| 0.8000 | 0.2000 | 0.9931 |
| 0.9000 | 0.1000 | 0.9931 |
| 1.0000 | 0.0000 | 0.9931 |

