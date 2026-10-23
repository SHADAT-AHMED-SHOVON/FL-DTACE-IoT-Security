# %%
# =====================================================================
# CELL 1 : SETUP + CONFIG (the ONLY cell you normally edit)
# =====================================================================
import os, sys, glob, time, json, math, subprocess, warnings, random
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, matthews_corrcoef
warnings.filterwarnings("ignore"); sns.set_theme(style="whitegrid", context="talk")
torch.set_num_threads(1)

@dataclass
class Config:
    # ---------------- run identity ----------------
    RUN_TAG: str = "big"               
    OUT_ROOT: str = "C:/Users/shada/Desktop/test"                  
    SEED: int = 42
    # ---------------- data ----------------
    USE_SYNTHETIC: bool = False
    CSV_PATH: str = "D:/SHOVON/Academic/AIUB/@11th Semester (Summer 25-26)/Thesis/Main Paper/Code/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"               
    CSV_NAME: str = "ML-EdgeIIoT-dataset.csv"
    USE_GPU: bool = True
    MAX_ROWS: Optional[int] = None
    TEST_SIZE: float = 0.30
    VAL_SIZE: float = 0.15               
    STREAM_N: Optional[int] = None
    KEEP_PORTS: bool = True
    INCLUDE_BEH_IN_X: bool = False       
    HOLDOUT_ATTACKS: Tuple[str, ...] = ()  
    # ---------------- Phase 1: micro-edge federated TinyML classifier ----------------
    N_SUBSTATIONS: int = 5
    NON_IID: bool = True               
    NON_IID_ALPHA: float = 1.0          
    FL_ROUNDS: int = 30
    LOCAL_EPOCHS: int = 2
    BATCH_SIZE: int = 256
    LR: float = 3e-3
    HIDDEN: Tuple[int, int] = (32, 16)  
    QUANT_BITS: int = 8
    P_TH: Optional[float] = None       
    P_TH_BENIGN_PCTL: float = 99.0      
    # ---------------- Phase 2: DTACE ----------------
    W1: float = 0.6
    W2: float = 0.4                      # W1 + W2 = 1
    T_TH: Optional[float] = None
    T_TH_BENIGN_PCTL: float = 3.0
    ALPHA: float = 0.1               
    ZCAP: float = 3.0
    SD_FLOOR: float = 0.05
    # ---------------- Phase 3: latency model (ms) ----------------
    LATENCY_BUDGET_MS: float = 10.0
    FASTPATH_MARGIN: float = 0.9
    EDGE_SLOWDOWN: float = 3.0
    AIR_MS: Tuple[float, float] = (1.0, 5.0)
    UPLINK_MS: Tuple[float, float] = (0.3, 1.5)
    QUEUE_MEAN_MS: float = 0.8
    BURST_P: float = 0.05
    BURST_MS: Tuple[float, float] = (3.0, 8.0)
    CACHE_LOOKUP_MS: float = 0.02
    DC_STORE_MS: Tuple[float, float] = (0.1, 0.4)
    CLOUD_WAN_MEDIAN_MS: float = 22.0
    CLOUD_WAN_SIGMA: float = 0.35
    CLOUD_PDP_MS: Tuple[float, float] = (0.5, 1.5)
    # ---------------- Phase 4: feedback loop ----------------
    SIG_K: int = 5
    SIG_BINS: int = 20
    RULE_MIN_SIGHTINGS: int = 5
    MAX_RULES: int = 2000
    BCAST_HOP_MS: float = 1.0
    N_PRINT: int = 8

CFG = Config()
assert abs(CFG.W1 + CFG.W2 - 1.0) < 1e-9, "W1 + W2 must equal 1"
if CFG.OUT_ROOT.startswith("/content/drive"):
    from google.colab import drive; drive.mount("/content/drive")
OUT_DIR = os.path.join(CFG.OUT_ROOT or ".", f"results_{CFG.RUN_TAG}_s{CFG.SEED}_w{CFG.W1:.2f}")
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
DEV = torch.device("cuda" if (CFG.USE_GPU and torch.cuda.is_available()) else "cpu")

class Tee:
    def __init__(self, path): self.f, self.o = open(path, "w", encoding="utf-8"), sys.__stdout__
    def write(self, t): self.o.write(t); self.f.write(t)
    def flush(self): self.o.flush(); self.f.flush()
sys.stdout = Tee(f"{OUT_DIR}/console_log.txt")
print(f"Run: tag={CFG.RUN_TAG} seed={CFG.SEED} w1={CFG.W1} w2={CFG.W2} holdout={CFG.HOLDOUT_ATTACKS} | output -> {OUT_DIR}")
print("Training/scoring device:", DEV, "|", torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU only")
print("NOTE: per-packet latency streaming always runs on single-thread CPU (models an edge device).")

# %%
# =====================================================================
# CELL 2 : DATA PREPROCESSING -> x_i (telemetry), B_norm,i (behaviour), labels
# =====================================================================
LABEL_COLS = ["Attack_label", "Attack_type"]
ID_COL = "ip.src_host"
DROP_COLS = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
             "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", "http.request.uri.query",
             "tcp.options", "tcp.payload", "tcp.srcport", "mqtt.msg", "dns.qry.name", "http.referer"]
if not CFG.KEEP_PORTS: DROP_COLS += ["tcp.dstport", "udp.port"]
BEH_CANDIDATES = ["tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn", "tcp.connection.synack",
                  "tcp.flags", "tcp.flags.ack", "udp.time_delta", "dns.retransmission",
                  "dns.retransmit_request", "icmp.seq_le"]

def _to_numeric(s):
    if pd.api.types.is_numeric_dtype(s): return s.astype(float)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9: return num
    hx = s.astype(str).map(lambda v: int(v, 16) if v.lower().startswith("0x") else np.nan)
    if hx.notna().mean() > 0.9: return hx.astype(float)
    return pd.Series(pd.factorize(s.astype(str))[0], index=s.index).astype(float)

def find_csv():
    if CFG.CSV_PATH.startswith("/content/drive") and not os.path.exists(CFG.CSV_PATH):
        from google.colab import drive; drive.mount("/content/drive")
    if CFG.CSV_PATH:
        if not os.path.exists(CFG.CSV_PATH): raise FileNotFoundError(f"CSV_PATH not found: {CFG.CSV_PATH}")
        return CFG.CSV_PATH
    try: import kagglehub
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"]); import kagglehub
    root = kagglehub.dataset_download("mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot")
    hits = glob.glob(os.path.join(root, "**", CFG.CSV_NAME), recursive=True)
    if not hits: raise FileNotFoundError("CSV not found; set CFG.CSV_PATH manually.")
    return hits[0]

def read_csv_sampled(path):
    with open(path, "rb") as f: n_lines = sum(1 for _ in f) - 1
    frac = 1.0 if CFG.MAX_ROWS is None else min(1.0, CFG.MAX_ROWS / n_lines)
    print(f"CSV rows in file: {n_lines:,} | sampling fraction: {frac:.2%}")
    parts = [ch if frac >= 1.0 else ch.sample(frac=frac, random_state=CFG.SEED)
             for ch in pd.read_csv(path, low_memory=False, chunksize=200_000)]
    return pd.concat(parts, ignore_index=True)

def make_synthetic(n=60_000, seed=0):
    rng = np.random.default_rng(seed)
    types = ["Normal", "DDoS_UDP", "DDoS_TCP_SYN", "Port_Scanning", "SQL_injection", "Backdoor"]
    at = rng.choice(types, size=n, p=[.3, .14, .14, .14, .14, .14]); d = 24
    X = rng.normal(0.5, 0.17, (n, d))
    shifts = {"DDoS_UDP": [0, 1, 2, 3], "DDoS_TCP_SYN": [4, 5, 6], "Port_Scanning": [7, 8, 9],
              "SQL_injection": [10, 11], "Backdoor": [12, 13]}
    for t, dims in shifts.items():
        m = at == t; X[np.ix_(m, dims)] += rng.normal(0.45, 0.2, (m.sum(), len(dims)))
    df = pd.DataFrame(X, columns=[f"tel_{i:02d}" for i in range(d)])
    syn = (rng.random(n) < .10).astype(float); synack = (rng.random(n) < .10).astype(float)
    rst = (rng.random(n) < .02).astype(float); fin = (rng.random(n) < .05).astype(float)
    ack = (rng.random(n) < .80).astype(float); td = rng.lognormal(-3, .6, n)
    m = at == "DDoS_TCP_SYN"; syn[m] = 1; synack[m] = 0; ack[m] = 0
    m = at == "Port_Scanning"; syn[m] = (rng.random(m.sum()) < .9); rst[m] = (rng.random(m.sum()) < .6)
    m = at == "DDoS_UDP"; td[m] = rng.lognormal(-7, .5, m.sum())
    df["tcp.connection.syn"], df["tcp.connection.synack"], df["tcp.connection.rst"] = syn, synack, rst
    df["tcp.connection.fin"], df["tcp.flags.ack"], df["udp.time_delta"] = fin, ack, td
    df[ID_COL] = ["10.0.0.%d" % v for v in rng.integers(0, 60, n)]
    df["Attack_type"] = at; df["Attack_label"] = (at != "Normal").astype(int)
    return df

def load_and_map():
    df = make_synthetic() if CFG.USE_SYNTHETIC else read_csv_sampled(find_csv())
    df = df.sample(frac=1, random_state=CFG.SEED).reset_index(drop=True)
    if CFG.MAX_ROWS is not None: df = df.iloc[:CFG.MAX_ROWS]
    y = df["Attack_label"].astype(int).values; atype = df["Attack_type"].astype(str).values
    dev = pd.factorize(df[ID_COL].astype(str))[0] if ID_COL in df else np.random.randint(0, 50, len(df))
    feat = df.drop(columns=[c for c in LABEL_COLS + DROP_COLS if c in df.columns]).apply(_to_numeric)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat = feat.loc[:, feat.nunique() > 1]
    beh = [c for c in BEH_CANDIDATES if c in feat.columns]
    tel = [c for c in feat.columns if (c not in beh) or CFG.INCLUDE_BEH_IN_X]
    print(f"Rows={len(df):,} | attack ratio={y.mean():.2%} | x_i dims={len(tel)} | B_norm dims={len(beh)}")
    print("  attack types:", dict(pd.Series(atype).value_counts()))
    print("  B_norm,i columns :", beh); print("  x_i columns:", tel)
    return dict(X=feat[tel].values.astype(np.float64), B=feat[beh].values.astype(np.float64),
                dev=dev, y=y, atype=atype)

def split_and_scale(D):
    idx = np.arange(len(D["y"]))
    tr, te = train_test_split(idx, test_size=CFG.TEST_SIZE, stratify=D["y"], random_state=CFG.SEED)
    tr, va = train_test_split(tr, test_size=CFG.VAL_SIZE, stratify=D["y"][tr], random_state=CFG.SEED)
    if CFG.HOLDOUT_ATTACKS:                                           # unseen-attack experiment
        ho = np.isin(D["atype"], CFG.HOLDOUT_ATTACKS); tr, va = tr[~ho[tr]], va[~ho[va]]
        print(f"HOLD-OUT (never seen in training/validation): {CFG.HOLDOUT_ATTACKS}")
    sx = MinMaxScaler().fit(D["X"][tr]); ben = tr[D["y"][tr] == 0]; sb = MinMaxScaler().fit(D["B"][ben])
    sc = lambda a: np.clip(sx.transform(a), 0.0, 1.0).astype(np.float32)      # clip: outliers cannot explode the net
    S = dict(Xtr=sc(D["X"][tr]), ytr=D["y"][tr], atr=D["atype"][tr], Xval=sc(D["X"][va]), yval=D["y"][va],
             Xte=sc(D["X"][te]), yte=D["y"][te], atte=D["atype"][te], devte=D["dev"][te],
             Bval=sb.transform(D["B"][va]), Bte=sb.transform(D["B"][te]))
    Xb = S["Xtr"][S["ytr"] == 0]; S["MU_X"] = Xb.mean(0); S["SD_X"] = np.maximum(Xb.std(0), CFG.SD_FLOOR)
    Bb = sb.transform(D["B"][ben]); S["MU_B"] = Bb.mean(0); S["SD_B"] = np.maximum(Bb.std(0), CFG.SD_FLOOR)
    print(f"Train={len(tr):,} (benign {int((S['ytr']==0).sum()):,}) | Val={len(va):,} | Test={len(te):,} (attack share {S['yte'].mean():.1%})")
    return S

def behaviour_norm(B_scaled, MU_B, SD_B):
    z = np.abs((B_scaled - MU_B) / SD_B)
    return 1.0 - np.clip(z, 0, CFG.ZCAP).mean(axis=-1) / CFG.ZCAP

def make_partitions(S):
    """Sub-station datasets D_k. NON_IID: every class (incl. Normal) is split over sub-stations with Dirichlet(alpha)."""
    N = CFG.N_SUBSTATIONS; r = np.random.default_rng(CFG.SEED); idx = [[] for _ in range(N)]
    for c in np.unique(S["atr"]):
        ids = r.permutation(np.where(S["atr"] == c)[0])
        pr = r.dirichlet(np.ones(N) * CFG.NON_IID_ALPHA) if CFG.NON_IID else np.ones(N) / N
        for k, part in enumerate(np.split(ids, (np.cumsum(pr)[:-1] * len(ids)).astype(int))): idx[k] += part.tolist()
    parts = [np.array(i) for i in idx]
    if min(len(p) for p in parts) < 5 * CFG.BATCH_SIZE:
        print("WARNING: very small sub-station -> falling back to IID partition"); parts = np.array_split(r.permutation(len(S["ytr"])), N)
    for k, p in enumerate(parts):
        print(f"  sub-station {k}: |D_k|={len(p):6d} | attack share={S['ytr'][p].mean():.1%} | types={len(np.unique(S['atr'][p]))}")
    return [(S["Xtr"][p], S["ytr"][p]) for p in parts]

# %%
# =====================================================================
# CELL 3 : PHASE 1 (federated TinyML classifier, FedAvg, baselines) + PHASE 2 (DTACE)
# =====================================================================
class TinyMLP(nn.Module):
    """d -> h1 -> h2 -> 1 logit ; P_i = sigmoid(logit) = attack probability."""
    def __init__(self, d, hidden=(32, 16)):
        super().__init__(); h1, h2 = hidden
        self.net = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def quantize(t, bits=8):
    qmax = 2 ** (bits - 1) - 1; scale = t.abs().max().clamp(min=1e-12) / qmax
    return torch.round(t / scale).clamp(-qmax, qmax).to(torch.int8), scale

def local_train(W_g, Xk, yk, d, epochs, pw):
    """Line 3: train on local D_k starting from W_g ; return dW_k = W_k - W_g (D_k never leaves)."""
    m = TinyMLP(d, CFG.HIDDEN).to(DEV); m.load_state_dict(W_g); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=CFG.LR); lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=DEV))
    X = torch.from_numpy(Xk).to(DEV); Y = torch.from_numpy(yk.astype(np.float32)).to(DEV); n = len(X); losses = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, CFG.BATCH_SIZE):
            b = perm[i:i + CFG.BATCH_SIZE]; loss = lossf(m(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.detach())
    Wk = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    return {k: Wk[k] - W_g[k] for k in W_g}, float(torch.stack(losses).mean())

@torch.no_grad()
def predict_proba(model, X, bs=16384):
    """Lines 4-6 (batch): P_i = sigmoid(f(x_i))."""
    model = model.to(DEV).eval(); out = []
    for i in range(0, len(X), bs): out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + bs]).to(DEV))).cpu().numpy())
    model.cpu(); return np.concatenate(out)

def to_model(W, d):
    m = TinyMLP(d, CFG.HIDDEN); m.load_state_dict(W); m.eval(); return m

def val_auc(model, S):
    f, t, _ = roc_curve(S["yval"], predict_proba(model, S["Xval"])); return auc(f, t)

def federated_train(parts, S, d, W0, pw, quantized=True, tag=""):
    """Line 12: W_g' = W_g + sum_k (|D_k|/|D|) dW_k ; dW_k 8-bit quantised if quantized=True."""
    W_g = {k: v.clone() for k, v in W0.items()}; n_tot = sum(len(p[0]) for p in parts); up = 0
    for r in range(1, CFG.FL_ROUNDS + 1):
        agg = {k: torch.zeros_like(v) for k, v in W_g.items()}; lo = []
        for Xk, yk in parts:
            dW, l = local_train(W_g, Xk, yk, d, CFG.LOCAL_EPOCHS, pw); lo.append(l)
            for k in dW:
                if quantized:
                    q, s = quantize(dW[k], CFG.QUANT_BITS); up += q.numel() + 4; dW[k] = q.float() * s
                else: up += dW[k].numel() * 4
                agg[k] += (len(Xk) / n_tot) * dW[k]
        W_g = {k: W_g[k] + agg[k] for k in W_g}
        if r == 1 or r % 5 == 0 or r == CFG.FL_ROUNDS:
            print(f"  [{tag}] round {r:02d}/{CFG.FL_ROUNDS} | local loss={np.mean(lo):.4f} | val AUC={val_auc(to_model(W_g, d), S):.4f}")
    return to_model(W_g, d), up

def standalone_train(Xk, yk, d, W0, pw):
    """Centralised / local-only baseline with the same total number of epochs as FL."""
    dW, _ = local_train(W0, Xk, yk, d, CFG.FL_ROUNDS * CFG.LOCAL_EPOCHS, pw)
    return to_model({k: W0[k] + dW[k] for k in W0}, d)

def dtace_trust(P, B, w1, w2):
    """Line 14:  T_i = w1*(1 - P_i) + w2*B_norm,i"""
    return w1 * (1.0 - P) + w2 * B

def device_ema(u, dev, alpha):
    """Per-device trust (used ONLY for the negative-result ablation)."""
    return pd.Series(np.asarray(u, float)).groupby(np.asarray(dev)).transform(
        lambda s: s.ewm(alpha=alpha, adjust=False).mean()).values

def calibrate(P_vb, B_vb):
    """P_th and T_th from BENIGN validation packets only."""
    P_th = CFG.P_TH if CFG.P_TH is not None else float(np.percentile(P_vb, CFG.P_TH_BENIGN_PCTL))
    keep = P_vb <= P_th; T = dtace_trust(P_vb[keep], B_vb[keep], CFG.W1, CFG.W2)
    T_th = CFG.T_TH if CFG.T_TH is not None else float(np.percentile(T, CFG.T_TH_BENIGN_PCTL))
    return P_th, T_th

# %%
# =====================================================================
# CELL 4 : PHASE 3 (latency, fast path) + PHASE 4 (guarded feedback) + streaming pipeline + metrics
# =====================================================================
class RuleTable:
    def __init__(self): self.rules, self.n = {}, 0
    def add(self, sig):
        dims, vals = sig; s = self.rules.setdefault(dims, set())
        if vals not in s and self.n < CFG.MAX_RULES: s.add(vals); self.n += 1; return True
        return False
    def match(self, x):
        for dims, vs in self.rules.items():
            if tuple(np.round(x[list(dims)] * CFG.SIG_BINS).astype(int).tolist()) in vs: return True
        return False

def make_signature(x, mu, sd):
    """U_threat signature = the K dims where the packet deviates most from benign + their quantised values."""
    dims = tuple(sorted(np.argsort(np.abs(x - mu) / sd)[-CFG.SIG_K:].tolist()))
    return dims, tuple(np.round(x[list(dims)] * CFG.SIG_BINS).astype(int).tolist())

def rule_is_safe(sig, Xval_benign):
    """Guard: a rule is broadcast only if it matches NO benign validation packet."""
    dims, vals = sig
    return not (np.round(Xval_benign[:, list(dims)] * CFG.SIG_BINS).astype(int) == np.array(vals)).all(1).any()

def run_pipeline(model, P_th, T_th, S, seed=0, fastpath=True, feedback=True):
    """Streaming execution of Lines 1-24, packet by packet, with latency tracking."""
    rng = np.random.default_rng(seed); pc = time.perf_counter; SL = CFG.EDGE_SLOWDOWN; N = CFG.N_SUBSTATIONS
    stations = [RuleTable() for _ in range(N)]; cache, pending, rows = {}, {}, []
    Xvb = S["Xval"][S["yval"] == 0]; t_p2_est, bc_total, rejected = 0.05, 0.0, 0
    n = len(S["yte"]) if CFG.STREAM_N is None else min(CFG.STREAM_N, len(S["yte"]))
    with torch.no_grad():
        for _ in range(100): model(torch.from_numpy(S["Xte"][0]).unsqueeze(0))
    def learn(x):
        nonlocal bc_total, rejected
        if not feedback: return
        sg = make_signature(x, S["MU_X"], S["SD_X"]); c = pending.get(sg, 0) + 1; pending[sg] = c
        if c == CFG.RULE_MIN_SIGHTINGS:
            if rule_is_safe(sg, Xvb):
                for st in stations: st.add(sg)
                bc_total += CFG.BCAST_HOP_MS * (N - 1)
            else: rejected += 1
    for i in range(n):
        x = S["Xte"][i]; dv = int(S["devte"][i]); k = dv % N
        row = dict(idx=i, dev=dv, y=int(S["yte"][i]), atype=S["atte"][i], P=np.nan, B=np.nan, T=np.nan, path="", fast=False,
                   decision="", t_air=0.0, t_p1=0.0, t_up=0.0, queue=0.0, t_p2=0.0, comp_ms=0.0)
        t_air = rng.uniform(*CFG.AIR_MS)
        t0 = pc(); hit = feedback and stations[k].match(x); t_rule = (pc() - t0) * 1e3 * SL
        if hit:
            row.update(path="RULE_DROP", decision="FL_BLOCK", t_air=t_air, t_p1=t_rule, latency=t_air + t_rule, comp_ms=t_rule / SL)
            rows.append(row); continue
        t0 = pc(); xt = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad(): lg = float(model(xt)[0])
        p = 1.0 / (1.0 + math.exp(-max(min(lg, 30.0), -30.0))); t_p1 = (pc() - t0) * 1e3 * SL
        row.update(P=p, t_air=t_air, t_p1=t_p1, comp_ms=t_p1 / SL)
        if p > P_th:                                        # Lines 7-9
            row.update(path="MICRO_PEP", decision="FL_BLOCK", latency=t_air + t_p1); learn(x); rows.append(row); continue
        t_up = rng.uniform(*CFG.UPLINK_MS)
        queue = rng.exponential(CFG.QUEUE_MEAN_MS) + (rng.uniform(*CFG.BURST_MS) if rng.random() < CFG.BURST_P else 0.0)
        pred = t_air + t_p1 + t_up + queue + t_p2_est
        use_fast = fastpath and pred > CFG.FASTPATH_MARGIN * CFG.LATENCY_BUDGET_MS and dv in cache
        t0 = pc(); p_use = cache[dv] if use_fast else p
        b = float(behaviour_norm(S["Bte"][i], S["MU_B"], S["SD_B"])); T = float(dtace_trust(p_use, b, CFG.W1, CFG.W2))
        t_p2 = (pc() - t0) * 1e3 * SL
        if use_fast: t_p2 += CFG.CACHE_LOOKUP_MS; queue = 0.0
        else: t_p2_est = 0.9 * t_p2_est + 0.1 * t_p2
        lat = t_air + t_p1 + t_up + queue + t_p2
        if T >= T_th:                                       # Lines 15-16
            dec = "ALLOW"; lat += rng.uniform(*CFG.DC_STORE_MS)
            if not use_fast: cache[dv] = 0.8 * cache.get(dv, p) + 0.2 * p
        else: dec = "ISOLATE"; learn(x)                     # Lines 17-19
        row.update(B=b, T=T, path="FASTPATH" if use_fast else "MACRO_PEP", fast=use_fast, decision=dec,
                   t_up=t_up, queue=queue, t_p2=t_p2, latency=lat, comp_ms=(t_p1 + t_p2) / SL)
        rows.append(row)
    tr = pd.DataFrame(rows); tr.attrs.update(rules=sum(s.n for s in stations) // max(N, 1), bcast_ms=bc_total, rejected=rejected); return tr

def cloud_zta_latency(n, seed=1):
    rng = np.random.default_rng(seed); wan = lambda: rng.lognormal(np.log(CFG.CLOUD_WAN_MEDIAN_MS), CFG.CLOUD_WAN_SIGMA, n)
    return (rng.uniform(*CFG.AIR_MS, n) + wan() + rng.exponential(CFG.QUEUE_MEAN_MS, n) + rng.uniform(*CFG.CLOUD_PDP_MS, n) + wan())

def metrics_from_pred(y, yp):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1); prec = tp / max(tp + fp, 1)
    return dict(accuracy=(tp + tn) / len(y), precision=prec, recall=tpr, f1=2 * prec * tpr / max(prec + tpr, 1e-12),
                fpr=fpr, balanced_acc=(tpr + 1 - fpr) / 2, mcc=float(matthews_corrcoef(y, yp)),
                precision_at_50pct_prevalence=tpr / max(tpr + fpr, 1e-12))

def metrics_from_trace(tr): return metrics_from_pred(tr["y"].values, (tr["decision"] != "ALLOW").astype(int).values)

def roc_stats(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return dict(auc=float(auc(fpr, tpr)), **{f"tpr@fpr{int(f*100)}%": float(np.interp(f, fpr, tpr)) for f in (0.01, 0.02, 0.05)})

# %%
# =====================================================================
# CELL 5 : RUN  (training, FL comparison, calibration, ablation, streaming, console output)
# =====================================================================
t_start = time.time()
D = load_and_map(); S = split_and_scale(D); d = S["Xtr"].shape[1]
pw = float((S["ytr"] == 0).sum() / max((S["ytr"] == 1).sum(), 1))          # class balancing weight (attack-heavy data)
print(f"positive-class weight (balances benign vs attack): {pw:.3f}")
parts = make_partitions(S)
W0 = {k: v.clone() for k, v in TinyMLP(d, CFG.HIDDEN).state_dict().items()}

print("\n=== FL-DTACE: federated training, 8-bit quantised updates ===")
model_q, bytes_q = federated_train(parts, S, d, W0, pw, True, "FL int8")
print("\n=== Baseline: standard FedAvg (float32) ===")
model_f, bytes_f = federated_train(parts, S, d, W0, pw, False, "FedAvg fp32")
print("\n=== Baseline: centralised classifier (pools all data - no privacy) ===")
model_c = standalone_train(S["Xtr"], S["ytr"], d, W0, pw)
print("=== Baseline: local-only classifiers (no FL) ===")
models_local = [standalone_train(Xk, yk, d, W0, pw) for Xk, yk in parts]
print(f"\nUplink bytes: int8={bytes_q:,} | fp32={bytes_f:,} | saving={1 - bytes_q / bytes_f:.1%}")

vb = S["yval"] == 0
P_vb = predict_proba(model_q, S["Xval"])[vb]; B_vb = behaviour_norm(S["Bval"][vb], S["MU_B"], S["SD_B"])
P_TH, T_TH = calibrate(P_vb, B_vb)
print(f"Thresholds (benign-validation calibrated) -> P_th={P_TH:.4f} | T_th={T_TH:.4f} | w1={CFG.W1} w2={CFG.W2}")

y, dev_te, atte = S["yte"], S["devte"], S["atte"]
P_te = predict_proba(model_q, S["Xte"]); B_te = behaviour_norm(S["Bte"], S["MU_B"], S["SD_B"]); T_te = dtace_trust(P_te, B_te, CFG.W1, CFG.W2)
K_FULL, K_CLF, K_B = "FL-DTACE: classifier + B_norm (proposed)", "Classifier only (FL int8)", "B_norm only (standard ZTA rules)"
K_DEV, K_CTRL = "NEG. RESULT: + per-device trust", "NEG. RESULT control: shuffled device ids"
scores = {K_CLF: P_te, K_B: 1 - B_te, K_FULL: 1 - T_te,
          K_DEV: 1 - device_ema(T_te, dev_te, CFG.ALPHA),
          K_CTRL: 1 - device_ema(T_te, np.random.default_rng(0).permutation(dev_te), CFG.ALPHA)}
ablation = {k: roc_stats(y, v) for k, v in scores.items()}
print("\n=== ABLATION (full test set, threshold-free) ===\n" + pd.DataFrame(ablation).T.round(4).to_string())
unseen = {}
if CFG.HOLDOUT_ATTACKS:
    ho = np.isin(atte, CFG.HOLDOUT_ATTACKS)
    for nm, mask in [("SEEN attack types", (y == 0) | ~ho), ("UNSEEN attack types", (y == 0) | ho)]:
        unseen[nm] = {k: roc_stats(y[mask], v[mask]) for k, v in scores.items() if k in (K_CLF, K_B, K_FULL)}
        print(f"\n--- {nm} vs benign ---\n" + pd.DataFrame(unseen[nm]).T.round(4).to_string())

pf, pc_ = predict_proba(model_f, S["Xte"]), predict_proba(model_c, S["Xte"]); loc = [predict_proba(m, S["Xte"]) for m in models_local]
fl_cmp = {"Centralised (no privacy)": roc_stats(y, pc_), "FedAvg float32": roc_stats(y, pf), "FL-DTACE int8": roc_stats(y, P_te)}
la = [roc_stats(y, p)["auc"] for p in loc]
fl_cmp["Local-only (mean of sub-stations)"] = dict(auc=float(np.mean(la)), worst=float(np.min(la)), best=float(np.max(la)))
print("\n=== FL vs centralised vs local-only (classifier score, full test set) ===\n" + pd.DataFrame(fl_cmp).T.round(4).to_string())
a_full, a_clf = ablation[K_FULL]["auc"], ablation[K_CLF]["auc"]
print(f"\nVERDICT: AUC classifier-only={a_clf:.4f} | FL-DTACE={a_full:.4f} | B-only={ablation[K_B]['auc']:.4f}  "
      f"(B_norm fusion changes AUC by {a_full - a_clf:+.4f})")

print("\n=== Streaming pipeline (final decisions) ===")
trace = run_pipeline(model_q, P_TH, T_TH, S, seed=7, fastpath=True, feedback=True)
trace_nf = run_pipeline(model_q, P_TH, T_TH, S, seed=7, fastpath=False, feedback=True)
trace_nb = run_pipeline(model_q, P_TH, T_TH, S, seed=7, fastpath=True, feedback=False)
show = pd.concat([trace.head(CFG.N_PRINT)] + [trace[trace.path == p].head(2) for p in ["MICRO_PEP", "MACRO_PEP", "FASTPATH", "RULE_DROP"]
                 if p in set(trace.path)]).drop_duplicates("idx")
print("\nPer-packet trace (P_i, B_norm,i, T_i, decision, latency ms):")
print(show[["idx", "dev", "y", "P", "B", "T", "path", "decision", "latency"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
m = metrics_from_trace(trace); m_nb = metrics_from_trace(trace_nb)
lat, lat_nf, lat_nb = trace["latency"].values, trace_nf["latency"].values, trace_nb["latency"].values; cloud = cloud_zta_latency(len(trace))
print("\n=== Detection (streaming, final decisions) ===\n", {k: round(float(v), 4) for k, v in m.items()})
print("Feedback OFF:", {k: round(float(v), 4) for k, v in m_nb.items() if k in ("recall", "fpr", "mcc", "balanced_acc")})
rd = trace[trace.path == "RULE_DROP"]
print(f"Feedback rules: {trace.attrs['rules']} per sub-station | rule drops={len(rd)} (of which benign/false={int((rd.y == 0).sum())}) | rules rejected by benign guard={trace.attrs['rejected']}")
print("\n=== End-to-end latency (ms): measured compute + modelled network ===")
for name, a in [("FL-DTACE (fast path)", lat), ("FL-DTACE (no fast path)", lat_nf), ("FL-DTACE (feedback off)", lat_nb), ("Cloud ZTA (simulated)", cloud)]:
    print(f"{name:26s} mean={a.mean():6.2f} p50={np.percentile(a, 50):6.2f} p95={np.percentile(a, 95):6.2f} p99={np.percentile(a, 99):6.2f} <10ms={np.mean(a < 10):.2%}")
cmp_ = trace.comp_ms.values
print(f"Compute-only (measured on this CPU, no network, no slowdown): mean={cmp_.mean():.3f} ms | p99={np.percentile(cmp_, 99):.3f} ms")
print(f"Path usage: {trace.path.value_counts().to_dict()} | fast-path ratio={trace.fast.mean():.2%}")
print(f"Total wall time: {time.time() - t_start:.1f}s")

# %%
# =====================================================================
# CELL 6 : PLOTS (1 ROC, 2 latency CDF, 3 trust scatter, 4 ablation, 5 latency breakdown)
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 7))
for name, c in [(K_FULL, "#d62728"), (K_CLF, "#1f77b4"), (K_B, "#2ca02c")]:
    f_, t_, _ = roc_curve(y, scores[name]); ax.plot(f_, t_, lw=2.5, color=c, label=f"{name}  AUC={ablation[name]['auc']:.3f}")
f_, t_, _ = roc_curve(y, pf); ax.plot(f_, t_, lw=1.8, ls="--", color="#9467bd", label=f"Standard FedAvg (fp32)  AUC={fl_cmp['FedAvg float32']['auc']:.3f}")
ax.scatter([m["fpr"]], [m["recall"]], marker="*", s=300, color="gold", edgecolor="k", zorder=5, label="Pipeline operating point")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Plot 1: Detection Efficacy (ROC)")
ax.legend(fontsize=9, loc="lower right"); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot1_roc.png", dpi=300); plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
for name, a, c, ls in [("FL-DTACE edge (fast path)", lat, "#d62728", "-"), ("FL-DTACE edge (no fast path)", lat_nf, "#ff7f0e", "--"),
                       ("Cloud-based ZTA (simulated WAN)", cloud, "#1f77b4", "-")]:
    xs = np.sort(a); ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, ls=ls, lw=2.5, label=f"{name} (p99={np.percentile(a, 99):.1f} ms)")
ax.axvline(CFG.LATENCY_BUDGET_MS, color="k", ls=":", lw=2)
ax.text(CFG.LATENCY_BUDGET_MS * 1.05, 0.25, f"10 ms budget\n{np.mean(lat < 10):.1%} of packets under", fontsize=12)
ax.set_xscale("log"); ax.set(xlabel="End-to-end decision latency (ms, log)", ylabel="CDF", title="Plot 2: Latency Guarantee (CDF)")
ax.legend(fontsize=10, loc="center right"); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot2_latency_cdf.png", dpi=300); plt.show()

has_T = trace["T"].notna(); mac = trace[has_T].sample(n=min(4000, int(has_T.sum())), random_state=1).sort_values("idx")
fig, ax = plt.subplots(figsize=(11, 6))
for dec, c in [("ALLOW", "#2ca02c"), ("ISOLATE", "#d62728")]:
    for yv, mk in [(0, "o"), (1, "X")]:
        s_ = mac[(mac.decision == dec) & (mac.y == yv)]
        ax.scatter(s_.idx, s_["T"], c=c, marker=mk, s=26 if yv == 0 else 55, alpha=.6, label=f"{dec} | true {'benign' if yv == 0 else 'attack'}")
ax.axhline(T_TH, color="k", ls="--", lw=2); ax.text(mac.idx.min(), T_TH + .01, f"T_th = {T_TH:.3f}", fontsize=12)
ax.set(xlabel="Packet index (stream order)", ylabel="Dynamic trust score T_i", title="Plot 3: DTACE Trust Score - ALLOW vs ISOLATE")
ax.legend(fontsize=10, loc="lower left", ncol=2); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot3_trust_scatter.png", dpi=300); plt.show()

ab = pd.DataFrame(ablation).T[["auc", "tpr@fpr5%"]]
fig, ax = plt.subplots(figsize=(11, 6)); ab.plot.barh(ax=ax, color=["#1f77b4", "#d62728"]); ax.invert_yaxis()
ax.set(xlim=(0, 1.05), title="Plot 4: Ablation - what each component adds"); ax.legend(["AUC", "TPR @ 5% FPR"], loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot4_ablation.png", dpi=300); plt.show()

comp = trace[trace.path.isin(["MACRO_PEP", "FASTPATH"])][["t_air", "t_p1", "t_up", "queue", "t_p2"]].mean()
fig, ax = plt.subplots(figsize=(11, 3.5)); left = 0
for nm, c in zip(comp.index, sns.color_palette("deep", 5)):
    ax.barh(["edge path"], [comp[nm]], left=left, color=c, label=f"{nm}={comp[nm]:.2f} ms"); left += comp[nm]
ax.set(xlabel="mean ms (packets reaching macro-edge)", title="Plot 5: Latency breakdown (air/queue/uplink modelled; p1/p2 measured x slowdown)")
ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.35)); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plot5_latency_breakdown.png", dpi=300); plt.show()

# %%
# =====================================================================
# CELL 7 : TABLES, per-attack results, w1/w2 sensitivity, SAVE EVERYTHING (+ zip download in Colab)
# =====================================================================
per = trace.assign(detected=(trace.decision != "ALLOW")).groupby("atype").agg(
    packets=("idx", "count"), detection_rate=("detected", "mean"), mean_latency_ms=("latency", "mean")).round(4)
print("Per-attack-type result (Normal row = false-positive rate):\n", per)
sens = pd.DataFrame([(round(w1, 1), round(1 - w1, 1), round(roc_stats(y, 1 - dtace_trust(P_te, B_te, w1, 1 - w1))["auc"], 4))
                     for w1 in np.linspace(0, 1, 11)], columns=["w1", "w2", "AUC (classifier + B_norm)"])
print("\nw1/w2 sensitivity (report as sensitivity only; do NOT tune on test):\n", sens.to_string(index=False))

def md(df):
    cols = [str(c) for c in df.columns]; out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)
lat_rows = [{"System": n_, "mean_ms": a.mean(), "p50_ms": np.percentile(a, 50), "p95_ms": np.percentile(a, 95),
             "p99_ms": np.percentile(a, 99), "pct_under_10ms": 100 * np.mean(a < 10)}
            for n_, a in [("FL-DTACE (fast path)", lat), ("FL-DTACE (no fast path)", lat_nf), ("FL-DTACE (feedback off)", lat_nb), ("Cloud ZTA (simulated)", cloud)]]
fb = pd.DataFrame([{"Feedback loop": "ON", **{k: float(m[k]) for k in ("recall", "fpr", "mcc", "balanced_acc")}, "mean_latency_ms": lat.mean()},
                   {"Feedback loop": "OFF", **{k: float(m_nb[k]) for k in ("recall", "fpr", "mcc", "balanced_acc")}, "mean_latency_ms": lat_nb.mean()}])
tables = [("Detection (streaming final decisions)", pd.DataFrame([{k: float(v) for k, v in m.items()}])),
          ("Ablation (full test set)", pd.DataFrame(ablation).T.reset_index().rename(columns={"index": "variant"})),
          ("FL vs centralised vs local-only", pd.DataFrame(fl_cmp).T.reset_index().rename(columns={"index": "method"})),
          ("Latency (ms)", pd.DataFrame(lat_rows)), ("Feedback loop ON/OFF", fb),
          ("Communication cost", pd.DataFrame([{"update": "int8 dW", "uplink_bytes": int(bytes_q)}, {"update": "float32 dW", "uplink_bytes": int(bytes_f)}])),
          ("Per-attack detection", per.reset_index()), ("w1/w2 sensitivity", sens)]
for nm, u in unseen.items(): tables.append((f"Hold-out test: {nm}", pd.DataFrame(u).T.reset_index().rename(columns={"index": "variant"})))
with open(f"{OUT_DIR}/thesis_tables.md", "w", encoding="utf-8") as f:
    f.write(f"# Tables | tag={CFG.RUN_TAG} seed={CFG.SEED} w1={CFG.W1} w2={CFG.W2} | P_th={P_TH:.4f} T_th={T_TH:.4f} | rows={len(D['y']):,} | holdout={CFG.HOLDOUT_ATTACKS}\n\n")
    for ttl, t in tables: f.write(f"## {ttl}\n\n{md(t)}\n\n")
trace.to_csv(f"{OUT_DIR}/pipeline_trace.csv", index=False); per.to_csv(f"{OUT_DIR}/per_attack.csv")
json.dump(dict(config={k: str(getattr(CFG, k)) for k in CFG.__dataclass_fields__}, P_th=P_TH, T_th=T_TH,
               metrics={k: float(v) for k, v in m.items()}, ablation=ablation, fl_comparison=fl_cmp, unseen=unseen,
               latency={r["System"]: {k: float(v) for k, v in r.items() if k != "System"} for r in lat_rows},
               compute_only_ms=dict(mean=float(cmp_.mean()), p99=float(np.percentile(cmp_, 99))),
               uplink_bytes=dict(int8=int(bytes_q), fp32=int(bytes_f)), rows=int(len(D["y"])), test_attack_share=float(y.mean())),
          open(f"{OUT_DIR}/summary.json", "w"), indent=2)
print(f"\nSaved to {OUT_DIR}/  (5 plots, console_log.txt, thesis_tables.md, summary.json, pipeline_trace.csv, per_attack.csv)")
sys.stdout.flush(); sys.stdout = sys.__stdout__
try:
    from google.colab import files; import shutil
    shutil.make_archive(OUT_DIR, "zip", OUT_DIR); files.download(OUT_DIR + ".zip")
except Exception: pass

# %%
# =====================================================================
# CELL 8 (OPTIONAL, after several runs): aggregate mean +- std over seeds
# =====================================================================
ROOT = CFG.OUT_ROOT or "."
rows = []
for p in sorted(glob.glob(os.path.join(ROOT, "results_*", "summary.json"))):
    s = json.load(open(p)); c = s["config"]
    rows.append(dict(run=os.path.basename(os.path.dirname(p)), tag=c["RUN_TAG"], seed=int(c["SEED"]), w1=float(c["W1"]),
                     AUC_full=s["ablation"][K_FULL]["auc"], AUC_clf=s["ablation"][K_CLF]["auc"], AUC_B=s["ablation"][K_B]["auc"],
                     TPR5=s["ablation"][K_FULL]["tpr@fpr5%"], recall=s["metrics"]["recall"], fpr=s["metrics"]["fpr"],
                     mcc=s["metrics"]["mcc"], bal_acc=s["metrics"]["balanced_acc"],
                     p99_ms=s["latency"]["FL-DTACE (fast path)"]["p99_ms"], under10=s["latency"]["FL-DTACE (fast path)"]["pct_under_10ms"]))
R = pd.DataFrame(rows)
if len(R):
    print("All runs:\n", R.round(4).to_string(index=False))
    base = R[R.w1.round(2) == 0.60]
    print("\nMEAN +- STD over seeds (w1=0.6 runs only), per tag:")
    print(base.groupby("tag")[["AUC_full", "AUC_clf", "AUC_B", "TPR5", "recall", "fpr", "mcc", "bal_acc", "p99_ms", "under10"]].agg(["mean", "std"]).round(4).T.to_string())
    R.to_csv(os.path.join(ROOT, "all_runs_summary.csv"), index=False)
else:
    print("No results_*/summary.json found under", ROOT)
