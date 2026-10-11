"""
Permutation null model for sermon co-occurrence in manuscripts (vectorised).

Pipeline per century
  1. Build the manuscript-sermon incidence (edge list -> integer index arrays).
  2. Observed sermon-sermon weights  W_obs = B.T @ B  (number of manuscript
     pairs in which two sermons co-occur; identical to the old double loop).
  3. Null model: curveball algorithm (Strona et al. 2014). Two manuscripts
     trade the sermons they do not share. This keeps the number of sermons per
     manuscript and the frequency of every sermon exactly fixed, and never puts
     the same sermon twice in one manuscript. TRADES_PER_MS * n_manuscripts
     trades per iteration gives a fully mixed random network.
  4. For every random network: W_rand = B_rand.T @ B_rand (scipy.sparse),
     looked up only at the observed pairs, and exceedance counters updated.
  5. p-value per observed edge (binomial fitted to the simulated null),
     Benjamini-Hochberg FDR correction, binary adjacency matrix
     sermon x sermon (1 = significant edge), saved as .npy and labelled .csv,
     plus an edge table with weights, null mean/sd, z-score and p-value.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binom, false_discovery_control
from tqdm import tqdm
from datetime import datetime

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT_IN = "./data/data_raw/"
MS_SE_PATH = f"{ROOT_IN}sf_mss_okt_25.xlsx"
ROOT_OUT = "./data/data_deriv"

CENTURIES = [13] # list of centuries to run, or None for all
NUMBER_ITERATIONS = 1000
TRADES_PER_MS = 10            # curveball trades per iteration = this * n_manuscripts
THRESHOLD_ALPHA = 0.01        # applied to the FDR corrected q value
TAIL = "upper"                # "upper": edge is significant when observed
                              #          weight is HIGHER than the null
                              # "lower": significant when LOWER than the null
SEED = 42
PLOT_BOXPLOT = False
N_BOXPLOT = 10                # number of edges kept for the boxplot

SAVE_ADJ = True
SAVE_EDGES = False

# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def create_ms_se_arrays(sub_df: pd.DataFrame):
    """Return integer-coded edge arrays plus the label arrays.

    ms_idx[k], se_idx[k] is the k-th manuscript-sermon edge.
    Sermon labels are sorted, so index order equals string order.
    """
    s = sub_df[["shelfmark", "certain"]].copy()
    s = s[s["certain"].apply(lambda x: isinstance(x, str))]
    s["certain"] = s["certain"].str.split(", ")
    s["certain"] = s["certain"].apply(lambda x: [e.strip() for e in x])
    s = s.explode("certain")
    s = s[s["certain"].notna() & (s["certain"] != "")]
    s = s.drop_duplicates(["shelfmark", "certain"])   # a sermon counts once per ms

    ms_labels, ms_idx = np.unique(s["shelfmark"].astype(str).to_numpy(), return_inverse=True)
    se_labels, se_idx = np.unique(s["certain"].astype(str).to_numpy(), return_inverse=True)
    return ms_idx.astype(np.int64), se_idx.astype(np.int64), ms_labels, se_labels


def cooccurrence(ms_idx: np.ndarray, se_idx: np.ndarray, n_ms: int, n_se: int) -> sp.csr_matrix:
    """Sermon x sermon co-occurrence weights (upper triangle, diagonal removed)."""
    data = np.ones(len(ms_idx), dtype=np.int32)
    B = sp.csr_matrix((data, (ms_idx, se_idx)), shape=(n_ms, n_se))  # duplicates are summed
    W = (B.T @ B).tocsr()
    return sp.triu(W, k=1, format="csr")


def lookup(W: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray, n_se: int) -> np.ndarray:
    """Values of sparse W at (rows, cols); zeros where W has no entry."""
    W = W.tocoo()
    keys = W.row.astype(np.int64) * n_se + W.col
    order = np.argsort(keys)
    keys, vals = keys[order], W.data[order]

    q = rows.astype(np.int64) * n_se + cols
    pos = np.searchsorted(keys, q)
    pos_c = np.minimum(pos, len(keys) - 1)
    hit = (pos < len(keys)) & (keys[pos_c] == q)
    out = np.zeros(len(q), dtype=np.int64)
    out[hit] = vals[pos_c[hit]]
    return out


def curveball(ms_sets: list[set], n_trades: int, rng: np.random.Generator) -> list[set]:
    """One random network: manuscripts trade their unshared sermons in pairs."""
    sets = [s.copy() for s in ms_sets]
    for a, b in rng.integers(0, len(sets), size=(n_trades, 2)):
        shared = sets[a] & sets[b]
        pool = list((sets[a] | sets[b]) - shared)
        k = len(sets[a]) - len(shared)
        if a == b or k == 0 or k == len(pool):
            continue
        pool = rng.permutation(pool)
        sets[a] = shared | set(pool[:k])
        sets[b] = shared | set(pool[k:])
    return sets


# --------------------------------------------------------------------------- #
# Permutation test
# --------------------------------------------------------------------------- #
def permutation_test(sub_df: pd.DataFrame, n_iter: int, trades_per_ms: float,
                     rng: np.random.Generator, keep_dist_for: int = 0):
    ms_idx, se_idx, ms_labels, se_labels = create_ms_se_arrays(sub_df)
    n_ms, n_se, E = len(ms_labels), len(se_labels), len(ms_idx)

    W_obs = cooccurrence(ms_idx, se_idx, n_ms, n_se).tocoo()
    rows, cols, obs = W_obs.row, W_obs.col, W_obs.data.astype(np.int64)
    n_pairs = len(obs)
    print(f"Edges ms-se: {E}, sermons: {n_se}, manuscripts: {n_ms}, "
          f"unique se-se edges: {n_pairs}, se-se edges: {int(obs.sum())}")

    n_ge = np.zeros(n_pairs, dtype=np.int64)    # random >= observed
    n_le = np.zeros(n_pairs, dtype=np.int64)    # random <= observed
    s1 = np.zeros(n_pairs, dtype=np.float64)    # running sum for null mean
    s2 = np.zeros(n_pairs, dtype=np.float64)    # running sum of squares
    dist = np.zeros((n_iter, keep_dist_for), dtype=np.int64) if keep_dist_for else None

    ms_sets = [set(se_idx[ms_idx == m]) for m in range(n_ms)]
    sizes = [len(s) for s in ms_sets]
    ms_rand = np.repeat(np.arange(n_ms), sizes)           # fixed: ms sizes never change
    n_trades = int(trades_per_ms * n_ms)

    for it in tqdm(range(n_iter)):
        sets = curveball(ms_sets, n_trades, rng)
        se_rand = np.fromiter((se for s in sets for se in s), dtype=np.int64, count=E)
        rnd = lookup(cooccurrence(ms_rand, se_rand, n_ms, n_se), rows, cols, n_se)
        n_ge += rnd >= obs
        n_le += rnd <= obs
        s1 += rnd
        s2 += rnd.astype(np.float64) ** 2
        if dist is not None:
            dist[it] = rnd[:keep_dist_for]

    mean = s1 / n_iter
    sd = np.sqrt(np.maximum(s2 / n_iter - mean ** 2, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (obs - mean) / sd, np.nan)

    edges = pd.DataFrame({
        "sermon_1": se_labels[cols],          # se1 > se2, as in the original
        "sermon_2": se_labels[rows],
        "weight_obs": obs,
        "null_mean": mean,
        "null_sd": sd,
        "z": z,
        "p_emp_upper": (n_ge + 1) / (n_iter + 1),  # share of random >= observed
        "p_emp_lower": (n_le + 1) / (n_iter + 1),  # share of random <= observed
    })
    # Empirical p cannot go below 1/(n_iter+1), too coarse for FDR over
    # thousands of pairs. A binomial with the simulated mean and variance gives
    # the exact tail (it matches the empirical p where both are resolvable).
    mean_, var_ = np.maximum(mean, 1 / n_iter), sd ** 2
    p_ = np.clip(1 - var_ / mean_, 1e-9, 1)
    n_ = np.maximum(np.round(mean_ / p_), obs)
    p_ = np.minimum(mean_ / n_, 1)
    edges["p_upper"] = binom.sf(obs - 1, n_, p_)
    edges["p_lower"] = binom.cdf(obs, n_, p_)
    # Benjamini-Hochberg FDR correction over all tested pairs
    edges["q_upper"] = false_discovery_control(edges["p_upper"])
    edges["q_lower"] = false_discovery_control(edges["p_lower"])
    return edges, rows, cols, se_labels, dist


def adjacency_from_edges(edges: pd.DataFrame, rows, cols, n_se: int,
                         alpha: float, tail: str) -> np.ndarray:
    sig = edges[f"q_{tail}"].to_numpy() < alpha
    A = np.zeros((n_se, n_se), dtype=np.uint8)
    A[rows[sig], cols[sig]] = 1
    A[cols[sig], rows[sig]] = 1               # undirected -> symmetric
    return A


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(ROOT_OUT, exist_ok=True)

    msSeDf = pd.read_excel(MS_SE_PATH)
    print(msSeDf.columns)

    date = datetime.now().strftime('%d%m%y')

    for century, sub_df in msSeDf.groupby("century"):
        if CENTURIES is not None and century not in CENTURIES:
            continue
        print(f"\n=== Century {century} ===")

        edges, rows, cols, se_labels, dist = permutation_test(
            sub_df, NUMBER_ITERATIONS, TRADES_PER_MS, rng,
            keep_dist_for=N_BOXPLOT if PLOT_BOXPLOT else 0)

        A = adjacency_from_edges(edges, rows, cols, len(se_labels), THRESHOLD_ALPHA, TAIL)
        pcol = f"q_{TAIL}"
        edges["significant"] = (edges[pcol] < THRESHOLD_ALPHA).astype(np.uint8)
        print(f"Significant edges ({TAIL} tail, alpha={THRESHOLD_ALPHA}): "
              f"{int(edges['significant'].sum())} of {len(edges)}")

        base = os.path.join(ROOT_OUT, f"se_se_c{int(century)}")

        if SAVE_ADJ:
            pd.DataFrame(A, index=se_labels, columns=se_labels).to_csv(f"{base}_a%{THRESHOLD_ALPHA*100}_adj_{date}.csv")
            print(f"Saved {base}_adjacency.csv")
        if SAVE_EDGES:
            edges.sort_values(pcol).to_csv(f"{base}_a%{THRESHOLD_ALPHA*100}_edges_{date}.csv", index=False)
            print(f"Saved {base}_edges.csv")

        if PLOT_BOXPLOT and dist is not None and dist.shape[1]:
            k = dist.shape[1]
            plt.boxplot([dist[:, j] for j in range(k)])
            plt.scatter(range(1, k + 1), edges["weight_obs"].to_numpy()[:k], color="red")
            plt.xticks(range(1, k + 1),
                       [f"{a}\n{b}" for a, b in zip(edges["sermon_1"][:k], edges["sermon_2"][:k])],
                       rotation=90, fontsize=6)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
