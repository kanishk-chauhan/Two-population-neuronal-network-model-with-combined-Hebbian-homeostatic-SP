"""
Extract the data needed for every data-driven figure in plot_results.ipynb from the
raw simulation output of two-pop-neural-net-homeo-hebb-plasticity.cc, and save it as
one CSV file per figure in figure_data/.

This script needs the raw simulation output, which is not part of the repository.
It is included to document exactly how every value in figure_data/ was obtained.

Usage (run from this folder):
    python extract_figure_data.py [--raw-root PATH]

--raw-root is the folder that contains the raw data folders (e.g., Data/).
Default: the parent folder of this script.

Raw file formats (as written by two-pop-neural-net-homeo-hebb-plasticity.cc):
  STDP+SP(...).dat           one row per save event (end of relaxation, then every
                             10 simulated minutes), 16 columns:
                               0 odpr_1   1 fi_mean_1   2 Avg_W_1      3 node_degree_density_1
                               4 odpr_2   5 fi_mean_2   6 Avg_W_2      7 node_degree_density_2
                               8 odpr_12  9 Avg_W_1to2 10 ndd_1to2    11 Avg_W_2to1
                              12 ndd_2to1 13 pruned    14 added       15 iSPupdate
  W_STDP+SP(...).dat         initial N x N weight matrix (A*W), followed by the final one
  Correlation_dist_...(...).dat
                             row 0: 101 bin edges on [-1, 1]; then, per save event,
                             one row of same-pop counts and one row of cross-pop counts
                             (100 counts + a trailing 0 each)

Raw file names: the names used below are those of the files used for the published figures.
Adapt the *_filename() helpers if your runs are named differently (the file names written by
the current version of two-pop-neural-net-homeo-hebb-plasticity.cc are set in setup_files()).
"""

import argparse
import csv
import os

import numpy as np

# ---------------------------------------------------------------------------------
# Simulation-output conventions
# ---------------------------------------------------------------------------------
SAVE_INTERVAL_MIN = 10          # minutes between saved rows / correlation histograms
TW_MS = 6e3                     # simulation window length Tw (ms)
AVERAGE_OVER_H = 5              # hours averaged at the end of a run for the maps
AVERAGE_OVER_ROWS = int(AVERAGE_OVER_H * (60 / SAVE_INTERVAL_MIN))   # = 30 rows

# SP-model variants: key -> (raw data folder, SP model, r_Hom / r_Hebb)
VARIANTS = {
    "Original":  ("DataSpontaneous",   "original", 1.0),
    "Original1": ("DataSpontaneous1",  "original", 0.1),
    "HebHom2":   ("DataSpont_HebHom2", "extended", 1.0),
    "HebHom3":   ("DataSpont_HebHom3", "extended", 0.1),
}
VARIANT_COLUMNS = ["variant", "sp_model", "r_hom_over_r_hebb"]

MEAN_INITIAL_W = np.arange(0.1, 1.0, 0.1)        # initial intra-pop mean synaptic weight
MEAN_INITIAL_NDD = np.arange(0.05, 0.31, 0.05)   # initial intra-pop node degree density 
P_CROSS_POP = [0.05, 0.1, 0.15]                  # initial inter-pop node degree density
CROSS_POP_DELAY = 10.0                           # ms
SAME_POP_DELAY = 3.0                             # ms

# Columns of STDP+SP(...).dat used by the time-trace figures
TRACE_COLUMNS = {0: "odpr_1", 1: "fi_mean_1", 2: "Avg_W_1", 3: "node_degree_density_1",
                 9: "Avg_W_1to2", 10: "node_degree_density_1to2",
                 11: "Avg_W_2to1", 12: "node_degree_density_2to1"}


# ---------------------------------------------------------------------------------
# Raw file names
# ---------------------------------------------------------------------------------
def spont_filename(Wmean, beta, p_cross, cross_pop_delay=CROSS_POP_DELAY, same_pop_delay=SAME_POP_DELAY):
    return "STDP+SP(Wmean=%.1f,beta=%.3f,p_cross_pop0=%.3f,cross_pop_delay=%.1f,same_pop_delay=%.1f)1.dat" % (
        Wmean, beta, p_cross, cross_pop_delay, same_pop_delay)


def spont_weight_filename(Wmean, beta, p_cross, cross_pop_delay=CROSS_POP_DELAY, same_pop_delay=SAME_POP_DELAY):
    return "W_" + spont_filename(Wmean, beta, p_cross, cross_pop_delay, same_pop_delay)


def spont_corr_dist_filename(Wmean, beta, p_cross, cross_pop_delay=CROSS_POP_DELAY, same_pop_delay=SAME_POP_DELAY):
    return "Correlation_dist_pRVS(Wmean=%.1f,beta=%.3f,p_cross_pop0=%.3f,cross_pop_delay=%.1f,same_pop_delay=%.1f)1.dat" % (
        Wmean, beta, p_cross, cross_pop_delay, same_pop_delay)


def corr_study_filename(corr_dep_add, corr_th0, delta_corr, same_pop_delay, trial_number):
    return "STDP+SP(corr-dep-add-%s,corr_th=%.1f,delta_corr=%.1f,same_pop_delay=%.1f)%d.dat" % (
        corr_dep_add, corr_th0, delta_corr, same_pop_delay, trial_number)


def corr_study_corr_dist_filename(corr_dep_add, corr_th0, delta_corr, same_pop_delay, trial_number):
    return "Correlation_dist_pRVS(corr-dep-add-%s,corr_th=%.1f,delta_corr=%.1f,same_pop_delay=%.1f)%d.dat" % (
        corr_dep_add, corr_th0, delta_corr, same_pop_delay, trial_number)


# ---------------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------------
def fmt(x):
    """Numbers are written as the shortest string that reads back to the same float."""
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def write_csv(path, description, header, rows):
    """Write a CSV file whose first lines are '#' comments describing its content."""
    with open(path, "w", newline="") as fh:
        for line in description.strip("\n").split("\n"):
            fh.write("# " + line + "\n" if line else "#\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow([fmt(v) for v in row])
    print("wrote %s (%d rows)" % (path, len(rows)))


def bin_columns(bin_edges):
    """One column per histogram bin, named by the bin's edges as '[lo,hi)'."""
    return ["[%s,%s)" % (fmt(lo), fmt(hi)) for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]


def variant_fields(key):
    _, sp_model, ratio = VARIANTS[key]
    return [key, sp_model, ratio]


# ---------------------------------------------------------------------------------
# Figure: adjacency_matrix.jpeg
# ---------------------------------------------------------------------------------
def extract_adjacency_matrix(raw_root, out_dir):
    variant = "HebHom3"
    fname = spont_weight_filename(0.8, 0.1, 0.1)
    W = np.loadtxt(os.path.join(raw_root, VARIANTS[variant][0], fname))
    N2, N = W.shape
    W0 = W[:N, :]                      # first N rows = initial weight matrix
    post, pre = np.nonzero(W0)
    rows = [[int(i), int(j), W0[i, j]] for i, j in zip(post, pre)]
    description = f"""
Data for figure adjacency_matrix.jpeg (initial adjacency and weight matrices).
Source: {VARIANTS[variant][0]}/{fname}, first N rows (= initial weight matrix A*W).
Only nonzero entries are listed; all other entries of the N x N matrix are 0.
N = {N} neurons; neurons 0..{N // 2 - 1} form population 1, {N // 2}..{N - 1} population 2.
Columns: post = postsynaptic neuron index (matrix row), pre = presynaptic neuron index
(matrix column), weight = synaptic weight W[post][pre].
"""
    write_csv(os.path.join(out_dir, "adjacency_matrix.csv"), description, ["post", "pre", "weight"], rows)


# ---------------------------------------------------------------------------------
# Figures: spontaneous_maps_old_vs_new_SP.jpeg, spontaneous_map_comparison_old_vs_new_SP.jpeg
# ---------------------------------------------------------------------------------
def late_time_order_parameters(raw_root, key, Wmean, beta, p_cross):
    data = np.loadtxt(os.path.join(raw_root, VARIANTS[key][0], spont_filename(Wmean, beta, p_cross)))
    last = len(data)
    return (np.mean(data[last - AVERAGE_OVER_ROWS:last, 0]),    # population 1
            np.mean(data[last - AVERAGE_OVER_ROWS:last, 4]))    # population 2


def extract_spontaneous_maps(raw_root, out_dir):
    p_cross = 0.1
    rows = []
    for key in ["HebHom3", "HebHom2", "Original1", "Original"]:
        for w in MEAN_INITIAL_W:
            for p in MEAN_INITIAL_NDD:
                op1, op2 = late_time_order_parameters(raw_root, key, w, p, p_cross)
                rows.append(variant_fields(key) + ["%.1f" % w, "%.3f" % p, "%.3f" % p_cross, op1, op2])
    description = f"""
Data for figure spontaneous_maps_old_vs_new_SP.jpeg.
Source: STDP+SP(Wmean=..,beta=..,p_cross_pop0={p_cross:.3f},cross_pop_delay={CROSS_POP_DELAY:.1f},same_pop_delay={SAME_POP_DELAY:.1f})1.dat
in each variant's raw data folder, for every (Wmean, beta) on the grid.
order_par_pop1 / order_par_pop2 = Kuramoto order parameter of population 1 / 2 (columns 0 / 4 of the
raw file), averaged over the last {AVERAGE_OVER_ROWS} saved rows (last {AVERAGE_OVER_H} simulated hours).
variant: Original / Original1 = original SP model, HebHom2 / HebHom3 = extended SP model;
r_hom_over_r_hebb = ratio of homeostatic to Hebbian SP rates.
Wmean = initial intra-pop mean synaptic weight, beta = initial intra-pop node degree density,
p_cross_pop0 = initial inter-pop node degree density.
"""
    header = VARIANT_COLUMNS + ["Wmean", "beta", "p_cross_pop0", "order_par_pop1", "order_par_pop2"]
    write_csv(os.path.join(out_dir, "spontaneous_maps_old_vs_new_SP.csv"), description, header, rows)


def extract_spontaneous_map_comparison(raw_root, out_dir):
    rows = []
    for key in ["Original1", "Original", "HebHom3", "HebHom2"]:
        for pc in P_CROSS_POP:
            for w in MEAN_INITIAL_W:
                for p in MEAN_INITIAL_NDD:
                    op1, _ = late_time_order_parameters(raw_root, key, w, p, pc)
                    rows.append(variant_fields(key) + ["%.1f" % w, "%.3f" % p, "%.3f" % pc, op1])
    description = f"""
Data for figure spontaneous_map_comparison_old_vs_new_SP.jpeg.
Source: STDP+SP(Wmean=..,beta=..,p_cross_pop0=..,cross_pop_delay={CROSS_POP_DELAY:.1f},same_pop_delay={SAME_POP_DELAY:.1f})1.dat
in each variant's raw data folder, for every (p_cross_pop0, Wmean, beta).
order_par_pop1 = Kuramoto order parameter of population 1 (column 0 of the raw file), averaged over
the last {AVERAGE_OVER_ROWS} saved rows (last {AVERAGE_OVER_H} simulated hours).
variant: Original / Original1 = original SP model, HebHom2 / HebHom3 = extended SP model;
r_hom_over_r_hebb = ratio of homeostatic to Hebbian SP rates.
Wmean = initial intra-pop mean synaptic weight, beta = initial intra-pop node degree density,
p_cross_pop0 = initial inter-pop node degree density.
"""
    header = VARIANT_COLUMNS + ["Wmean", "beta", "p_cross_pop0", "order_par_pop1"]
    write_csv(os.path.join(out_dir, "spontaneous_map_comparison_old_vs_new_SP.csv"), description, header, rows)


# ---------------------------------------------------------------------------------
# Figure: correlation_intrapop_sync_old_vs_new_SP.jpeg
# ---------------------------------------------------------------------------------
def normalized_same_pop_histograms(path):
    """Bin edges and per-save-event same-pop histograms normalized to relative frequency."""
    raw = np.loadtxt(path)
    bin_edges = raw[0, :]
    C = raw[1::2, :-1].astype(float)        # same-pop counts, trailing 0 dropped
    totals = C.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return bin_edges, C / totals


def extract_correlation_intrapop_sync(raw_root, out_dir):
    Wmean = 0.8
    rows = []
    edges_ref = None
    for key in ["HebHom3", "HebHom2", "Original1", "Original"]:
        for pc in P_CROSS_POP:
            path = os.path.join(raw_root, VARIANTS[key][0], spont_corr_dist_filename(Wmean, 0.1, pc))
            bin_edges, C_same = normalized_same_pop_histograms(path)
            if edges_ref is None:
                edges_ref = bin_edges
            assert np.array_equal(bin_edges, edges_ref)
            for k, hist in enumerate(C_same):
                t0 = k * SAVE_INTERVAL_MIN / 60
                t1 = (k + 1) * SAVE_INTERVAL_MIN / 60
                rows.append(variant_fields(key) + ["%.1f" % Wmean, "%.3f" % 0.1, "%.3f" % pc, k, t0, t1] + list(hist))
    description = f"""
Data for figure correlation_intrapop_sync_old_vs_new_SP.jpeg (full simulated run; the figure shows
the first 5 hours).
Source: Correlation_dist_pRVS(Wmean={Wmean:.1f},beta=0.100,p_cross_pop0=..,cross_pop_delay={CROSS_POP_DELAY:.1f},same_pop_delay={SAME_POP_DELAY:.1f})1.dat
in each variant's raw data folder.
One row per saved histogram (every {SAVE_INTERVAL_MIN} simulated minutes). Bin columns '[lo,hi)' hold
the distribution of pairwise spike correlations between neurons of the same population, normalized
to relative frequency within the row. The histogram with index k is drawn on the time interval
[time_start_h, time_end_h] = [k, k+1] * {SAVE_INTERVAL_MIN} min.
variant: Original / Original1 = original SP model, HebHom2 / HebHom3 = extended SP model;
r_hom_over_r_hebb = ratio of homeostatic to Hebbian SP rates.
"""
    header = VARIANT_COLUMNS + ["Wmean", "beta", "p_cross_pop0", "histogram_index", "time_start_h", "time_end_h"] \
        + bin_columns(edges_ref)
    write_csv(os.path.join(out_dir, "correlation_intrapop_sync_old_vs_new_SP.csv"), description, header, rows)


# ---------------------------------------------------------------------------------
# Figures: spontaneous_sync_traces_old_vs_new_SP.jpeg, spontaneous_desync_traces_old_vs_new_SP.jpeg
# ---------------------------------------------------------------------------------
def extract_traces(raw_root, out_dir, Wmean, figure_name, state):
    rows = []
    for key in ["HebHom3", "HebHom2", "Original", "Original1"]:
        for pc in P_CROSS_POP:
            d = np.loadtxt(os.path.join(raw_root, VARIANTS[key][0], spont_filename(Wmean, 0.1, pc)))
            for r in range(d.shape[0]):
                rows.append(variant_fields(key) + ["%.1f" % Wmean, "%.3f" % 0.1, "%.3f" % pc, r,
                                                   r * SAVE_INTERVAL_MIN / 60]
                            + [d[r, c] for c in TRACE_COLUMNS])
    description = f"""
Data for figure {figure_name}.jpeg (time evolution during {state}).
Source: STDP+SP(Wmean={Wmean:.1f},beta=0.100,p_cross_pop0=..,cross_pop_delay={CROSS_POP_DELAY:.1f},same_pop_delay={SAME_POP_DELAY:.1f})1.dat
in each variant's raw data folder.
One row per saved row of the raw file: row_index 0 is the end of the relaxation phase (t = 0), then
one row every {SAVE_INTERVAL_MIN} simulated minutes; time_h = row_index * {SAVE_INTERVAL_MIN} / 60.
Measure columns (raw file column in brackets): odpr_1 [0] = order parameter of population 1,
fi_mean_1 [1] = mean firing rate of population 1 (Hz), Avg_W_1 [2] = mean intra-pop synaptic weight,
node_degree_density_1 [3] = intra-pop node degree density, Avg_W_1to2 [9] / Avg_W_2to1 [11] = mean
inter-pop weight 1->2 / 2->1, node_degree_density_1to2 [10] / node_degree_density_2to1 [12] =
inter-pop node degree density 1->2 / 2->1.
variant: Original / Original1 = original SP model, HebHom2 / HebHom3 = extended SP model;
r_hom_over_r_hebb = ratio of homeostatic to Hebbian SP rates.
"""
    header = VARIANT_COLUMNS + ["Wmean", "beta", "p_cross_pop0", "row_index", "time_h"] + list(TRACE_COLUMNS.values())
    write_csv(os.path.join(out_dir, figure_name + ".csv"), description, header, rows)


# ---------------------------------------------------------------------------------
# Figure: ndd_for_varying_td_intra_and_corr_dep_add_parameters.jpeg
# ---------------------------------------------------------------------------------
CORR_STUDY_DIR = "Data_corr_study"
CORR_STUDY_TRIALS = list(range(1, 11))
CORR_STUDY_DELAYS = [1, 3, 5]
CORR_STUDY_CONDITIONS = [("ON", 0.5, 0.6), ("ON", 0.3, 0.4), ("OFF", 0.5, 0.6)]
CORR_STUDY_CONDITION_COLUMNS = ["corr_dep_add", "corr_th", "delta_corr", "same_pop_delay", "n_trials"]


def extract_corr_study(raw_root, out_dir):
    figure_name = "ndd_for_varying_td_intra_and_corr_dep_add_parameters"
    n = len(CORR_STUDY_TRIALS)

    # Panels A-C: intra-pop node degree density, mean and std over trials
    rows = []
    for cda, cth0, dcorr in CORR_STUDY_CONDITIONS:
        for spd in CORR_STUDY_DELAYS:
            per_trial = [np.loadtxt(os.path.join(raw_root, CORR_STUDY_DIR,
                                                 corr_study_filename(cda, cth0, dcorr, spd, t)))
                         for t in CORR_STUDY_TRIALS]
            time_hr = per_trial[0][:, 15] * TW_MS / 3.6e6
            ndd = np.array([d[:, 3] for d in per_trial])
            mean, std = ndd.mean(axis=0), ndd.std(axis=0)
            cond = [cda, "%.1f" % cth0, "%.1f" % dcorr, "%.1f" % spd, n]
            rows += [cond + [t, m, s] for t, m, s in zip(time_hr, mean, std)]
    description = f"""
Data for figure {figure_name}.jpeg, panels A-C.
Source: {CORR_STUDY_DIR}/STDP+SP(corr-dep-add-..,corr_th=..,delta_corr=..,same_pop_delay=..)<trial>.dat,
trials {CORR_STUDY_TRIALS[0]}-{CORR_STUDY_TRIALS[-1]}.
ndd_intra_mean / ndd_intra_std = mean / standard deviation (numpy std, ddof=0) across the n_trials
trials of the intra-pop node degree density of population 1 (column 3 of the raw file).
time_h = iSPupdate (column 15) * Tw / 3.6e6 h, with Tw = {TW_MS:g} ms; row with time_h = 0 is the end of
the relaxation phase.
corr_dep_add = ON: extended SP model (correlation-dependent addition with threshold corr_th and width
delta_corr); OFF: original SP model (corr_th, delta_corr have no effect). same_pop_delay = intra-pop
synaptic delay (ms).
"""
    header = CORR_STUDY_CONDITION_COLUMNS + ["time_h", "ndd_intra_mean", "ndd_intra_std"]
    write_csv(os.path.join(out_dir, figure_name + "_ndd.csv"), description, header, rows)

    # Panels D-L: intra-pop correlation distribution pooled over trials
    rows = []
    edges_ref = None
    for cda, cth0, dcorr in CORR_STUDY_CONDITIONS:
        for spd in CORR_STUDY_DELAYS:
            pooled_same = 0.0
            for t in CORR_STUDY_TRIALS:
                raw = np.loadtxt(os.path.join(raw_root, CORR_STUDY_DIR,
                                              corr_study_corr_dist_filename(cda, cth0, dcorr, spd, t)))
                bin_edges = raw[0, :]
                pooled_same = pooled_same + raw[1::2, :-1]
            pooled_same = pooled_same / n
            totals = pooled_same.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1
            C_intra = pooled_same / totals
            if edges_ref is None:
                edges_ref = bin_edges
            assert np.array_equal(bin_edges, edges_ref)
            cond = [cda, "%.1f" % cth0, "%.1f" % dcorr, "%.1f" % spd, n]
            for k, hist in enumerate(C_intra):
                t0 = k * SAVE_INTERVAL_MIN / 60.0
                t1 = (k + 1) * SAVE_INTERVAL_MIN / 60.0
                rows.append(cond + [k, t0, t1] + list(hist))
    description = f"""
Data for figure {figure_name}.jpeg, panels D-L (full simulated run; the figure shows the first 5 hours).
Source: {CORR_STUDY_DIR}/Correlation_dist_pRVS(corr-dep-add-..,corr_th=..,delta_corr=..,same_pop_delay=..)<trial>.dat,
trials {CORR_STUDY_TRIALS[0]}-{CORR_STUDY_TRIALS[-1]}.
One row per saved histogram (every {SAVE_INTERVAL_MIN} simulated minutes). Bin columns '[lo,hi)' hold the
distribution of pairwise spike correlations between neurons of the same population: counts are
averaged over the n_trials trials and then normalized to relative frequency within the row.
The histogram with index k is drawn on the time interval [time_start_h, time_end_h] = [k, k+1] * {SAVE_INTERVAL_MIN} min.
corr_dep_add = ON: extended SP model; OFF: original SP model. same_pop_delay = intra-pop synaptic delay (ms).
"""
    header = CORR_STUDY_CONDITION_COLUMNS + ["histogram_index", "time_start_h", "time_end_h"] + bin_columns(edges_ref)
    write_csv(os.path.join(out_dir, figure_name + "_corr_dist.csv"), description, header, rows)


# ---------------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--raw-root", default=os.path.join(here, ".."),
                        help="folder containing the raw data folders (default: parent of this script)")
    parser.add_argument("--out-dir", default=os.path.join(here, "figure_data"),
                        help="output folder for the CSV files (default: figure_data/ next to this script)")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    extract_adjacency_matrix(args.raw_root, args.out_dir)
    extract_spontaneous_maps(args.raw_root, args.out_dir)
    extract_spontaneous_map_comparison(args.raw_root, args.out_dir)
    extract_correlation_intrapop_sync(args.raw_root, args.out_dir)
    extract_traces(args.raw_root, args.out_dir, 0.8, "spontaneous_sync_traces_old_vs_new_SP", "synchronization")
    extract_traces(args.raw_root, args.out_dir, 0.2, "spontaneous_desync_traces_old_vs_new_SP", "desynchronization")
    extract_corr_study(args.raw_root, args.out_dir)


if __name__ == "__main__":
    main()
