# Two-population neuronal network model with combined Hebbian–homeostatic structural plasticity

Code and figure data for the article

> **Self-organized synchronization from combined Hebbian–Homeostatic structural plasticity: a two-population neuronal network model**
> Kanishk Chauhan and Peter A. Tass
> *Frontiers in Network Physiology* (under review)

The repository contains the network simulation and everything needed to regenerate the figures.
The raw simulation output is **not** included (it is too large). Instead, the data plotted in each
figure were extracted from the raw output into small CSV files in `figure_data/`.

| File | Purpose |
|---|---|
| `two-pop-neural-net-homeo-hebb-plasticity.cc` | The simulation. A single-file C++14 program that simulates two coupled populations of leaky integrate-and-fire neurons with spike-timing-dependent plasticity (STDP) and structural plasticity (SP), and writes time series of network measures. It produced all simulation data in the article. |
| `plot_results.ipynb` | Generates all figures from `figure_data/` and saves them to `figures/`. |
| `figure_data/` | One CSV file per data-driven figure, containing exactly the values that are plotted. |
| `extract_figure_data.py` | The script that created `figure_data/` from the raw simulation output. It is included to document how every value was obtained; it cannot be run without the raw data. |

---

**Model variants used in the figures:**

| Variant (figure label) | Key in `figure_data/` | `corr_dep_add` (argument 1) | `rate_hom` (in the code) | `rate_hebb` (in the code) |
|---|---|---|---|---|
| Original SP model, r<sub>Hom</sub> = r<sub>Hebb</sub> | `Original` | `OFF` | `1e-3` | `1e-3` |
| Original SP model, r<sub>Hom</sub> = r<sub>Hebb</sub>/10 | `Original1` | `OFF` | `1e-4` | `1e-3` |
| Extended SP model, r<sub>Hom</sub> = r<sub>Hebb</sub> | `HebHom2` | `ON` | `1e-3` | `1e-3` |
| Extended SP model, r<sub>Hom</sub> = r<sub>Hebb</sub>/10 | `HebHom3` | `ON` | `1e-4` | `1e-3` |

### Simulation protocol

1. **Relaxation (300 s, no plasticity).** The network runs uncoupled (κ = 0) for 150 s and then
   coupled (κ = 8) for 150 s.
2. **Plasticity phase (8 simulated days).** STDP acts continuously. After every 6-s window the
   correlations are updated and one SP step is applied.
3. **Recording.** Network measures are written at the end of the relaxation phase and then every
   10 simulated minutes.

The degree of synchrony of each population is the Kuramoto order parameter. It is computed from
phases interpolated between spikes, sampled every 100 ms and averaged over each 6-s window.

---

## Running the simulation

Compile with

```bash
g++ -std=c++14 -O3 two-pop-neural-net-homeo-hebb-plasticity.cc -o two_pop_sim
```

Run with eight arguments:

```bash
./two_pop_sim <corr_dep_add> <beta> <p_cross> <W_mean> <corr_th> <delta_corr> <same_pop_delay> <trial>
# example: extended SP model, beta = 0.1, p_cross = 0.1, W_mean = 0.8, c_th = 0.5, Δc = 0.6, 3 ms, trial 1
./two_pop_sim ON 0.1 0.1 0.8 0.5 0.6 3 1
```

| # | Argument | Meaning |
|---|---|---|
| 1 | `corr_dep_add` | `ON`: extended SP model (with correlation-dependent addition). `OFF`: original SP model. |
| 2 | `beta` | Initial intra-population node degree density |
| 3 | `p_cross` | Initial inter-population node degree density |
| 4 | `W_mean` | Initial mean synaptic weight |
| 5 | `corr_th` | Correlation threshold c<sub>th</sub> of correlation-dependent addition (no effect with `OFF`) |
| 6 | `delta_corr` | Width Δc of correlation-dependent addition (no effect with `OFF`) |
| 7 | `same_pop_delay` | Intra-population synaptic delay in ms (integer) |
| 8 | `trial` | Realization number. It seeds the random number generators for the initial network, the neuron parameters and the noise. |

All other parameters, including `rate_hom`, `rate_hebb` and the inter-population delay
`cross_pop_delay`, are constants in `main()`.

**Output.** The program writes to `../Data/` relative to the directory it is launched from. **This
folder must exist before the run**, because the program does not create it. The files are:

| File | Content |
|---|---|
| `STDP+SP(...)<trial>.dat` | One row per save (see above), 16 columns: `odpr_1 fi_mean_1 Avg_W_1 ndd_1 odpr_2 fi_mean_2 Avg_W_2 ndd_2 odpr_12 Avg_W_1to2 ndd_1to2 Avg_W_2to1 ndd_2to1 pruned added iSPupdate`. `odpr`: order parameter; `fi_mean`: mean firing rate (Hz); `Avg_W`: mean weight; `ndd`: node degree density; `1to2`/`2to1`: inter-population connections from population 1 to 2 and from 2 to 1; `pruned`/`added`: SP changes in the last step; `iSPupdate`: number of 6-s plasticity windows so far. Intra-population measures exclude inter-population connections. |
| `W_STDP+SP(...)<trial>.dat` | Initial and final N × N weight matrices, stacked (row = postsynaptic neuron) |
| `Correlation_dist(...)<trial>.dat` | Histograms of c<sub>ij</sub> at every save of the plasticity phase: one row of 101 bin edges on [−1, 1], then per save one row of intra-population counts and one row of inter-population counts |
| `FR_STDP+SP(...)<trial>.dat` | Firing rates of all neurons: after the uncoupled part of the relaxation phase, and at the end |
| `Per_neuron(...)<trial>.dat` | Per-neuron mean in/out weights and node degree densities (intra and inter), 8 rows per save |
| `Steady_state_STDP+SP(...)<trial>.dat` | Complete final state of the network (positions, neuron states, traces, adjacency, weight and delay matrices) |

> **Note on file names.** The raw files used for the article were named slightly differently from
> the names the program now writes. For example, the spontaneous-activity runs were named
> `STDP+SP(Wmean=0.8,beta=0.100,p_cross_pop0=0.100,cross_pop_delay=10.0,same_pop_delay=3.0)1.dat`,
> and the correlation histograms `Correlation_dist_pRVS(...)`. The names the program writes are
> set in `setup_files()`. They do not contain `same_pop_delay`, `corr_th` or `delta_corr`, so runs
> that differ only in these arguments overwrite each other. If you run such parameter sweeps, add
> these parameters to the names in `setup_files()`. To re-extract figure data from your own runs,
> adapt the `*_filename()` helpers in `extract_figure_data.py` to your names.

---

## Reproducing the figures

**Requirements:** Python 3 with numpy, pandas, matplotlib, networkx and Jupyter, plus a LaTeX
installation, because the figures use matplotlib's `text.usetex`. Tested with Python 3.11.0,
numpy 1.26.4, pandas 2.3.3, matplotlib 3.10.8 and networkx 3.2.1.

Open `plot_results.ipynb` in this folder and run all cells **from top to bottom**. Some cells change
global matplotlib settings that later cells inherit, so running cells out of order changes line
widths. The figures are written to `figures/`.

Each CSV file in `figure_data/` starts with lines beginning with `#` that describe its source files,
columns and any averaging. Read the files with, for example,
`pandas.read_csv(path, comment="#")`. Histogram columns are named by their bin edges, `[lo,hi)`.

### Figure data and simulation settings

Unless stated otherwise, all simulations used:

- inter-population delay `cross_pop_delay` = 10 ms (fixed in the code)
- intra-population delay 3 ms (argument 7)
- `corr_th` = 0.5 and `delta_corr` = 0.6 (arguments 5–6)
- trial 1 (argument 8)
- the SP variant set via `corr_dep_add` (argument 1) and `rate_hom` (in the code), as in the variant
  table above

#### `Network.jpeg` — network structure
- **Data:** none. The notebook builds an illustrative network in Python using the same placement and
  connection rules as the simulation.
- **Illustration settings:** β = 0.07 and p<sub>cross</sub> = 0.002, chosen to keep the drawing
  readable. It is not a network from the simulations.

#### `adjacency_matrix.jpeg` — initial adjacency and weight matrices
- **Data:** `adjacency_matrix.csv`, the nonzero entries (`post`, `pre`, `weight`) of the initial
  weight matrix.
- **Simulation:** `ON 0.1 0.1 0.8 0.5 0.6 3 1` with `rate_hom = 1e-4`.
- The initial matrices depend only on β, p<sub>cross</sub>, W<sub>mean</sub> and the trial, not on
  the SP variant.

#### `structural_plasticity_figure.jpeg` — SP probability functions
- **Data:** none. The curves are computed analytically in the notebook with the SP parameters above
  (c<sub>th</sub> = 0.5, Δc = 0.6).

#### `spontaneous_maps_old_vs_new_SP.jpeg` — regions of synchrony and desynchrony
- **Data:** `spontaneous_maps_old_vs_new_SP.csv`. It holds the order parameters of populations 1
  and 2, averaged over the last 5 h of each run.
- **Simulations:** all four variants, on a grid of `W_mean` ∈ {0.1, 0.2, …, 0.9} × `beta` ∈ {0.05,
  0.10, …, 0.30}, with `p_cross` = 0.1.
- **Plot:** a region counts as synchronized where the order parameter exceeds 0.5.

#### `spontaneous_map_comparison_old_vs_new_SP.jpeg` — dependence on the inter-population node degree density
- **Data:** `spontaneous_map_comparison_old_vs_new_SP.csv`, the order parameter of population 1
  averaged over the last 5 h.
- **Simulations:** as for the previous figure, for `p_cross` ∈ {0.05, 0.1, 0.15}.

#### `correlation_intrapop_sync_old_vs_new_SP.jpeg` — intra-population spike-correlation distribution over time
- **Data:** `correlation_intrapop_sync_old_vs_new_SP.csv`. It holds the intra-population correlation
  histograms, normalized to relative frequency, every 10 min over the full run. The figure shows the
  first 5 h.
- **Simulations:** all four variants, with `W_mean` = 0.8, `beta` = 0.1 and `p_cross` ∈ {0.05, 0.1,
  0.15}.

#### `spontaneous_sync_traces_old_vs_new_SP.jpeg` — time evolution towards synchrony
- **Data:** `spontaneous_sync_traces_old_vs_new_SP.csv`, the plotted network measures every 10 min
  over the 8-day run:
  - order parameter, mean firing rate, mean weight and node degree density of population 1
  - inter-population mean weights and node degree densities in both directions
- **Simulations:** all four variants, with `W_mean` = 0.8, `beta` = 0.1 and `p_cross` ∈ {0.05, 0.1,
  0.15}.

#### `spontaneous_desync_traces_old_vs_new_SP.jpeg` — time evolution towards desynchrony
- **Data:** `spontaneous_desync_traces_old_vs_new_SP.csv`, with the same columns as the previous figure.
- **Simulations:** as for the previous figure, but with `W_mean` = 0.2.

#### `ndd_for_varying_td_intra_and_corr_dep_add_parameters.jpeg` — role of the intra-population delay and of the correlation-dependent addition
- **Data:**
  - `…_ndd.csv`: the mean and standard deviation over trials 1–10 of the intra-population node
    degree density of population 1 (panels A–C)
  - `…_corr_dist.csv`: the intra-population correlation histograms averaged over the 10 trials and
    normalized to relative frequency, over the full run; the figure shows the first 5 h (panels D–L)
- **Simulations:** `rate_hom = 1e-4`, `W_mean` = 0.8, `beta` = 0.1, `p_cross` = 0.1, trials 1–10.
  Three conditions, each run with intra-population delays `same_pop_delay` ∈ {1, 3, 5} ms:
  - `ON 0.1 0.1 0.8 0.5 0.6 <delay> <trial>`
  - `ON 0.1 0.1 0.8 0.3 0.4 <delay> <trial>`
  - `OFF 0.1 0.1 0.8 0.5 0.6 <delay> <trial>` (with `OFF`, `corr_th` and `delta_corr` have no effect)

---

## Regenerating `figure_data/`

`extract_figure_data.py` reads the raw simulation output and rewrites `figure_data/`. For each
figure it reads the same files and applies the same reduction as the plotting code originally did:
column selection, 5-h averaging, trial averaging and normalization.

```bash
python extract_figure_data.py --raw-root <folder containing the raw data folders>
```

It expects the raw data folder layout used for the article:

- `DataSpontaneous/` (`Original`)
- `DataSpontaneous1/` (`Original1`)
- `DataSpont_HebHom2/` (`HebHom2`)
- `DataSpont_HebHom3/` (`HebHom3`)
- `Data_corr_study/`

These folders are not part of the repository.

## License

This repository is released under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE.md).
You may use, modify and share the code and data, including modified versions, for noncommercial
purposes such as academic research and teaching, provided you give appropriate credit. Commercial
use is not permitted.

## Citation

If you use this code or data, please cite the article above. Citation details will be added once it
is published.
