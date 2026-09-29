/*
 *        File: two-pop-neural-net-homeo-hebb-plasticity.cc
 *      Author: Kanishk Chauhan
 *        Date: January 2026
 * Description: Simulates a two-population plastic neuronal network with LIF model neurons and STDP accompanied by our model of structural plasticity. 
 *              Structural plasticity combines the original homeostatic + weight-dependent plasticity (Chauhan et al., PLoS Comp Biol 2024) with a Hebbian-type correlation-dependent plasticity.                         
 */

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <complex>
#include <random>
#include <chrono>
#include <map>
#include <set>

using namespace std;

// -------- Random number generator (Adapted from Numerical Recipes Book) ----------------------------
struct Ran {
    typedef unsigned int long long Ullong;
    typedef unsigned int Uint;
    Ullong u,v,w;
    Ran(Ullong j) : v(4101842887655102017LL), w(1) {
        u = j ^ v; int64();
        v = u; int64();
        w = v; int64();
    }
    inline Ullong int64() {
        u = u * 2862933555777941757LL + 7046029254386353087LL;
        v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
        w = 4294957665U*(w & 0xffffffff) + (w >> 32);
        Ullong x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
        return (x + v) ^ w;
    }
    inline double doub() { return 5.42101086242752217E-20 * int64(); }
    inline Uint int32() { return (Uint)int64(); }
};

struct Normaldev_BM : Ran {
    typedef double Doub;
    typedef unsigned int long long Ullong;
    Doub mu,sig;
    Doub storedval;
    Normaldev_BM(Doub mmu, Doub ssig, Ullong i)
        : Ran(i), mu(mmu), sig(ssig), storedval(0.) {}
    Doub dev() {
        Doub v1,v2,rsq,fac;
        if (storedval == 0.) {
            do {
                v1=2.0*doub()-1.0;
                v2=2.0*doub()-1.0;
                rsq=v1*v1+v2*v2;
            } while (rsq >= 1.0 || rsq == 0.0);
            fac=sqrt(-2.0*log(rsq)/rsq);
            storedval = v1*fac;
            return mu + sig*v2*fac;
        } else {
            fac = storedval;
            storedval = 0.;
            return mu + sig*fac;
        }
    }
};
// ---------------------------------------------------------------------------------

// Constants
const int m     = 32;
const int N     = m * m;    // total neurons
const int N_pop = N / 2;    // neurons per population

// -------- Function Prototypes ----------------------------------------------------
vector<vector<double>> Adjacency_Matrix(int N, double p);
vector<vector<double>> Dist_Dep_Adjacency_Matrix(double p, const vector<vector<double>>& Zd, int number);
void add_cross_population_connections(vector<vector<double>>& A, double p_cross, int number,
                                      const double xi[], double network_dimensions);
void InF(double dt, double gleak[], double gsyn[], double gnoise[], double noise[],
         double V[], double Vth[], double Istim[],
         double Vrest, double Vsyn, double Vth_rest, double Tsyn, double Tth,
         double C0, double knoise, int N);
// Legacy STDP function kept for reference 
void fastSTDP(int N, double dt, double ed, double ep,
              double train_pre[], double train_post[], double trace_pre[], double trace_post[],
              double inlinks[], double outlinks[],
              vector<vector<double>>& W, const vector<vector<int>>& iLinks,
              const vector<vector<int>>& oLinks,
              const vector<int>& pre_spikers, const vector<int>& post_spikers,
              const bool is_pre_spiker[], const bool is_post_spiker[]);
// STDP using per-delay-group traces and arrival lists
void delayAwareFastSTDP(int N, double dt, double ed, double ep,
                        double trace_post[],
                        const vector<vector<double>>& trace_pre_d,
                        const vector<vector<int>>& arrivals,
                        const vector<int>& post_spikers,
                        vector<vector<uint8_t>>& is_arriving,
                        const bool is_post_spiker[],
                        const vector<vector<vector<int>>>& iLinks_by_delay,
                        const vector<vector<vector<int>>>& oLinks_by_delay,
                        vector<vector<double>>& W, double dt_inv, int num_delays);
void SP(Ran& random, vector<vector<double>>& W, vector<vector<double>>& A, vector<vector<double>>& A0, const vector<vector<double>>& Zd, double fi[], double inlinks[],
        double Tw, double Wmin, double rate_hom, double rate_hebb, double c, double f1, double f2, double nu, int& pruned, int& added, double& n_con, vector<vector<int>>& iLinks,
        const double xi[], double network_dimensions, double lower_bound_cross_pop, double lower_bound_same_pop, const vector<vector<double>>& Corr, double corr_th0, double nu_corr, string corr_dep_add);
void update_corr_trace(vector<vector<double>>& corr_trace,
                       const vector<vector<double>>& spike_times_window,
                       const double last_spike_before_window[],
                       const vector<vector<int>>& delay_steps_matrix,
                       double Tslow, double dt, int N);
void compute_correlation(const vector<vector<double>>& corr_trace,
                        const double fi[], const double spikeTimes[],
                        double Tslow, double t_now, int N,
                        vector<vector<double>>& Corr);
void compute_correlation_distribution(const vector<vector<double>>& Corr, int N,
                                      int n_bins, vector<double>& bin_edges,
                                      vector<int>& same_pop_counts,
                                      vector<int>& cross_pop_counts);
void save_correlation_distribution(const vector<vector<double>>& Corr, int N,
                                   ofstream& corr_dist_file, int n_bins);
void setup_files(int trial_number, ofstream& data, ofstream& W_file, ofstream& A_file, ofstream& FR_file, ofstream& steadystate, double same_pop_delay, 
                 ofstream& per_neuron_w_ndd_file, ofstream& corr_dist_file, string corr_dep_add, double corr_th0, double delta_corr, double W_mean, double p, double p_cross);
void initialize_network_geometry(Ran& random, double network_dimensions,
                                  double xi[], double yi[],
                                  vector<vector<double>>& Zd,
                                  vector<vector<double>>& Dd);
void initialize_neuron_states(Ran& random, Normaldev_BM& normal_random,
                               double gleak[], double V[], double Vth[], double fi[],
                               double gsyn[], double gnoise[], double flag[],
                               double spikeTimes[], double spikeTimes0[], double train_post[], double trace_post[],
                               double Istim[], int counts2[],
                               double Vrest, double Vreset, double Vth_rest,
                               double gleak0, double gleak_std);
void update_link_lists(const vector<vector<double>>& A, vector<vector<int>>& iLinks,
                       vector<vector<int>>& oLinks, double inlinks[], double outlinks[]);

void compute_delay_matrix(const vector<vector<double>>& Dd,
                           vector<vector<int>>& delay_steps_matrix,
                           vector<int>& unique_delays, map<int,int>& delay_to_index,
                           int& max_delay_steps,
                           double min_syn_delay, double delay_slope, double cross_pop_delay,
                           double dt, int N);
void build_delay_grouped_links(const vector<vector<double>>& A,
                                const vector<vector<int>>& delay_steps_matrix,
                                const vector<int>& unique_delays,
                                const map<int,int>& delay_to_index, int N,
                                vector<vector<vector<int>>>& iLinks_by_delay,
                                vector<vector<vector<int>>>& oLinks_by_delay);
void calculate_mean_weight_fr_ndd(const vector<vector<double>>& A,
    const vector<vector<double>>& W, double fi[],
    double& n_con,   double& Avg_W,   double& fi_mean,   double& node_degree_density,
    double& n_con_1, double& Avg_W_1, double& fi_mean_1, double& node_degree_density_1,
    double& n_con_2, double& Avg_W_2, double& fi_mean_2, double& node_degree_density_2,
    double& n_con_1to2, double& Avg_W_1to2, double& node_degree_density_1to2,
    double& n_con_2to1, double& Avg_W_2to1, double& node_degree_density_2to1,
    bool include_cross);
void calculate_per_neuron_weight_ndd(const vector<vector<double>>& A,
    const vector<vector<double>>& W,
    double avg_in_same_pop_w_each_neuron[],  double avg_out_same_pop_w_each_neuron[],
    double avg_in_cross_pop_w_each_neuron[], double avg_out_cross_pop_w_each_neuron[],
    double avg_in_same_pop_ndd_each_neuron[],  double avg_out_same_pop_ndd_each_neuron[],
    double avg_in_cross_pop_ndd_each_neuron[], double avg_out_cross_pop_ndd_each_neuron[]);
void save_per_neuron(const vector<vector<double>>& A, const vector<vector<double>>& W,
    double avg_in_same_pop_w_each_neuron[],  double avg_out_same_pop_w_each_neuron[],
    double avg_in_cross_pop_w_each_neuron[], double avg_out_cross_pop_w_each_neuron[],
    double avg_in_same_pop_ndd_each_neuron[],  double avg_out_same_pop_ndd_each_neuron[],
    double avg_in_cross_pop_ndd_each_neuron[], double avg_out_cross_pop_ndd_each_neuron[],
    ofstream& per_neuron_w_ndd_file);
void run_simulation_window(
    double base_time, long long base_time_step, bool plasticity_on, double k, double pi, Ran& random,
    double dt, int nt, double Vrest, double Vsyn, double Vth_rest,
    double Tsyn, double Tth, double C0, double knoise, double fnoise,
    double tp, double td, double ed, double ep,
    double Tslow, double Tspike, double Vspike, double Vreset, double Vth_spike,
    double gleak[], double gsyn[], double gnoise[], double V[], double Vth[],
    double Istim[], double fi[], double flag[],
    double spikeTimes[], double spikeTimes0[], int counts2[],
    double trace_post[],
    vector<vector<double>>& spike_history,
    vector<vector<double>>& trace_pre_d,
    vector<vector<double>>& W, const vector<vector<double>>& A,
    vector<double>& noise_flat,
    bool is_post_spiker[],
    vector<vector<uint8_t>>& is_arriving,
    const vector<vector<int>>& delay_steps_matrix,
    const vector<int>& unique_delays,
    int max_delay_steps, int num_delays,
    const vector<vector<vector<int>>>& iLinks_by_delay,
    const vector<vector<vector<int>>>& oLinks_by_delay,
    double& odpr_1, double& odpr_2, double& odpr_12,
    vector<vector<double>>& spike_times_window);
void save_final_state(ofstream& steadystate,
                      double xi[], double yi[], double gleak[], double V[], double Vth[],
                      double fi[], double gsyn[], double gnoise[], double flag[],
                      double trace_post[],
                      const vector<vector<double>>& trace_pre_d,
                      const vector<vector<double>>& A, const vector<vector<double>>& W,
                      const vector<vector<int>>& delay_steps_matrix, int N);
void total_run_time(const chrono::high_resolution_clock::time_point& start);

// =================================================================================
// main
// =================================================================================
int main(int argc, char* argv[]) {
    auto simulation_start_time = chrono::high_resolution_clock::now();

    if (argc != 9) {
        cerr << "Incorrect command use! Correct usage: "
             << argv[0] << " <corr_dep_add> <initial intra-pop node degree density> <initial inter-pop node degree density> <initial mean synaptic weight> <corr_threshold> <delta_corr> <min_syn_delay> <simulation_run_number>" << endl;
        return 1;
    }

    // Parameter declarations 
    double Tw, tp, tr, td, dt, dt_inv, rtime0, rtime1, ed, ep, knoise, fnoise;
    double Tth, Tsyn, k, Vsyn, Vrest, Vth_rest, C0, Vreset, seed, network_dimensions;
    double Vspike, Tspike, Vth_spike, Tslow, W_sd, Wmin, rate_hom, rate_hebb, c1, f1, f2, nu;
    double lower_bound_cross_pop, lower_bound_same_pop;
    double corr_th0, nu_corr, epsilon_c, corr_soft_lower_bound, corr_soft_upper_bound, delta_corr;
    double gleak0, gleak_std, max_err, fi_err, Avg_Werr;
    double f0, fstd, eta, p0, fT, delta_f, n_con, Avg_W, node_degree_density, fi_mean;
    double Avg_W0, fi_mean0, a, W_mean, W_mean1, W_mean2, W_mean12, p, p_cross, pi;
    double odpr_1, odpr_2, odpr_12, cross_pop_delay, min_syn_delay, delay_slope;
    double n_con_1, Avg_W_1, fi_mean_1, node_degree_density_1, node_degree_density_1to2;
    double n_con_2, Avg_W_2, fi_mean_2, node_degree_density_2, node_degree_density_2to1;
    double n_con_1to2, Avg_W_1to2, n_con_2to1, Avg_W_2to1;
    int nt, iSPupdate, nSPupdate, pruned, added, num_itrn, trial_number, sp_it, flg, max_delay_steps, num_delays, n_bins;

    // State variable arrays
    double V[N], Vth[N], gleak[N], fi[N], gsyn[N], gnoise[N], Istim[N], flag[N];
    double trace_post[N], inlinks[N], outlinks[N], spikeTimes[N], spikeTimes0[N];
    double train_post[N];           // current-step spike values (used in init, not in STDP)
    double xi[N], yi[N];
    int    counts2[N];

    // O(1) membership lookup — persistent between fastSTDP and the flag-clearing loop
    bool is_post_spiker[N] = {};

    // Parse command-line arguments
    string corr_dep_add = argv[1]; // ON or OFF
    p = atof(argv[2]); // intra-pop node degree density
    p_cross = atof(argv[3]); 
    W_mean = atof(argv[4]);
    corr_th0 = atof(argv[5]);
    delta_corr = atof(argv[6]);
    min_syn_delay = atoi(argv[7]); // in ms (intra-pop synaptic delay)
    trial_number = atoi(argv[8]);
    
    a       = 1.4;
    cross_pop_delay = 10; // in ms (inter-pop synaptic delay)
    W_sd    = 0.05;

    bool include_cross_conn_in_measures = false; // false excludes cross-poplulation connections only from average weight and node degree measure, it still allows those connections to effect neurons.
    network_dimensions = 2.0;

    // Set min_syn_delay=3.0, delay_slope=0.0, cross_pop_delay=3.0 for simplest case
    delay_slope     =  0.0;  // ms/mm: linear delay increase with distance: kept 0 for these simulations, so the intra-pop delay is 3ms for all contacts residing within one population

    seed = 100 * trial_number;
    Ran random(seed);
    Normaldev_BM normal_random(0, 1, seed);

    ofstream data, W_file, A_file, FR_file, steadystate, per_neuron_w_ndd_file, corr_dist_file;
    double avg_in_same_pop_w_each_neuron[N],    avg_out_same_pop_w_each_neuron[N];
    double avg_in_cross_pop_w_each_neuron[N],   avg_out_cross_pop_w_each_neuron[N];
    double avg_in_same_pop_ndd_each_neuron[N],  avg_out_same_pop_ndd_each_neuron[N];
    double avg_in_cross_pop_ndd_each_neuron[N], avg_out_cross_pop_ndd_each_neuron[N];
    setup_files(trial_number, data, W_file, A_file, FR_file, steadystate, min_syn_delay, per_neuron_w_ndd_file, corr_dist_file, corr_dep_add, corr_th0, delta_corr, W_mean, p, p_cross);

    // Time and STDP parameters 
    dt      = 0.1; // in ms
    dt_inv  = 1.0 / dt;
    rtime0  = 300e3;
    Tw      = 6e3; // in ms
    nt      = Tw / dt;
    vector<double> noise_flat(nt * N, 0.0);
    nSPupdate = 8 * 1440 * 60e3 / Tw; // using 24 hours = 1440 minutes
    tp      = 10.0; // in ms
    tr      = 4.0;
    td      = tr * tp;
    Tslow   = 30e3; // in ms
    eta     = 0.02;
    ep      = eta;
    ed      = a * eta / tr;
    pi      = 4 * atan(1);
    max_err = 1e-3;

    // Neuron model parameters 
    C0       = 3.0;
    Vrest    = -38.0;
    Vreset   = -67.0;
    Vsyn     =  0.0;
    Vth_rest = -40.0;
    Vth_spike = 0.0;
    Vspike   = 20.0;
    Tsyn     =  1.0; // in ms
    Tth      =  5.0; // in ms
    Tspike   =  1.0; // in ms
    f0       =  3.0;
    fstd     =  0.3;
    gleak0   = (f0 - 0.9177303000633497) / 125.6691632725962;
    gleak_std = fstd / 125.6691632725962;
    knoise   = 0.06;
    fnoise   = 20.0;

    // Structural plasticity parameters 
    Wmin  = 0.001;
    fT    = 4.5;
    delta_f = 1.0;
    f1    = fT - delta_f / 2.0;
    f2    = fT + delta_f / 2.0;
    c1    = exp(-1.0 / Wmin);
    p0    = 0.01;
    nu    = delta_f / (2.0 * log((1.0 - p0) / p0));
    rate_hebb    = 1e-3; // maximum rate for hebbian SP 
    rate_hom    = 1e-4; // maximum rate for homeostatic SP = 1e-4 
    lower_bound_cross_pop = 0.02;
    lower_bound_same_pop = 0.02;
    epsilon_c = 0.05;
    corr_soft_lower_bound = corr_th0 - delta_corr / 2.0;
    corr_soft_upper_bound = corr_th0 + delta_corr / 2.0;
    nu_corr  = delta_corr / (2.0 * log((1-epsilon_c)/epsilon_c));
    n_bins = 100;

    // ---- Network geometry and distance matrices ----
    // Zd[i][j] = exp(-dist/dst)  for same-population pairs (used for connectivity)
    // Dd[i][j] = raw Euclidean distance in mm (used for delay computation)
    vector<vector<double>> Zd(N, vector<double>(N, 0));
    vector<vector<double>> Dd(N, vector<double>(N, 0));
    initialize_network_geometry(random, network_dimensions, xi, yi, Zd, Dd);

    // ---- Neuron state initialisation ----
    // trace_pre_d (per-delay-group presynaptic traces) initialised to 0 by construction below.
    initialize_neuron_states(random, normal_random, gleak, V, Vth, fi, gsyn, gnoise, flag,
                             spikeTimes, spikeTimes0, train_post, trace_post, Istim, counts2,
                             Vrest, Vreset, Vth_rest, gleak0, gleak_std);

    // ---- Connectivity and weights ----
    // for completely disconnected initial network, uncomment the next three lines and comment out the ones for connected initial network
    // vector<vector<double>> A(N, vector<double>(N, 0));
    // vector<vector<double>> A0 = A;
    // vector<vector<double>> W(N, vector<double>(N, 0));
    // for connected initial network, uncomment the next seven lines and comment out the ones for completely disconnected initial network
    vector<vector<double>> A = Dist_Dep_Adjacency_Matrix(p, Zd, trial_number);
    add_cross_population_connections(A, p_cross, trial_number, xi, network_dimensions);
    vector<vector<double>> A0 = A;
    vector<vector<double>> W(N, vector<double>(N, 0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (A[i][j] == 1) W[i][j] = W_mean + W_sd * (2 * random.doub() - 1);

    // delay matrix (pre-computed for ALL neuron pairs, not just connected ones) 
    // This means SP-added connections automatically inherit the correct delay.
    vector<vector<int>> delay_steps_matrix(N, vector<int>(N, 0));
    vector<int>     unique_delays;
    map<int,int>    delay_to_index;
    compute_delay_matrix(Dd, delay_steps_matrix, unique_delays, delay_to_index,
                         max_delay_steps, min_syn_delay, delay_slope, cross_pop_delay, dt, N);
    num_delays = (int)unique_delays.size();
    cout << "Delay groups: " << num_delays
         << "  max_delay_steps: " << max_delay_steps
         << "  (max delay = " << (max_delay_steps - 1) * dt << " ms)" << endl;

    // ---- spike history buffer ----
    // spike_history[i][j % max_delay_steps] = dt_inv when neuron i fires at step j, else 0.
    vector<vector<double>> spike_history(N, vector<double>(max_delay_steps, 0.0));

    // ---- per-delay-group presynaptic traces ----
    // trace_pre_d[k][i]: presynaptic trace of neuron i for delay group k.
    vector<vector<double>> trace_pre_d(num_delays, vector<double>(N, 0.0));

    // ---- is_arriving[k][i] membership flag ----
    // Cleared after each STDP call; passed into run_simulation_window.
    vector<vector<uint8_t>> is_arriving(num_delays, vector<uint8_t>(N, 0));

    // ---- Flat link lists (kept for SP internals) ----
    vector<vector<int>> iLinks(N, vector<int>(N, 0));
    vector<vector<int>> oLinks(N, vector<int>(N, 0));
    update_link_lists(A, iLinks, oLinks, inlinks, outlinks);
    for (int i = 0; i < N; ++i) {
        iLinks[i].resize((int)inlinks[i]);  iLinks[i].shrink_to_fit();
        oLinks[i].resize((int)outlinks[i]); oLinks[i].shrink_to_fit();
    }

    // ---- delay-grouped link lists ----
    // iLinks_by_delay[post][k] = presynaptic partners of post with delay index k (ascending order)
    // oLinks_by_delay[pre][k]  = postsynaptic partners of pre with delay index k (ascending order)
    vector<vector<vector<int>>> iLinks_by_delay(N, vector<vector<int>>(num_delays));
    vector<vector<vector<int>>> oLinks_by_delay(N, vector<vector<int>>(num_delays));
    build_delay_grouped_links(A, delay_steps_matrix, unique_delays, delay_to_index, N,
                               iLinks_by_delay, oLinks_by_delay);

    vector<vector<double>> corr_trace(N, vector<double>(N, 0.0));
    vector<vector<double>> Corr(N, vector<double>(N, 0.0));
    vector<vector<double>> spike_times_window(N);
    double last_spike_before_window[N] = {};

    calculate_mean_weight_fr_ndd(A, W, fi, n_con, Avg_W, fi_mean, node_degree_density,
                                  n_con_1, Avg_W_1, fi_mean_1, node_degree_density_1,
                                  n_con_2, Avg_W_2, fi_mean_2, node_degree_density_2,
                                  n_con_1to2, Avg_W_1to2, node_degree_density_1to2, n_con_2to1, Avg_W_2to1, node_degree_density_2to1,
                                  include_cross_conn_in_measures);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) W_file << A[i][j] * W[i][j] << " ";
        W_file << endl;
    }

    long long base_time_step = 0;

    // =============================================================================
    // Relaxation phase (no plasticity)
    // =============================================================================
    cout << "--- Relaxation Phase ---" << endl;
    num_itrn = rtime0 / Tw;
    for (int i = 0; i < num_itrn; ++i) {
        k = (i < num_itrn / 2) ? 0.0 : 8.0;
        for (int ii = 0; ii < N; ++ii) spike_times_window[ii].clear();
        run_simulation_window(
            i * Tw, base_time_step, false, k, pi, random, dt, nt,
            Vrest, Vsyn, Vth_rest, Tsyn, Tth, C0, knoise, fnoise,
            tp, td, ed, ep, Tslow, Tspike, Vspike, Vreset, Vth_spike,
            gleak, gsyn, gnoise, V, Vth, Istim, fi, flag,
            spikeTimes, spikeTimes0, counts2, trace_post,
            spike_history, trace_pre_d, W, A, noise_flat,
            is_post_spiker, is_arriving,
            delay_steps_matrix, unique_delays, max_delay_steps, num_delays,
            iLinks_by_delay, oLinks_by_delay, odpr_1, odpr_2, odpr_12,
            spike_times_window);
        base_time_step += nt;

        calculate_mean_weight_fr_ndd(A, W, fi, n_con, Avg_W, fi_mean, node_degree_density,
                                      n_con_1, Avg_W_1, fi_mean_1, node_degree_density_1,
                                      n_con_2, Avg_W_2, fi_mean_2, node_degree_density_2,
                                      n_con_1to2, Avg_W_1to2, node_degree_density_1to2, n_con_2to1, Avg_W_2to1, node_degree_density_2to1,
                                      include_cross_conn_in_measures);
        cout << "Without plasticity | Pop1: " << odpr_1 << " " << fi_mean_1 << " "
             << Avg_W_1 << " " << node_degree_density_1
             << " | Pop2: " << odpr_2 << " " << fi_mean_2 << " "
             << Avg_W_2 << " " << node_degree_density_2
             << " | R12: " << odpr_12 << " Avg_W_1to2: " << Avg_W_1to2
             << " Avg_W_2to1: " << Avg_W_2to1 << endl;

        if (i == num_itrn / 2 - 1){
            for (int ii = 0; ii < N; ++ii) FR_file << fi[ii] << "  ";
            FR_file << endl;
        }
    }

    data << left << setw(15) << odpr_1       << left << setw(15) << fi_mean_1
         << left << setw(15) << Avg_W_1      << left << setw(15) << node_degree_density_1
         << left << setw(15) << odpr_2       << left << setw(15) << fi_mean_2
         << left << setw(15) << Avg_W_2      << left << setw(15) << node_degree_density_2
         << left << setw(15) << odpr_12      << left << setw(15) << Avg_W_1to2
         << left << setw(15) << node_degree_density_1to2  << left << setw(15) << Avg_W_2to1
         << left << setw(15) << node_degree_density_2to1 << left << setw(15) << 0 << left << setw(15) << 0 << left << setw(15) << 0 << endl;
    save_per_neuron(A, W, avg_in_same_pop_w_each_neuron,   avg_out_same_pop_w_each_neuron, avg_in_cross_pop_w_each_neuron,  avg_out_cross_pop_w_each_neuron, avg_in_same_pop_ndd_each_neuron, avg_out_same_pop_ndd_each_neuron, avg_in_cross_pop_ndd_each_neuron,avg_out_cross_pop_ndd_each_neuron, per_neuron_w_ndd_file); //  saves the local (per-neuron) in/out weights and ndds

    // =============================================================================
    // Plasticity phase (STDP + SP)
    // =============================================================================
    cout << "\n--- Plasticity Phase ---" << endl;
    sp_it = 0; flg = 1;
    Avg_W0 = 0; fi_mean0 = 0;
    k = 8.0;
    int i = 0;
    iSPupdate = 0;

    while (iSPupdate < nSPupdate) {
        pruned = 0; added = 0;
        for (int ii = 0; ii < N; ++ii) last_spike_before_window[ii] = spikeTimes[ii];
        for (int ii = 0; ii < N; ++ii) spike_times_window[ii].clear();
        run_simulation_window(
            rtime0 + i * Tw, base_time_step, true, k, pi, random, dt, nt,
            Vrest, Vsyn, Vth_rest, Tsyn, Tth, C0, knoise, fnoise,
            tp, td, ed, ep, Tslow, Tspike, Vspike, Vreset, Vth_spike,
            gleak, gsyn, gnoise, V, Vth, Istim, fi, flag,
            spikeTimes, spikeTimes0, counts2, trace_post,
            spike_history, trace_pre_d, W, A, noise_flat,
            is_post_spiker, is_arriving,
            delay_steps_matrix, unique_delays, max_delay_steps, num_delays,
            iLinks_by_delay, oLinks_by_delay, odpr_1, odpr_2, odpr_12,
            spike_times_window);
        base_time_step += nt;

        update_corr_trace(corr_trace, spike_times_window, last_spike_before_window,
                          delay_steps_matrix, Tslow, dt, N);
        compute_correlation(corr_trace, fi, spikeTimes, Tslow,
                           base_time_step * dt, N, Corr);

        calculate_mean_weight_fr_ndd(A, W, fi, n_con, Avg_W, fi_mean, node_degree_density,
                                      n_con_1, Avg_W_1, fi_mean_1, node_degree_density_1,
                                      n_con_2, Avg_W_2, fi_mean_2, node_degree_density_2,
                                      n_con_1to2, Avg_W_1to2, node_degree_density_1to2, n_con_2to1, Avg_W_2to1, node_degree_density_2to1,
                                      include_cross_conn_in_measures);

        A0 = A;
        SP(random, W, A, A0, Zd, fi, inlinks, Tw, Wmin, rate_hom, rate_hebb, c1, f1, f2, nu, pruned, added, n_con, iLinks, xi, network_dimensions, lower_bound_cross_pop, lower_bound_same_pop, Corr, corr_th0, nu_corr, corr_dep_add);

        // Rebuild flat link lists after SP
        update_link_lists(A, iLinks, oLinks, inlinks, outlinks);
        iSPupdate++;
        for (int ii = 0; ii < N; ++ii) {
            iLinks[ii].resize((int)inlinks[ii]);  iLinks[ii].shrink_to_fit();
            oLinks[ii].resize((int)outlinks[ii]); oLinks[ii].shrink_to_fit();
        }
        // Rebuild delay-grouped link lists after SP.
        // delay_steps_matrix is pre-computed for all pairs — no update needed.
        build_delay_grouped_links(A, delay_steps_matrix, unique_delays, delay_to_index, N,
                                   iLinks_by_delay, oLinks_by_delay);

        if (iSPupdate % int(round(10 * 60e3 / Tw)) == 0 || iSPupdate == nSPupdate - 1) { // saving data every 10th minute
            data << left << setw(15) << odpr_1     << left << setw(15) << fi_mean_1
                 << left << setw(15) << Avg_W_1    << left << setw(15) << node_degree_density_1
                 << left << setw(15) << odpr_2     << left << setw(15) << fi_mean_2
                 << left << setw(15) << Avg_W_2    << left << setw(15) << node_degree_density_2
                 << left << setw(15) << odpr_12    << left << setw(15) << Avg_W_1to2
                 << left << setw(15) << node_degree_density_1to2  << left << setw(15) << Avg_W_2to1
                 << left << setw(15) << node_degree_density_2to1  << left << setw(15) << pruned     << left << setw(15) << added
                 << left << setw(15) << iSPupdate << endl;
            cout << "Pop1: " << odpr_1 << " " << fi_mean_1 << " " << Avg_W_1 << " " << node_degree_density_1
                 << " | Pop2: " << odpr_2 << " " << fi_mean_2 << " " << Avg_W_2 << " " << node_degree_density_2
                 << " | R12: " << odpr_12 << " Avg_W_1to2: " << Avg_W_1to2 << " Avg_W_2to1: " << Avg_W_2to1
                 << " | " << pruned << " " << added << " " << iSPupdate << endl;
            save_per_neuron(A, W, avg_in_same_pop_w_each_neuron,   avg_out_same_pop_w_each_neuron, avg_in_cross_pop_w_each_neuron,  avg_out_cross_pop_w_each_neuron, avg_in_same_pop_ndd_each_neuron, avg_out_same_pop_ndd_each_neuron, avg_in_cross_pop_ndd_each_neuron,avg_out_cross_pop_ndd_each_neuron, per_neuron_w_ndd_file); //  saves the local (per-neuron) in/out weights and ndds
            save_correlation_distribution(Corr, N, corr_dist_file, n_bins);
        }
        i++;
    }

    // =============================================================================
    // Finalisation
    // =============================================================================
    cout << "\n--- Simulation Finished. Saving final state. ---" << endl;
    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < N; ++jj) W_file << A[ii][jj] * W[ii][jj] << " ";
        W_file << endl;
    }
    // saving final firing rate
    for (int ii = 0; ii < N; ++ii) FR_file << fi[ii] << "  ";
    FR_file << endl;

    save_final_state(steadystate, xi, yi, gleak, V, Vth, fi, gsyn, gnoise, flag,
                     trace_post, trace_pre_d, A, W, delay_steps_matrix, N);

    data.close();
    W_file.close();
    FR_file.close();
    steadystate.close();
    per_neuron_w_ndd_file.close();
    total_run_time(simulation_start_time);
    return 0;
}

// =================================================================================
// Function Definitions
// =================================================================================

vector<vector<double>> Adjacency_Matrix(int N, double p) {
    vector<double> zeros(N, 0);
    vector<vector<double>> matrix(N, zeros);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            double rnd = 0.0001 * (rand() % 10001);
            if (rnd <= p) matrix[i][j] = 1;
        }
    }
    return matrix;
}

vector<vector<double>> Dist_Dep_Adjacency_Matrix(double p, const vector<vector<double>>& Zd,int number) {
    double rnd, seed_, ncon;
    vector<double> zeros(N, 0);
    vector<vector<double>> matrix(N, zeros);
    seed_ = 1000 * number;
    mt19937 rnd_gen(seed_);
    vector<int> neuron_indices(N_pop);
    uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    iota(neuron_indices.begin(), neuron_indices.end(), 0);
    ncon = 0;
    while (p - ncon / (double)N_pop / (N_pop - 1) > 0.001) {
        shuffle(neuron_indices.begin(), neuron_indices.end(), rnd_gen);
        for (int i : neuron_indices) {
            for (int j = 0; j < N_pop; j++) {
                rnd = uniform_dist(rnd_gen);
                if (i!=j && matrix[i][j]==0 && rnd < p*Zd[i][j]) {
                    matrix[i][j] = 1; ncon += 1;
                }
            }
            if (ncon > p * N_pop * (N_pop - 1)) break;
        }
    }
    iota(neuron_indices.begin(), neuron_indices.end(), N_pop);
    ncon = 0;
    while (p - ncon / (double)N_pop / (N_pop - 1) > 0.001) {
        shuffle(neuron_indices.begin(), neuron_indices.end(), rnd_gen);
        for (int i : neuron_indices) {
            for (int j = N_pop; j < N; j++) {
                rnd = uniform_dist(rnd_gen);
                if (i!=j && matrix[i][j]==0 && rnd < p*Zd[i][j]) {
                    matrix[i][j] = 1; ncon += 1;
                }
            }
            if (ncon > p * N_pop * (N_pop - 1)) break;
        }
    }
    return matrix;
}

void add_cross_population_connections(vector<vector<double>>& A, double p_cross, int number, const double xi[], double network_dimensions) {
    mt19937 rnd_gen(10000 * number);
    double x_threshold = 0.1 * network_dimensions;
    int n_connect = (int)(p_cross * N_pop);

    // Pop1->Pop2: for each neuron i in Pop2, connect to n_connect neurons from Pop1
    for (int i = N_pop; i < N; i++) { // for post neuron in pop 2
        vector<int> candidates;
        for (int j = 0; j < N_pop; j++) { // pre neuron in pop 1
            if (fabs(xi[i] - xi[j]) <= x_threshold) candidates.push_back(j);
        }
        shuffle(candidates.begin(), candidates.end(), rnd_gen);
        int n = min(n_connect, (int)candidates.size());
        for (int k = 0; k < n; k++) A[i][candidates[k]] = 1;
    }

    // Pop2->Pop1: for each neuron i in Pop1, connect to n_connect neurons from Pop2
    for (int i = 0; i < N_pop; i++) {
        vector<int> candidates;
        for (int j = N_pop; j < N; j++) {
            if (fabs(xi[i] - xi[j]) <= x_threshold) candidates.push_back(j);
        }
        shuffle(candidates.begin(), candidates.end(), rnd_gen);
        int n = min(n_connect, (int)candidates.size());
        for (int k = 0; k < n; k++) A[i][candidates[k]] = 1;
    }
}

void compute_correlation_distribution(const vector<vector<double>>& Corr, int N,
                                      int n_bins, vector<double>& bin_edges,
                                      vector<int>& same_pop_counts,
                                      vector<int>& cross_pop_counts) {
    double bin_width = 2.0 / n_bins;
    bin_edges.resize(n_bins + 1);
    same_pop_counts.assign(n_bins, 0);
    cross_pop_counts.assign(n_bins, 0);
    for (int k = 0; k <= n_bins; k++)
        bin_edges[k] = -1.0 + k * bin_width;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            double val = max(-1.0, min(1.0, Corr[i][j]));
            int bin = (int)floor((val + 1.0) / bin_width);
            if (bin == n_bins) bin = n_bins - 1; // handles val == 1.0 exactly
            bool same_pop = (i < N/2) == (j < N/2);
            if (same_pop) same_pop_counts[bin]++;
            else          cross_pop_counts[bin]++;
        }
    }
}

void save_correlation_distribution(const vector<vector<double>>& Corr, int N, ofstream& corr_dist_file, int n_bins) {
    static bool edges_written = false; // initializer runs only on the very first call; the
                                        // variable then persists as true across all later calls
    vector<double> bin_edges;
    vector<int>    same_pop_counts, cross_pop_counts;
    compute_correlation_distribution(Corr, N, n_bins, bin_edges, same_pop_counts, cross_pop_counts);
    if (!edges_written) {
        for (int k = 0; k <= n_bins; k++)
            corr_dist_file << left << setw(15) << bin_edges[k];
        corr_dist_file << "\n";
        edges_written = true;
    }
    for (int k = 0; k < n_bins; k++)
        corr_dist_file << left << setw(15) << same_pop_counts[k];
    corr_dist_file << left << setw(15) << 0 << "\n";
    for (int k = 0; k < n_bins; k++)
        corr_dist_file << left << setw(15) << cross_pop_counts[k];
    corr_dist_file << left << setw(15) << 0 << "\n";
}

void setup_files(int trial_number, ofstream& data, ofstream& W_file, ofstream& A_file, ofstream& FR_file, ofstream& steadystate, double same_pop_delay, 
                 ofstream& per_neuron_w_ndd_file, ofstream& corr_dist_file, string corr_dep_add, double corr_th0, double delta_corr, double W_mean, double p, double p_cross) {
    char f1[500]={}, f2[500]={}, f3[500]={}, f4[500]={}, f5[500]={}, f6[500]={}, f7[500]={};
    string Address = "../Data/";
    snprintf(f1, sizeof(f1), "STDP+SP(corr-dep-add-%s,Wmean=%.1f,beta=%.3f,p_cross_pop0=%.3f)%d.dat", corr_dep_add.c_str(), W_mean, p, p_cross, trial_number);
    snprintf(f2, sizeof(f2), "W_STDP+SP(corr-dep-add-%s,Wmean=%.1f,beta=%.3f,p_cross_pop0=%.3f)%d.dat", corr_dep_add.c_str(), W_mean, p, p_cross, trial_number);
    snprintf(f3, sizeof(f3), "A_STDP+SP(corr-dep-add-%s,Wmean=%.1f,beta=%.3f,p_cross_pop0=%.3f)%d.dat", corr_dep_add.c_str(), W_mean,  p, p_cross, trial_number);
    snprintf(f4, sizeof(f4), "FR_STDP+SP(corr-dep-add-%s,Wmean=%.1f,beta=%.3f,p_cross_pop0=%.3f)%d.dat", corr_dep_add.c_str(), W_mean, p, p_cross, trial_number);
    snprintf(f5, sizeof(f5), "Steady_state_STDP+SP(corr-dep-add-%s,Wmean=%.1f,beta=%.3f,p_cross_pop0=%.3f)%d.dat", corr_dep_add.c_str(), W_mean, p, p_cross, trial_number);
    snprintf(f6, sizeof(f6), "Per_neuron(corr-dep-add-%s,Wmean=%.1f,beta=%.3f,p_cross_pop0=%.3f)%d.dat", corr_dep_add.c_str(), W_mean, p, p_cross, trial_number);
    snprintf(f7, sizeof(f7), "Correlation_dist(corr-dep-add-%s,Wmean=%.1f,beta=%.3f,p_cross_pop0=%.3f)%d.dat", corr_dep_add.c_str(), W_mean, p, p_cross, trial_number);
    data.open(Address + f1);
    W_file.open(Address + f2);
    FR_file.open(Address + f4);
    steadystate.open(Address + f5);
    per_neuron_w_ndd_file.open(Address + f6);
    corr_dist_file.open(Address + f7);
}


// Compute 2D neuron positions with Gaussian jitter, then:
//   Zd[i][j] = exp(-dist/dst)  for same-population pairs (used for connectivity)
//   Dd[i][j] = raw Euclidean distance in mm       (used for delay computation)
void initialize_network_geometry(Ran& random, double network_dimensions,
                                  double xi[], double yi[],
                                  vector<vector<double>>& Zd,
                                  vector<vector<double>>& Dd) {
    double hr  = network_dimensions / (m - 1);
    double dst = 0.25 * network_dimensions;
    double pi  = 4 * atan(1);

    for (int i = 0; i < N; i++) {
        double un = random.doub();
        double vn = random.doub();
        xi[i] = fmod((i + 0.5), m) * hr + hr / 10.0 * sqrt(-2 * log(un)) * cos(2 * pi * vn);
        yi[i] = (0.5 + int(i / m)) * hr + hr / 10.0 * sqrt(-2 * log(un)) * sin(2 * pi * vn);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) { Dd[i][j] = 0.0; continue; }
            double dij = sqrt(pow(xi[i]-xi[j], 2) + pow(yi[i]-yi[j], 2));
            Dd[i][j] = dij;
            bool same_pop = (i < N/2) == (j < N/2);
            if (same_pop) Zd[i][j] = exp(-dij / dst);
        }
    }
}

void initialize_neuron_states(Ran& random, Normaldev_BM& normal_random,
                               double gleak[], double V[], double Vth[], double fi[],
                               double gsyn[], double gnoise[], double flag[],
                               double spikeTimes[], double spikeTimes0[], double train_post[], double trace_post[],
                               double Istim[], int counts2[],
                               double Vrest, double Vreset, double Vth_rest,
                               double gleak0, double gleak_std) {
    for (int i = 0; i < N; i++) {
        gleak[i]     = gleak0 + gleak_std * normal_random.dev();
        V[i]         = Vreset + (Vrest - Vreset) * random.doub();
        Vth[i]       = Vth_rest;
        fi[i]        = 0;  gsyn[i]     = 0;  gnoise[i]   = 0;
        flag[i]      = 1;  spikeTimes[i] = 0; spikeTimes0[i] = 0;
        train_post[i] = 0; trace_post[i] = 0;
        Istim[i]     = 0;  counts2[i]   = 0;
    }
}

void update_link_lists(const vector<vector<double>>& A, vector<vector<int>>& iLinks,
                       vector<vector<int>>& oLinks, double inlinks[], double outlinks[]) {
    for (int i = 0; i < N; ++i) {
        iLinks[i].resize(N); oLinks[i].resize(N);
        int pos_in = 0, pos_out = 0, nl_in = 0, nl_out = 0;
        for (int j = 0; j < N; ++j) {
            if (A[i][j] == 1) { iLinks[i][pos_in++] = j; nl_in++;  }
            if (A[j][i] == 1) { oLinks[i][pos_out++] = j; nl_out++; }
        }
        inlinks[i] = nl_in; outlinks[i] = nl_out;
    }
}

// Compute per-connection delays for ALL neuron pairs (not just connected ones).
//   Same-population: delay_ms = min_syn_delay + delay_slope * Dd[i][j]
//   Cross-population: delay_ms = cross_pop_delay
//   Discretised to dt resolution, minimum 1 step.
// Pre-computing for all pairs means SP-added connections automatically have
// the correct delay without any update inside SP.
// unique_delays  : sorted vector of all distinct delay values (in timesteps).
// max_delay_steps: max(unique_delays) + 1  (prevents read/write slot collision).
void compute_delay_matrix(const vector<vector<double>>& Dd,
                           vector<vector<int>>& delay_steps_matrix,
                           vector<int>& unique_delays, map<int,int>& delay_to_index,
                           int& max_delay_steps,
                           double min_syn_delay, double delay_slope, double cross_pop_delay,
                           double dt, int N) {
    set<int> delay_set;
    int cross_d = max(1, (int)round(cross_pop_delay / dt));
    delay_set.insert(cross_d);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) { delay_steps_matrix[i][j] = 0; continue; }
            bool same_pop = (i < N/2) == (j < N/2);
            int d;
            if (same_pop) {
                double delay_ms = min_syn_delay + delay_slope * Dd[i][j];
                d = max(1, (int)round(delay_ms / dt));
            } else {
                d = cross_d;
            }
            delay_steps_matrix[i][j] = d;
            delay_set.insert(d);
        }
    }
    unique_delays.assign(delay_set.begin(), delay_set.end());
    sort(unique_delays.begin(), unique_delays.end());
    delay_to_index.clear();
    for (int k = 0; k < (int)unique_delays.size(); k++)
        delay_to_index[unique_delays[k]] = k;
    max_delay_steps = unique_delays.back() + 1;
}

// Build delay-grouped link lists from the current adjacency matrix A.
//   iLinks_by_delay[post][k]: pre-synaptic partners of 'post' with delay index k.
//   oLinks_by_delay[pre][k] : post-synaptic partners of 'pre' with delay index k.
// Both are in ascending neuron-index order.
void build_delay_grouped_links(const vector<vector<double>>& A,
                                const vector<vector<int>>& delay_steps_matrix,
                                const vector<int>& unique_delays,
                                const map<int,int>& delay_to_index, int N,
                                vector<vector<vector<int>>>& iLinks_by_delay,
                                vector<vector<vector<int>>>& oLinks_by_delay) {
    int num_delays = (int)unique_delays.size();
    for (int i = 0; i < N; i++) {
        iLinks_by_delay[i].assign(num_delays, vector<int>());
        oLinks_by_delay[i].assign(num_delays, vector<int>());
    }
    // Iterate post then pre in ascending order → ascending-index lists in both maps
    for (int post = 0; post < N; post++) {
        for (int pre = 0; pre < N; pre++) {
            if (A[post][pre] == 1) {
                int d = delay_steps_matrix[post][pre];
                int k = delay_to_index.at(d);
                iLinks_by_delay[post][k].push_back(pre);  // ascending pre  order 
                oLinks_by_delay[pre][k].push_back(post);  // ascending post order 
            }
        }
    }
}

void calculate_mean_weight_fr_ndd(const vector<vector<double>>& A,
    const vector<vector<double>>& W, double fi[],
    double& n_con,   double& Avg_W,   double& fi_mean,   double& node_degree_density,
    double& n_con_1, double& Avg_W_1, double& fi_mean_1, double& node_degree_density_1,
    double& n_con_2, double& Avg_W_2, double& fi_mean_2, double& node_degree_density_2,
    double& n_con_1to2, double& Avg_W_1to2, double& node_degree_density_1to2,
    double& n_con_2to1, double& Avg_W_2to1, double& node_degree_density_2to1,
    bool include_cross) {
    double sum_w=0, sum_w_1=0, sum_w_2=0;
    double sum_w_1to2=0, sum_w_2to1=0;
    n_con=0; n_con_1=0; n_con_2=0; n_con_1to2=0; n_con_2to1=0;
    fi_mean=0; fi_mean_1=0; fi_mean_2=0;
    for (int i = 0; i < N; ++i) {
        fi_mean += fi[i];
        if (i < N/2) fi_mean_1 += fi[i]; else fi_mean_2 += fi[i];
        for (int j = 0; j < N; ++j) {
            if (A[i][j] == 1) {
                bool cross = (i < N/2) != (j < N/2);
                n_con++; sum_w += W[i][j];
                if (i < N/2) {
                    if (!cross || include_cross) { n_con_1++; sum_w_1 += W[i][j]; }
                } else {
                    if (!cross || include_cross) { n_con_2++; sum_w_2 += W[i][j]; }
                }
                // Cross-population directional weights (independent of include_cross flag)
                if ((i >= N/2) && (j < N/2)) { n_con_1to2++; sum_w_1to2 += W[i][j]; } // pre=Pop1, post=Pop2
                if ((i < N/2)  && (j >= N/2)) { n_con_2to1++; sum_w_2to1 += W[i][j]; } // pre=Pop2, post=Pop1
            }
        }
    }
    fi_mean /= N; fi_mean_1 /= N_pop; fi_mean_2 /= N_pop;
    node_degree_density   = n_con   / (double)N     / (N - 1);
    double max_deg = include_cross ? (N - 1) : (N_pop - 1);
    node_degree_density_1 = n_con_1 / (double)N_pop / max_deg;
    node_degree_density_2 = n_con_2 / (double)N_pop / max_deg;
    node_degree_density_1to2 = n_con_1to2 / (double)N_pop / N_pop;
    node_degree_density_2to1 = n_con_2to1 / (double)N_pop / N_pop;

    Avg_W      = (n_con      > 0) ? sum_w      / n_con      : 0.0;
    Avg_W_1    = (n_con_1    > 0) ? sum_w_1    / n_con_1    : 0.0;
    Avg_W_2    = (n_con_2    > 0) ? sum_w_2    / n_con_2    : 0.0;
    Avg_W_1to2 = (n_con_1to2 > 0) ? sum_w_1to2 / n_con_1to2 : 0.0;
    Avg_W_2to1 = (n_con_2to1 > 0) ? sum_w_2to1 / n_con_2to1 : 0.0;
}

// Per-neuron mean synaptic weight and node degree density, split by same/cross pop
// and incoming/outgoing direction.
// Convention: A[i][j] == 1 → j is pre, i is post (j→i).
// NDD normalisation: same-pop / (N_pop-1), cross-pop / N_pop.
void calculate_per_neuron_weight_ndd(const vector<vector<double>>& A,
    const vector<vector<double>>& W,
    double avg_in_same_pop_w_each_neuron[],  double avg_out_same_pop_w_each_neuron[],
    double avg_in_cross_pop_w_each_neuron[], double avg_out_cross_pop_w_each_neuron[],
    double avg_in_same_pop_ndd_each_neuron[],  double avg_out_same_pop_ndd_each_neuron[],
    double avg_in_cross_pop_ndd_each_neuron[], double avg_out_cross_pop_ndd_each_neuron[]) {
    for (int i = 0; i < N; ++i) {
        bool pop1_i = (i < N/2);
        double sum_in_same = 0, sum_in_cross = 0;
        double sum_out_same = 0, sum_out_cross = 0;
        int n_in_same = 0, n_in_cross = 0;
        int n_out_same = 0, n_out_cross = 0;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            bool same_pop = (pop1_i == (j < N/2));
            if (A[i][j] == 1) {  // j → i (incoming)
                if (same_pop) { sum_in_same  += W[i][j]; n_in_same++;  }
                else          { sum_in_cross += W[i][j]; n_in_cross++; }
            }
            if (A[j][i] == 1) {  // i → j (outgoing)
                if (same_pop) { sum_out_same  += W[j][i]; n_out_same++;  }
                else          { sum_out_cross += W[j][i]; n_out_cross++; }
            }
        }
        avg_in_same_pop_w_each_neuron[i]   = (n_in_same   > 0) ? sum_in_same   / n_in_same   : 0.0;
        avg_out_same_pop_w_each_neuron[i]  = (n_out_same  > 0) ? sum_out_same  / n_out_same  : 0.0;
        avg_in_cross_pop_w_each_neuron[i]  = (n_in_cross  > 0) ? sum_in_cross  / n_in_cross  : 0.0;
        avg_out_cross_pop_w_each_neuron[i] = (n_out_cross > 0) ? sum_out_cross / n_out_cross : 0.0;
        avg_in_same_pop_ndd_each_neuron[i]   = n_in_same   / (double)(N_pop - 1);
        avg_out_same_pop_ndd_each_neuron[i]  = n_out_same  / (double)(N_pop - 1);
        avg_in_cross_pop_ndd_each_neuron[i]  = n_in_cross  / (double)N_pop;
        avg_out_cross_pop_ndd_each_neuron[i] = n_out_cross / (double)N_pop;
    }
}

// Calls "calculate_per_neuron_weight_ndd" then writes the 8 result arrays as 8 rows
// (one row per array, N values each) to per_neuron_w_ndd_file.
// Row order: same-pop in/out weight, cross-pop in/out weight,
//            same-pop in/out NDD,    cross-pop in/out NDD.
void save_per_neuron(const vector<vector<double>>& A, const vector<vector<double>>& W,
    double avg_in_same_pop_w_each_neuron[],  double avg_out_same_pop_w_each_neuron[],
    double avg_in_cross_pop_w_each_neuron[], double avg_out_cross_pop_w_each_neuron[],
    double avg_in_same_pop_ndd_each_neuron[],  double avg_out_same_pop_ndd_each_neuron[],
    double avg_in_cross_pop_ndd_each_neuron[], double avg_out_cross_pop_ndd_each_neuron[],
    ofstream& per_neuron_w_ndd_file) {
    calculate_per_neuron_weight_ndd(A, W,
        avg_in_same_pop_w_each_neuron,   avg_out_same_pop_w_each_neuron,
        avg_in_cross_pop_w_each_neuron,  avg_out_cross_pop_w_each_neuron,
        avg_in_same_pop_ndd_each_neuron, avg_out_same_pop_ndd_each_neuron,
        avg_in_cross_pop_ndd_each_neuron,avg_out_cross_pop_ndd_each_neuron);
    double* rows[8] = {
        avg_in_same_pop_w_each_neuron,   avg_out_same_pop_w_each_neuron,
        avg_in_cross_pop_w_each_neuron,  avg_out_cross_pop_w_each_neuron,
        avg_in_same_pop_ndd_each_neuron, avg_out_same_pop_ndd_each_neuron,
        avg_in_cross_pop_ndd_each_neuron,avg_out_cross_pop_ndd_each_neuron
    };
    for (int r = 0; r < 8; ++r) {
        for (int ii = 0; ii < N; ++ii)
            per_neuron_w_ndd_file << left << setw(15) << rows[r][ii];
        per_neuron_w_ndd_file << endl;
    }
}

// =================================================================================
// Core simulation loop 
// =================================================================================
void run_simulation_window(
    double base_time, long long base_time_step, bool plasticity_on, double k, double pi, Ran& random,
    double dt, int nt, double Vrest, double Vsyn, double Vth_rest,
    double Tsyn, double Tth, double C0, double knoise, double fnoise,
    double tp, double td, double ed, double ep,
    double Tslow, double Tspike, double Vspike, double Vreset, double Vth_spike,
    double gleak[], double gsyn[], double gnoise[], double V[], double Vth[],
    double Istim[], double fi[], double flag[],
    double spikeTimes[], double spikeTimes0[], int counts2[],
    double trace_post[],
    vector<vector<double>>& spike_history,
    vector<vector<double>>& trace_pre_d,
    vector<vector<double>>& W, const vector<vector<double>>& A,
    vector<double>& noise_flat,
    bool is_post_spiker[],
    vector<vector<uint8_t>>& is_arriving,
    const vector<vector<int>>& delay_steps_matrix,
    const vector<int>& unique_delays,
    int max_delay_steps, int num_delays,
    const vector<vector<vector<int>>>& iLinks_by_delay,
    const vector<vector<vector<int>>>& oLinks_by_delay,
    double& odpr_1, double& odpr_2, double& odpr_12,
    vector<vector<double>>& spike_times_window) {

    double dt_inv      = 1.0 / dt;
    double current_time, sumcos1, sumsin1, sumcos2, sumsin2, isi, phi;
    int    n_noise_spikes, i_noise_spike;
    odpr_1 = 0; odpr_2 = 0; odpr_12 = 0;

    // arrivals[k]: neurons whose delayed spike arrives via delay group k this step
    vector<vector<int>> arrivals(num_delays);
    vector<int> post_spikers;

    int K_odp     = round(100.0 / dt);  // order-parameter sampling interval (every 100 ms)
    int odp_count = 0;

    // ---- Generate Poisson noise spike trains ----
    fill(noise_flat.begin(), noise_flat.end(), 0.0);
    n_noise_spikes = int(10 * (nt * dt) / 1000 * fnoise);
    for (int i = 0; i < N; ++i) {
        i_noise_spike = 0;
        for (int j = 0; j < n_noise_spikes; ++j) {
            i_noise_spike += int(-log(random.doub()) / fnoise * 1000 / dt);
            if (i_noise_spike >= nt) break;
            noise_flat[i_noise_spike * N + i] = dt_inv;
        }
    }

    for (int j = 0; j < nt; ++j) {
        current_time = base_time + j * dt;

        // Slot into which THIS step's spike will be written (after detection below).
        // max_delay_steps = max(all delays) + 1, so write_slot != any read_slot. 
        long long gs = base_time_step + j;
        int write_slot = (int)(gs % max_delay_steps);

        post_spikers.clear();

        // =========================================================================
        // PER-DELAY-GROUP: read delayed spikes, update traces, build arrivals
        // =========================================================================
        for (int dk = 0; dk < num_delays; dk++) {
            arrivals[dk].clear();
            int d         = unique_delays[dk];
            int read_slot = (int)((gs - d + max_delay_steps) % max_delay_steps);
            double* tpre  = trace_pre_d[dk].data();
            for (int i = 0; i < N; ++i) {
                double delayed_spike = spike_history[i][read_slot];
                tpre[i] += dt * (-tpre[i] / tp + delayed_spike);
                if (delayed_spike != 0.0) arrivals[dk].push_back(i);
            }
        }

        // =========================================================================
        // Step 2 — InF: update V and Vth using gsyn from the previous step
        // =========================================================================
        double* noise = noise_flat.data() + j * N;
        InF(dt, gleak, gsyn, gnoise, noise, V, Vth, Istim,
            Vrest, Vsyn, Vth_rest, Tsyn, Tth, C0, knoise, N);

        // =========================================================================
        // Step 3 — gsyn: decay then event-driven scatter from arrivals
        // =========================================================================
        double gsyn_decay = 1.0 - dt / Tsyn;
        for (int ii = 0; ii < N; ++ii) gsyn[ii] *= gsyn_decay;
        double k_over_N = k / N;
        for (int dk = 0; dk < num_delays; dk++) {
            for (int pre : arrivals[dk]) {
                for (int post : oLinks_by_delay[pre][dk]) {
                    gsyn[post] += k_over_N * W[post][pre];
                }
            }
        }

        // =========================================================================
        // Step 4 — Spike detection: clear write slot, check threshold, record spike
        // =========================================================================
        for (int i = 0; i < N; ++i) {
            spike_history[i][write_slot] = 0.0;   // clear slot (was max_delay_steps steps old)
            if (V[i] >= Vth[i] && flag[i] == 1) {
                spikeTimes0[i] = spikeTimes[i];
                spikeTimes[i]  = current_time;
                counts2[i]++;
                spike_history[i][write_slot] = dt_inv;
                Vth[i]  = Vth_spike;
                V[i]    = Vspike;
                flag[i] = 0;
                post_spikers.push_back(i);
                spike_times_window[i].push_back(current_time);
            }
            if (flag[i] == 0) {
                if (current_time < spikeTimes[i] + Tspike) V[i] = Vspike;
                else { V[i] = Vreset; flag[i] = 1; }
            }
        }

        // =========================================================================
        // Step 5 — trace_post and fi: driven by CURRENT (undelayed) spikes
        // =========================================================================
        for (int i = 0; i < N; ++i) {
            double spike_val = spike_history[i][write_slot];
            trace_post[i] += dt * (-trace_post[i] / td + spike_val);
            fi[i]         += dt / Tslow * (-fi[i] + spike_val * 1e3);
        }

        // =========================================================================
        // Step 6 — STDP: delay-aware version using per-delay-group traces
        // =========================================================================
        bool any_arrivals = false;
        for (int dk = 0; dk < num_delays && !any_arrivals; dk++)
            if (!arrivals[dk].empty()) any_arrivals = true;

        if (plasticity_on && (any_arrivals || !post_spikers.empty())) {
            // Set membership flags (cleared immediately after STDP)
            for (int dk = 0; dk < num_delays; dk++)
                for (int pre : arrivals[dk]) is_arriving[dk][pre] = 1;
            for (int idx : post_spikers) is_post_spiker[idx] = true;

            delayAwareFastSTDP(N, dt, ed, ep, trace_post, trace_pre_d,
                               arrivals, post_spikers, is_arriving, is_post_spiker,
                               iLinks_by_delay, oLinks_by_delay, W, dt_inv, num_delays);

            // Clear flags
            for (int dk = 0; dk < num_delays; dk++)
                for (int pre : arrivals[dk]) is_arriving[dk][pre] = 0;
            for (int idx : post_spikers) is_post_spiker[idx] = false;
        }

        // =========================================================================
        // Step 7 — Order parameter (subsampled every K_odp steps)
        // =========================================================================
        if (j % K_odp == 0) {
            sumcos1 = 0; sumsin1 = 0; sumcos2 = 0; sumsin2 = 0;
            for (int i = 0; i < N/2; ++i) {
                isi = spikeTimes[i] - spikeTimes0[i];
                phi = 2 * pi * ((current_time - spikeTimes0[i]) / isi + counts2[i] - 1);
                sumcos1 += cos(phi); sumsin1 += sin(phi);
            }
            for (int i = N/2; i < N; ++i) {
                isi = spikeTimes[i] - spikeTimes0[i];
                phi = 2 * pi * ((current_time - spikeTimes0[i]) / isi + counts2[i] - 1);
                sumcos2 += cos(phi); sumsin2 += sin(phi);
            }
            complex<double> Z1(sumcos1 / N_pop, sumsin1 / N_pop);
            complex<double> Z2(sumcos2 / N_pop, sumsin2 / N_pop);
            odpr_1 += abs(Z1);
            odpr_2 += abs(Z2);
            odpr_12 += (abs(Z1) < 1e-1 || abs(Z2) < 1e-1) ? 0.0 : abs(Z1) * abs(Z2) * cos(arg(Z1) - arg(Z2));
            odp_count++;
        }
    }
    odpr_1  /= odp_count;
    odpr_2  /= odp_count;
    odpr_12 /= odp_count;
}

// =================================================================================
// delayAwareFastSTDP
//
// Implements STDP with per-synapse (per-delay-group) presynaptic traces.
//
// LTP — when post spikes: for each incoming synapse (pre→post) with delay group dk,
//   use trace_pre_d[dk][pre] as the presynaptic trace.
//   Skip if is_arriving[dk][pre] == 1:  pre's spike arrived at THIS synapse at the
//   same step post fired — simultaneous activation, no weight change.
//
// LTD — when a delayed pre spike arrives at synapse (pre→post):
//   depress using trace_post[post].
//   Skip if is_post_spiker[post]: simultaneous — skip.
//
// Clamp weights to [0, 1] after LTP and LTD.
// =================================================================================
void delayAwareFastSTDP(int N, double dt, double ed, double ep,
                        double trace_post[],
                        const vector<vector<double>>& trace_pre_d,
                        const vector<vector<int>>& arrivals,
                        const vector<int>& post_spikers,
                        vector<vector<uint8_t>>& is_arriving,
                        const bool is_post_spiker[],
                        const vector<vector<vector<int>>>& iLinks_by_delay,
                        const vector<vector<vector<int>>>& oLinks_by_delay,
                        vector<vector<double>>& W, double dt_inv, int num_delays) {
    // ---- LTP: post fires → potentiate each incoming synapse ----
    for (int post : post_spikers) {
        double train_post_val = dt_inv;  // spike_history[post][write_slot] = dt_inv
        for (int dk = 0; dk < num_delays; dk++) {
            const double* tpre = trace_pre_d[dk].data();
            for (int pre : iLinks_by_delay[post][dk]) {
                if (is_arriving[dk][pre]) continue;  // simultaneous at this synapse — skip
                W[post][pre] += dt * ep * tpre[pre] * train_post_val;
            }
        }
    }

    // ---- LTD: pre's delayed spike arrives → depress each outgoing synapse ----
    for (int dk = 0; dk < num_delays; dk++) {
        for (int pre : arrivals[dk]) {
            double train_pre_val = dt_inv;  // spike_history[pre][read_slot] = dt_inv
            for (int post : oLinks_by_delay[pre][dk]) {
                if (is_post_spiker[post]) continue;  // simultaneous — skip
                W[post][pre] -= dt * ed * trace_post[post] * train_pre_val;
            }
        }
    }

    // ---- Clamp weights to [0, 1] ----
    for (int post : post_spikers) {
        for (int dk = 0; dk < num_delays; dk++) {
            for (int pre : iLinks_by_delay[post][dk]) {
                if (W[post][pre] > 1.0) W[post][pre] = 1.0;
                if (W[post][pre] < 0.0) W[post][pre] = 0.0;
            }
        }
    }
    for (int dk = 0; dk < num_delays; dk++) {
        for (int pre : arrivals[dk]) {
            for (int post : oLinks_by_delay[pre][dk]) {
                if (W[post][pre] > 1.0) W[post][pre] = 1.0;
                if (W[post][pre] < 0.0) W[post][pre] = 0.0;
            }
        }
    }
}

// Save final network state. 
void save_final_state(ofstream& steadystate,
                      double xi[], double yi[], double gleak[], double V[], double Vth[],
                      double fi[], double gsyn[], double gnoise[], double flag[],
                      double trace_post[],
                      const vector<vector<double>>& trace_pre_d,
                      const vector<vector<double>>& A, const vector<vector<double>>& W,
                      const vector<vector<int>>& delay_steps_matrix, int N) {
    auto save_arr = [&](double arr[]) {
        for (int i = 0; i < N; ++i) steadystate << arr[i] << endl;
    };
    auto save_matrix = [&](const vector<vector<double>>& mat, bool use_A = false) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                steadystate << (use_A ? mat[i][j] : A[i][j] * mat[i][j]) << endl;
    };

    save_arr(xi); save_arr(yi); save_arr(gleak); save_arr(V); save_arr(Vth);
    save_arr(fi); save_arr(gsyn); save_arr(gnoise); save_arr(flag);

    // train_pre placeholder (zeros)
    for (int i = 0; i < N; ++i) steadystate << 0.0 << endl;
    // train_post placeholder (zeros)
    for (int i = 0; i < N; ++i) steadystate << 0.0 << endl;
    // trace_pre: representative value from the first (minimum) delay group
    for (int i = 0; i < N; ++i) steadystate << trace_pre_d[0][i] << endl;
    save_arr(trace_post);

    save_matrix(A, true);   // adjacency matrix
    save_matrix(W, false);  // A * W weight matrix

    // delay matrix
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            steadystate << delay_steps_matrix[i][j] << endl;
}

void InF(double dt, double gleak[], double gsyn[], double gnoise[], double noise[],
         double V[], double Vth[], double Istim[],
         double Vrest, double Vsyn, double Vth_rest, double Tsyn, double Tth,
         double C0, double knoise, int N) {
    for (int ii = 0; ii < N; ii++) {
        V[ii]   = V[ii]   + dt/C0 * (gleak[ii]*(Vrest-V[ii]) + (gsyn[ii]+gnoise[ii])*(Vsyn-V[ii]) + Istim[ii]);
        Vth[ii] = Vth[ii] - dt/Tth * (Vth[ii] - Vth_rest);
    }
    for (int ii = 0; ii < N; ii++)
        gnoise[ii] = gnoise[ii] + dt/Tsyn * (-gnoise[ii] + knoise*Tsyn*noise[ii]);
}

// Legacy fastSTDP (kept for reference; not called in this code)
void fastSTDP(int N, double dt, double ed, double ep,
              double train_pre[], double train_post[], double trace_pre[], double trace_post[],
              double inlinks[], double outlinks[],
              vector<vector<double>>& W, const vector<vector<int>>& iLinks,
              const vector<vector<int>>& oLinks,
              const vector<int>& pre_spikers, const vector<int>& post_spikers,
              const bool is_pre_spiker[], const bool is_post_spiker[]) {
    int nl, pre, post;
    for (int p : post_spikers) {
        nl = inlinks[p];
        for (int jj = 0; jj < nl; jj++) {
            pre = iLinks[p][jj];
            if (is_pre_spiker[pre]) continue;
            W[p][pre] += dt * ep * trace_pre[pre] * train_post[p];
        }
    }
    for (int p : pre_spikers) {
        nl = outlinks[p];
        for (int jj = 0; jj < nl; jj++) {
            post = oLinks[p][jj];
            if (is_post_spiker[post]) continue;
            W[post][p] -= dt * ed * trace_post[post] * train_pre[p];
        }
    }
    auto clamp = [&](const vector<int>& spikers, const auto& links, bool is_post) {
        for (int s : spikers) {
            nl = is_post ? inlinks[s] : outlinks[s];
            for (int jj = 0; jj < nl; jj++) {
                int partner = links[s][jj];
                int p1 = is_post ? s : partner;
                int p2 = is_post ? partner : s;
                if (W[p1][p2] > 1) W[p1][p2] = 1;
                if (W[p1][p2] < 0) W[p1][p2] = 0;
            }
        }
    };
    clamp(post_spikers, iLinks, true);
    clamp(pre_spikers,  oLinks, false);
}

void update_corr_trace(vector<vector<double>>& corr_trace,
                       const vector<vector<double>>& spike_times_window,
                       const double last_spike_before_window[],
                       const vector<vector<int>>& delay_steps_matrix,
                       double Tslow, double dt, int N) {
    double coincidence_window = 30.0; // \tau_c in ms- only the spike pairs with time-difference less than this contribute to the correlation trace
    for (int i = 0; i < N; ++i) {
        double t_prev = last_spike_before_window[i];
        for (double t_post : spike_times_window[i]) {
            double decay = exp(-(t_post - t_prev) / Tslow);
            for (int j = 0; j < N; ++j) corr_trace[i][j] *= decay;
            for (int j = 0; j < N; ++j) { // find the spike of neuron j whose arrival time at i is closest to t_post
                if (i == j) continue;
                double t_target = t_post - delay_steps_matrix[i][j] * dt;
                double t_nearest  = last_spike_before_window[j];
                double dist_min   = fabs(t_target - t_nearest);
                const auto& jspk  = spike_times_window[j];
                if (!jspk.empty()) { // Binary search for the nearest spike
                    auto it = lower_bound(jspk.begin(), jspk.end(), t_target); // finds the first j-spike ≥ t_target.
                    if (it != jspk.end()) {
                        double d = *it - t_target;
                        if (d < dist_min) { t_nearest = *it; dist_min = d; }
                    }
                    if (it != jspk.begin()) {
                        auto prev = std::prev(it); // *prev(it) is the spike just before t_target
                        double d = t_target - *prev;
                        if (d < dist_min) { t_nearest = *prev; dist_min = d; }
                    }
                }
                if (t_nearest == t_target || abs(t_nearest - t_target) > coincidence_window) continue;
                corr_trace[i][j] += (t_nearest < t_target) ? 1.0 : -1.0;
            }
            t_prev = t_post;
        }
    }
}

void compute_correlation(const vector<vector<double>>& corr_trace,
                        const double fi[], const double spikeTimes[],
                        double Tslow, double t_now, int N,
                        vector<vector<double>>& Corr) {
    for (int i = 0; i < N; ++i) {
        // this decay is needed because the corr_trace was last updated with the last spike of neuron i. 
        // Some time elapsed between last spike and now, during which corr_trace should have continued decaying with time constant Tslow.
        double final_decay = exp(-(t_now - spikeTimes[i]) / Tslow); 
        double scale = final_decay * 1e3 / (fi[i] * Tslow);
        for (int j = 0; j < N; ++j)
            Corr[i][j] = (i != j) ? corr_trace[i][j] * scale : 0.0;
    }
}

// delay_steps_matrix is pre-computed for all pairs in compute_delay_matrix,
// so no delay update is needed here.  After SP returns, main() calls
// update_link_lists() then build_delay_grouped_links() to refresh link lists.
void SP(Ran& random, vector<vector<double>>& W, vector<vector<double>>& A,
        vector<vector<double>>& A0, const vector<vector<double>>& Zd,
        double fi[], double inlinks[],
        double Tw, double Wmin, double rate_hom, double rate_hebb, double c,
        double f1, double f2, double nu,
        int& pruned, int& added, double& n_con, vector<vector<int>>& iLinks,
        const double xi[], double network_dimensions,
        double lower_bound_cross_pop, double lower_bound_same_pop,
        const vector<vector<double>>& Corr, double corr_th0, double nu_corr, string corr_dep_add) {
    pruned = 0; added = 0;
    int lower_bound_count_cross = (int)round(lower_bound_cross_pop * N_pop);
    int lower_bound_count_same = (int)round(lower_bound_same_pop * (N_pop-1));
    double x_threshold    = 0.1 * network_dimensions;
    double P_heb = 1 - exp(-rate_hebb * Tw * 1e-3);
    double Ph = 1 - exp(-rate_hom * Tw * 1e-3);
    vector<int> cross_pop_in_degree_post_pruning(N, 0);

    // PRUNING
    for (int i = 0; i < N; i++) {
        bool row_in_pop1 = (i < N/2);
        int nl = (int)inlinks[i];
        // Count same-pop in-links separately so that cross-pop connections
        // do not inflate the pruning guard for same-pop afferents.
        int nl_same = 0;
        for (int kk = 0; kk < nl; kk++)
            if (row_in_pop1 == (iLinks[i][kk] < N/2)) nl_same++;
        for (int kk = 0; kk < nl; ) {
            int j = iLinks[i][kk];
            bool same_pop = (row_in_pop1 == (j < N/2));
            if (same_pop && nl_same > lower_bound_count_same &&
                random.doub() < (P_heb * exp(-W[i][j] / Wmin) + Ph / (1 + exp(-(fi[i] - f2) / nu)))) {
                A[i][j] = 0; W[i][j] = 0;
                iLinks[i][kk] = iLinks[i][nl - 1];
                iLinks[i].resize(--nl);
                nl_same--;
                pruned++; n_con--;
            }
            // cross-pop pruning
            else if (!same_pop && (nl - nl_same) > lower_bound_count_cross &&
                       random.doub() < (P_heb * exp(-W[i][j] / Wmin) + Ph / (1 + exp(-(fi[i] - f2) / nu)))) {
                A[i][j] = 0; W[i][j] = 0;
                iLinks[i][kk] = iLinks[i][nl - 1];
                iLinks[i].resize(--nl);
                pruned++; n_con--;
            }
            else { kk++; }
        }
        inlinks[i] = nl;
        cross_pop_in_degree_post_pruning[i] = nl - nl_same;
    }

    // ADDITION
    vector<int> j_order(N);
    for (int kk = 0; kk < N; kk++) j_order[kk] = kk;

    for (int i = 0; i < N; i++) {
        bool row_in_pop1 = (i < N/2);
        int  n_cross_in  = cross_pop_in_degree_post_pruning[i];

        for (int kk = N - 1; kk > 0; kk--) {
            int swap_idx = (int)(random.doub() * (kk + 1));
            swap(j_order[kk], j_order[swap_idx]);
        }

        for (int j : j_order) {
            if (i == j || A0[i][j] != 0) continue;
            bool same_pop = (row_in_pop1 == (j < N/2));
            double corr_based_add = (corr_dep_add == "ON") ? P_heb / (1.0 + exp(-(Corr[i][j] - corr_th0) / nu_corr)) : 0.0; // correlation-dependent addition function, gated by corr_dep_add
            if (same_pop) { // same-pop addition
                if (random.doub() < (Ph / (1 + exp((fi[i] - f1) / nu)) + corr_based_add) * Zd[i][j]) {
                    A[i][j] = 1; W[i][j] = 0.5 * random.doub();
                    added++; n_con++;
                }
            } else { // cross-pop addition
                if (fabs(xi[i] - xi[j]) <= x_threshold &&
                    random.doub() <  Ph / (1 + exp((fi[i] - f1) / nu)) + corr_based_add) {
                    A[i][j] = 1; W[i][j] = 0.5 * random.doub();
                    n_cross_in++;
                    added++; n_con++;
                }
            }
        }
    }
}

void total_run_time(const chrono::high_resolution_clock::time_point& start) {
    auto end      = chrono::high_resolution_clock::now();
    auto total_ms = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto h  = total_ms / 3600000;
    auto mn = (total_ms % 3600000) / 60000;
    auto s  = (total_ms % 60000)   / 1000;
    auto ms = total_ms % 1000;
    cout << "Simulation time: " << h << "h " << mn << "m " << s << "s " << ms << "ms\n";
}
