# Version 2.1
# MAXDIFF ANALYSIS TOOL 

import customtkinter as ctk
from tkinter import filedialog, messagebox, colorchooser, ttk
import threading
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import seaborn as sns
import textwrap
import matplotlib.colors as mcolors
import queue
import re

# ============================================================================
# OPTIONAL HB DEPENDENCIES - NumPyro/JAX for fast MCMC
# ============================================================================
HAS_NUMPYRO = False
NUMPYRO_ERROR = None

try:
    import jax
    jax.config.update('jax_platform_name', 'cpu')  # Force CPU to avoid GPU issues
    import jax.numpy as jnp
    from jax import random
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    # Try to enable parallel chains (may fail on some systems)
    try:
        numpyro.set_host_device_count(4)
    except:
        pass
    HAS_NUMPYRO = True
except ImportError as e:
    NUMPYRO_ERROR = str(e)
except Exception as e:
    NUMPYRO_ERROR = str(e)

# Set appearance mode and color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# ============================================================================
# HIERARCHICAL BAYES MAXDIFF (NumPyro Implementation)
# ============================================================================

class HierarchicalBayesMaxDiff:
    """
    Hierarchical Bayes MaxDiff estimation using NumPyro NUTS sampler.
    
    Implements the methodology from Displayr/Sawtooth:
    - Tricked logit likelihood (best and worst as separate multinomial choices)
    - Multivariate normal population distribution
    - Non-centered parameterization for efficient sampling
    - Last item is reference (utility fixed at 0)
    """
    
    def __init__(self, n_iterations=5000, n_warmup=2500, n_chains=4, target_accept=0.8):
        """
        Initialize HB MaxDiff estimator.
        
        Parameters:
        -----------
        n_iterations : int
            Post-warmup MCMC iterations per chain (default 5000)
        n_warmup : int  
            Warmup iterations for adaptation (default 2500)
        n_chains : int
            Number of parallel MCMC chains (default 4)
        target_accept : float
            Target acceptance rate for NUTS (default 0.8)
        """
        if not HAS_NUMPYRO:
            raise ImportError(
                "NumPyro and JAX are required for Hierarchical Bayes analysis.\n"
                "Install with: pip install numpyro jax jaxlib"
            )
        
        self.n_iterations = n_iterations
        self.n_warmup = n_warmup
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.fitted = False
        
    def _prepare_data(self, df, attribute_columns, pos_col, neg_col):
        """Convert DataFrame to efficient array format for NumPyro."""
        
        # Get unique items and create mapping
        unique_items = pd.unique(df[attribute_columns].values.ravel())
        self.items = np.array([item for item in unique_items if pd.notna(item)])
        self.n_items = len(self.items)
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}
        
        # Get respondents
        self.respondents = df['Response ID'].unique()
        self.n_respondents = len(self.respondents)
        resp_to_idx = {r: i for i, r in enumerate(self.respondents)}
        
        # Determine dimensions
        tasks_per_resp = df.groupby('Response ID').size()
        self.n_tasks = int(tasks_per_resp.iloc[0])  # Assumes balanced design
        self.n_items_per_task = len(attribute_columns)
        
        # Pre-allocate arrays
        items_in_tasks = np.zeros((self.n_respondents, self.n_tasks, self.n_items_per_task), dtype=np.int32)
        best_local_idx = np.zeros((self.n_respondents, self.n_tasks), dtype=np.int32)
        worst_local_idx = np.zeros((self.n_respondents, self.n_tasks), dtype=np.int32)
        
        # Fill arrays
        for resp_id in self.respondents:
            resp_idx = resp_to_idx[resp_id]
            resp_data = df[df['Response ID'] == resp_id].reset_index(drop=True)
            
            for task_idx in range(min(len(resp_data), self.n_tasks)):
                row = resp_data.iloc[task_idx]
                
                # Get items shown in this task
                task_items = []
                for col in attribute_columns:
                    if pd.notna(row[col]) and row[col] in self.item_to_idx:
                        task_items.append(self.item_to_idx[row[col]])
                
                # Pad if necessary
                while len(task_items) < self.n_items_per_task:
                    task_items.append(0)
                
                items_in_tasks[resp_idx, task_idx, :] = task_items[:self.n_items_per_task]
                
                # Find local position of best/worst within this task
                best_global = self.item_to_idx.get(row[pos_col], 0)
                worst_global = self.item_to_idx.get(row[neg_col], 0)
                
                try:
                    best_local_idx[resp_idx, task_idx] = task_items.index(best_global)
                except ValueError:
                    best_local_idx[resp_idx, task_idx] = 0
                    
                try:
                    worst_local_idx[resp_idx, task_idx] = task_items.index(worst_global)
                except ValueError:
                    worst_local_idx[resp_idx, task_idx] = 1 if best_local_idx[resp_idx, task_idx] == 0 else 0
        
        # Convert to JAX arrays
        self.items_in_tasks = jnp.array(items_in_tasks)
        self.best_local_idx = jnp.array(best_local_idx)
        self.worst_local_idx = jnp.array(worst_local_idx)
        
    def _model(self, items_in_tasks, best_local_idx, worst_local_idx, 
               n_respondents, n_items, n_tasks, n_items_per_task):
        """
        NumPyro model specification.
        
        Uses non-centered parameterization and tricked logit likelihood.
        Last item is reference (fixed at 0) for identification during sampling.
        Zero-centering is applied in post-processing.
        """
        n_free = n_items - 1  # Number of free utility parameters
        
        # === Population-level parameters ===
        mu = numpyro.sample('mu', dist.Normal(0.0, 2.0).expand([n_free]))
        sigma = numpyro.sample('sigma', dist.HalfNormal(1.5).expand([n_free]))
        
        # === Individual-level utilities (non-centered parameterization) ===
        z = numpyro.sample('z', dist.Normal(0.0, 1.0).expand([n_respondents, n_free]))
        u_free = mu + sigma * z  # Shape: (n_respondents, n_free)
        
        # Reference item (last) fixed at 0 for identification
        u_ref = jnp.zeros((n_respondents, 1))
        u = jnp.concatenate([u_free, u_ref], axis=1)  # Shape: (n_respondents, n_items)
        
        # Population mean with reference
        mu_full = jnp.concatenate([mu, jnp.zeros(1)])
        
        # === Likelihood (Tricked Logit) ===
        resp_idx = jnp.arange(n_respondents)[:, None, None]
        u_task = u[resp_idx, items_in_tasks]
        
        log_probs_best = jax.nn.log_softmax(u_task, axis=-1)
        log_probs_worst = jax.nn.log_softmax(-u_task, axis=-1)
        
        r_idx = jnp.arange(n_respondents)[:, None]
        t_idx = jnp.arange(n_tasks)[None, :]
        
        ll_best = log_probs_best[r_idx, t_idx, best_local_idx]
        ll_worst = log_probs_worst[r_idx, t_idx, worst_local_idx]
        
        total_ll = jnp.sum(ll_best) + jnp.sum(ll_worst)
        numpyro.factor('likelihood', total_ll)
        
        # Store raw (non-centered) values - centering happens in post-processing
        numpyro.deterministic('utilities', u)
        numpyro.deterministic('mu_full', mu_full)

    def fit(self, df, attribute_columns, pos_col, neg_col, 
            progress_callback=None, log_callback=None):
        """
        Run MCMC sampling and fit the HB model.
        
        Returns:
        --------
        results_df : DataFrame
            Population-level utilities with credible intervals
        """
        if log_callback:
            log_callback("Preparing data for HB estimation...")
        
        self._prepare_data(df, attribute_columns, pos_col, neg_col)
        
        if log_callback:
            log_callback(f"  {self.n_respondents} respondents, {self.n_items} items")
            log_callback(f"  {self.n_tasks} tasks per respondent, {self.n_items_per_task} items per task")
            log_callback(f"  MCMC: {self.n_iterations} iterations + {self.n_warmup} warmup per chain")
            log_callback(f"  Running {self.n_chains} chain(s)...")
        
        if progress_callback:
            progress_callback(0.05)
        
        # Initialize NUTS sampler
        nuts_kernel = NUTS(
            self._model, 
            target_accept_prob=self.target_accept,
            max_tree_depth=10
        )
        
        # Run chains one at a time for better progress reporting
        all_samples = []
        
        for chain_idx in range(self.n_chains):
            if log_callback:
                log_callback(f"\n  ═══ Chain {chain_idx + 1}/{self.n_chains} ═══")
            
            if progress_callback:
                # Progress: 5% for prep, then 80% divided among chains, then 15% for post-processing
                chain_start_progress = 0.05 + (0.80 * chain_idx / self.n_chains)
                progress_callback(chain_start_progress)
            
            # Create MCMC for single chain
            mcmc = MCMC(
                nuts_kernel,
                num_warmup=self.n_warmup,
                num_samples=self.n_iterations,
                num_chains=1,
                progress_bar=False  # False for GUI-only, can modify to True for python env
            )
            
            # Run this chain
            rng_key = random.PRNGKey(42 + chain_idx)
            
            try:
                mcmc.run(
                    rng_key,
                    items_in_tasks=self.items_in_tasks,
                    best_local_idx=self.best_local_idx,
                    worst_local_idx=self.worst_local_idx,
                    n_respondents=self.n_respondents,
                    n_items=self.n_items,
                    n_tasks=self.n_tasks,
                    n_items_per_task=self.n_items_per_task
                )
                
                # Collect samples from this chain
                chain_samples = mcmc.get_samples()
                all_samples.append(chain_samples)
                
                if log_callback:
                    log_callback(f"  Chain {chain_idx + 1} complete ✓")
                    
            except Exception as e:
                if log_callback:
                    log_callback(f"  Chain {chain_idx + 1} failed: {str(e)}")
                raise
        
        if progress_callback:
            progress_callback(0.85)
        
        if log_callback:
            log_callback(f"\nAll {self.n_chains} chains complete. Processing results...")
        
        # Combine samples from all chains
        self.samples = {}
        for key in all_samples[0].keys():
            # Concatenate along the sample dimension (axis 0)
            self.samples[key] = np.concatenate([s[key] for s in all_samples], axis=0)
        
        if log_callback:
            total_samples = self.samples['mu'].shape[0]
            log_callback(f"  Total posterior samples: {total_samples}")
        
        # Check convergence (comparing across chains)
        self._check_convergence_across_chains(all_samples, log_callback)
        
        # Compute posterior summaries
        self._compute_summaries()
        
        self.fitted = True
        
        if progress_callback:
            progress_callback(1.0)
        
        if log_callback:
            log_callback("HB estimation complete!")
        
        return self.get_population_results()
    
    def _check_convergence_across_chains(self, all_samples, log_callback=None):
        """Check convergence by comparing chains."""
        if len(all_samples) < 2:
            if log_callback:
                log_callback("  (Single chain - skipping R-hat diagnostic)")
            return
        
        try:
            # Stack chains: (n_chains, n_samples, n_params)
            mu_by_chain = np.stack([s['mu'] for s in all_samples], axis=0)
            n_chains, n_samples, n_params = mu_by_chain.shape
            
            # Compute R-hat for each parameter
            # Between-chain variance
            chain_means = mu_by_chain.mean(axis=1)  # (n_chains, n_params)
            B = n_samples * np.var(chain_means, axis=0, ddof=1)  # (n_params,)
            
            # Within-chain variance
            chain_vars = np.var(mu_by_chain, axis=1, ddof=1)  # (n_chains, n_params)
            W = np.mean(chain_vars, axis=0)  # (n_params,)
            
            # Pooled variance estimate
            var_pooled = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
            
            # R-hat
            r_hat = np.sqrt(var_pooled / (W + 1e-10))
            max_rhat = float(np.max(r_hat))
            mean_rhat = float(np.mean(r_hat))
            
            if log_callback:
                if max_rhat < 1.05:
                    log_callback(f"  Convergence: Excellent (R-hat: mean={mean_rhat:.3f}, max={max_rhat:.3f})")
                elif max_rhat < 1.1:
                    log_callback(f"  Convergence: Good (R-hat: mean={mean_rhat:.3f}, max={max_rhat:.3f})")
                elif max_rhat < 1.2:
                    log_callback(f"  Convergence: Acceptable (R-hat: mean={mean_rhat:.3f}, max={max_rhat:.3f})")
                else:
                    log_callback(f"  ⚠️ Convergence warning! R-hat: mean={mean_rhat:.3f}, max={max_rhat:.3f}")
                    log_callback(f"     Consider running more iterations or checking the model.")
                    
        except Exception as e:
            if log_callback:
                log_callback(f"  Could not compute R-hat: {e}")
        
    def _compute_summaries(self):
        """Compute posterior summaries from MCMC samples with proper zero-centering."""
        
        # Get raw samples (last item = 0 as reference)
        mu_samples_raw = np.array(self.samples['mu_full'])  # Shape: (n_samples, n_items)
        u_samples_raw = np.array(self.samples['utilities'])  # Shape: (n_samples, n_respondents, n_items)
        
        # Zero-center EACH DRAW independently
        # This distributes uncertainty evenly across all items
        mu_samples = mu_samples_raw - mu_samples_raw.mean(axis=1, keepdims=True)
        u_samples = u_samples_raw - u_samples_raw.mean(axis=2, keepdims=True)
        
        # Now compute summaries from zero-centered samples
        self.population_mean = mu_samples.mean(axis=0)
        self.population_std = mu_samples.std(axis=0)
        self.mu_percentile_2_5 = np.percentile(mu_samples, 2.5, axis=0)
        self.mu_percentile_97_5 = np.percentile(mu_samples, 97.5, axis=0)
        
        # Individual utilities (mean across posterior samples)
        self.individual_utilities = u_samples.mean(axis=0)  # Shape: (n_respondents, n_items)
        
        # Store for preference shares
        self.mu_samples = mu_samples
        self.u_samples = u_samples
    
    def get_population_results(self, rescale='zero_centered'):
        """
        Get population-level results with credible intervals.
        
        Parameters:
        -----------
        rescale : str
            'raw' - raw utilities (logit scale, last item = 0)
            'zero_centered' - shift so mean utility = 0
            'probability' - convert to 0-100 preference share scale
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scores = self.population_mean.copy()
        lower = self.mu_percentile_2_5.copy()
        upper = self.mu_percentile_97_5.copy()
        
        if rescale == 'zero_centered':
            # Shift so mean = 0 (similar to count analysis)
            shift = scores.mean()
            scores = scores - shift
            lower = lower - shift
            upper = upper - shift
            
        elif rescale == 'probability':
            # Convert to preference shares (0-100 scale)
            # Use posterior samples for proper uncertainty propagation
            exp_mu = np.exp(self.mu_samples - self.mu_samples.max(axis=1, keepdims=True))
            share_samples = exp_mu / exp_mu.sum(axis=1, keepdims=True) * 100
            scores = share_samples.mean(axis=0)
            lower = np.percentile(share_samples, 2.5, axis=0)
            upper = np.percentile(share_samples, 97.5, axis=0)
        
        results = pd.DataFrame({
            'Item': self.items,
            'Score': scores,
            '2.5th Percentile': lower,
            '97.5th Percentile': upper,
            'Negative Error': scores - lower,
            'Positive Error': upper - scores
        }).sort_values('Score', ascending=False)
        
        return results
    
    def get_individual_utilities(self, rescale='zero_centered'):
        """
        Get individual-level utility estimates.
        
        Returns DataFrame with respondents as rows, items as columns.
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        utilities = self.individual_utilities.copy()
        
        if rescale == 'zero_centered':
            # Zero-center each respondent's utilities
            utilities = utilities - utilities.mean(axis=1, keepdims=True)
        elif rescale == 'probability':
            # Convert each respondent to preference shares
            exp_u = np.exp(utilities - utilities.max(axis=1, keepdims=True))
            utilities = exp_u / exp_u.sum(axis=1, keepdims=True) * 100
        
        return pd.DataFrame(
            utilities,
            index=self.respondents,
            columns=self.items
        )
    
    def get_preference_shares(self):
        """
        Get preference shares with credible intervals.
        
        Preference share = probability of choosing each item first
        from a hypothetical choice among all items.
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Compute shares from each posterior draw, then summarize
        exp_mu = np.exp(self.mu_samples - self.mu_samples.max(axis=1, keepdims=True))
        share_samples = exp_mu / exp_mu.sum(axis=1, keepdims=True) * 100
        
        return pd.DataFrame({
            'Item': self.items,
            'Share (%)': share_samples.mean(axis=0),
            '2.5th Percentile': np.percentile(share_samples, 2.5, axis=0),
            '97.5th Percentile': np.percentile(share_samples, 97.5, axis=0)
        }).sort_values('Share (%)', ascending=False)


# ============================================================================
# DISPLAY STATISTICS & BALANCE CHECKING
# ============================================================================

def calculate_display_statistics(df, attribute_columns, pos_col, neg_col):
    """Calculate comprehensive display statistics for each item"""
    
    unique_items = pd.unique(df[attribute_columns].values.ravel())
    unique_items = [item for item in unique_items if pd.notna(item)]
    
    stats = []
    for item in unique_items:
        displays = (df[attribute_columns] == item).sum().sum()
        pos_count = (df[pos_col] == item).sum()
        neg_count = (df[neg_col] == item).sum()
        
        stats.append({
            'Item': item,
            'Times Displayed': displays,
            'Times Selected Best': pos_count,
            'Times Selected Worst': neg_count,
            'Times Unselected': displays - pos_count - neg_count,
            'Best Rate': pos_count / displays if displays > 0 else 0,
            'Worst Rate': neg_count / displays if displays > 0 else 0,
        })
    
    stats_df = pd.DataFrame(stats).sort_values('Times Displayed', ascending=False)
    
    # Calculate balance metrics
    display_counts = stats_df['Times Displayed'].values
    mean_displays = display_counts.mean() if len(display_counts) > 0 else 0
    std_displays = display_counts.std() if len(display_counts) > 0 else 0
    
    balance_metrics = {
        'total_displays': int(display_counts.sum()),
        'num_items': len(unique_items),
        'min_displays': int(display_counts.min()) if len(display_counts) > 0 else 0,
        'max_displays': int(display_counts.max()) if len(display_counts) > 0 else 0,
        'mean_displays': float(mean_displays),
        'std_displays': float(std_displays),
        'cv_displays': float(std_displays / mean_displays) if mean_displays > 0 else 0,
        'range_displays': int(display_counts.max() - display_counts.min()) if len(display_counts) > 0 else 0,
        'is_balanced': False,
        'balance_warnings': [],
        'balance_status': 'Unknown'
    }
    
    # Determine balance status and generate warnings
    warnings = []
    
    cv = balance_metrics['cv_displays']
    if cv < 0.01:
        balance_metrics['balance_status'] = 'Perfectly Balanced'
        balance_metrics['is_balanced'] = True
    elif cv < 0.05:
        balance_metrics['balance_status'] = 'Well Balanced'
        balance_metrics['is_balanced'] = True
    elif cv < 0.10:
        balance_metrics['balance_status'] = 'Reasonably Balanced'
        balance_metrics['is_balanced'] = True
        warnings.append(f"Minor imbalance detected (CV={cv:.1%}). Results are still valid.")
    elif cv < 0.20:
        balance_metrics['balance_status'] = 'Somewhat Unbalanced'
        balance_metrics['is_balanced'] = False
        warnings.append(f"⚠️ Moderate imbalance detected (CV={cv:.1%}). Consider this when interpreting results.")
    else:
        balance_metrics['balance_status'] = 'Highly Unbalanced'
        balance_metrics['is_balanced'] = False
        warnings.append(f"⚠️ HIGH IMBALANCE detected (CV={cv:.1%}). Results may be biased!")
    
    # Check for specific outliers
    mean_disp = balance_metrics['mean_displays']
    std_disp = balance_metrics['std_displays']
    
    if std_disp > 0:
        under_displayed = stats_df[stats_df['Times Displayed'] < mean_disp - 2*std_disp]['Item'].tolist()
        over_displayed = stats_df[stats_df['Times Displayed'] > mean_disp + 2*std_disp]['Item'].tolist()
        
        if under_displayed:
            items_str = ', '.join(str(x) for x in under_displayed[:5])
            warnings.append(f"⚠️ Under-displayed items (>2 SD below mean): {items_str}")
            if len(under_displayed) > 5:
                warnings[-1] += f" and {len(under_displayed)-5} more"
        
        if over_displayed:
            items_str = ', '.join(str(x) for x in over_displayed[:5])
            warnings.append(f"⚠️ Over-displayed items (>2 SD above mean): {items_str}")
            if len(over_displayed) > 5:
                warnings[-1] += f" and {len(over_displayed)-5} more"
    
    # Check for very low display counts
    low_threshold = 30
    low_display_items = stats_df[stats_df['Times Displayed'] < low_threshold]['Item'].tolist()
    if low_display_items:
        items_str = ', '.join(str(x) for x in low_display_items[:5])
        warnings.append(f"⚠️ Low display counts (<{low_threshold}): {items_str}")
        if len(low_display_items) > 5:
            warnings[-1] += f" and {len(low_display_items)-5} more"
        warnings.append("   Low counts may lead to unreliable estimates for these items.")
    
    # Check if all items have at least some best/worst selections
    never_best = stats_df[stats_df['Times Selected Best'] == 0]['Item'].tolist()
    never_worst = stats_df[stats_df['Times Selected Worst'] == 0]['Item'].tolist()
    
    if never_best:
        items_str = ', '.join(str(x) for x in never_best[:5])
        warnings.append(f"ℹ️ Items never selected as best: {items_str}")
        if len(never_best) > 5:
            warnings[-1] += f" and {len(never_best)-5} more"
    
    if never_worst:
        items_str = ', '.join(str(x) for x in never_worst[:5])
        warnings.append(f"ℹ️ Items never selected as worst: {items_str}")
        if len(never_worst) > 5:
            warnings[-1] += f" and {len(never_worst)-5} more"
    
    balance_metrics['balance_warnings'] = warnings
    
    return stats_df, balance_metrics


def format_display_report(stats_df, balance_metrics, output_terms):
    """Format a nice text report of display statistics"""
    pos_label, neg_label = output_terms
    
    lines = []
    lines.append("=" * 60)
    lines.append("📊 DISPLAY STATISTICS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total items: {balance_metrics['num_items']}")
    lines.append(f"Total displays across all tasks: {balance_metrics['total_displays']:,}")
    lines.append("")
    lines.append("BALANCE ASSESSMENT:")
    lines.append(f"  Status: {balance_metrics['balance_status']}")
    lines.append(f"  Min displays: {balance_metrics['min_displays']:,}")
    lines.append(f"  Max displays: {balance_metrics['max_displays']:,}")
    lines.append(f"  Mean displays: {balance_metrics['mean_displays']:,.1f}")
    lines.append(f"  Std deviation: {balance_metrics['std_displays']:,.1f}")
    lines.append(f"  Coefficient of variation: {balance_metrics['cv_displays']:.1%}")
    lines.append("")
    
    if balance_metrics['balance_warnings']:
        lines.append("WARNINGS & NOTES:")
        for warning in balance_metrics['balance_warnings']:
            lines.append(f"  {warning}")
        lines.append("")
    
    lines.append("DISPLAY COUNTS PER ITEM:")
    lines.append("-" * 60)
    lines.append(f"{'Item':<30} {'Displays':>10} {pos_label:>10} {neg_label:>10}")
    lines.append("-" * 60)
    
    for _, row in stats_df.iterrows():
        item_name = str(row['Item'])[:28]
        lines.append(f"{item_name:<30} {row['Times Displayed']:>10,} {row['Times Selected Best']:>10,} {row['Times Selected Worst']:>10,}")
    
    lines.append("-" * 60)
    lines.append("")
    
    return "\n".join(lines)


def plot_display_balance(stats_df, title="Item Display Frequency", output_terms=("Best", "Worst")):
    """Create a visualization of display balance"""
    pos_label, neg_label = output_terms
    
    stats_sorted = stats_df.sort_values('Times Displayed', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(stats_df) * 0.3)))
    
    # Left plot: Display counts
    ax1 = axes[0]
    mean_val = stats_df['Times Displayed'].mean()
    colors = ['#4CAF50' if x >= mean_val else '#FF9800' 
              for x in stats_sorted['Times Displayed']]
    
    y_pos = range(len(stats_sorted))
    ax1.barh(y_pos, stats_sorted['Times Displayed'], color=colors, edgecolor='white')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([str(x)[:30] for x in stats_sorted['Item']], fontsize=9)
    ax1.set_xlabel('Times Displayed')
    ax1.set_title('Display Frequency per Item')
    
    ax1.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
    ax1.legend(loc='lower right')
    
    for i, (idx, row) in enumerate(stats_sorted.iterrows()):
        ax1.text(row['Times Displayed'] + 0.5, i, f"{row['Times Displayed']:,}", 
                va='center', fontsize=8)
    
    # Right plot: Selection rates
    ax2 = axes[1]
    
    bar_height = 0.35
    y_pos = np.arange(len(stats_sorted))
    
    ax2.barh(y_pos - bar_height/2, stats_sorted['Best Rate'] * 100, bar_height,
            label=f'% {pos_label}', color='#FFC000')
    ax2.barh(y_pos + bar_height/2, stats_sorted['Worst Rate'] * 100, bar_height,
            label=f'% {neg_label}', color='#5B9BD5')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([str(x)[:30] for x in stats_sorted['Item']], fontsize=9)
    ax2.set_xlabel('Selection Rate (%)')
    ax2.set_title('Selection Rates per Item')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 100)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# DATA FORMAT DETECTION AND CONVERSION
# ============================================================================

class DataFormatDetector:
    """Detects and converts various MaxDiff data formats"""
    
    @staticmethod
    def detect_format(df):
        cols = [c.lower() for c in df.columns]
        col_list = list(df.columns)
        
        has_response_id = any('response' in c and 'id' in c for c in cols) or 'responseid' in cols
        has_attributes = any(c.startswith('attribute') for c in cols)
        has_most_least = ('most' in cols and 'least' in cols) or ('best' in cols and 'worst' in cols)
        
        if has_response_id and has_attributes and has_most_least:
            return 'ready', "✓ Data appears to be in the correct format!"
        
        qualtrics_pattern = re.compile(r'(Q\d+|MD\d+|MaxDiff\d*)[-_]([\dA-Za-z]+)', re.IGNORECASE)
        qualtrics_matches = [c for c in col_list if qualtrics_pattern.match(c)]
        
        if len(qualtrics_matches) > 5:
            return 'qualtrics_wide', f"Detected Qualtrics-style format ({len(qualtrics_matches)} MaxDiff columns)"
        
        long_indicators = ['item', 'attribute', 'option', 'alternative', 'choice']
        selection_indicators = ['selected', 'chosen', 'response', 'answer', 'picked']
        task_indicators = ['task', 'set', 'question', 'trial', 'screen']
        
        has_item_col = any(any(ind in c for ind in long_indicators) for c in cols)
        has_selection_col = any(any(ind in c for ind in selection_indicators) for c in cols)
        has_task_col = any(any(ind in c for ind in task_indicators) for c in cols)
        
        if has_item_col and (has_selection_col or has_task_col):
            return 'long', "Detected long format (one row per item)"
        
        task_pattern = re.compile(r'(task|set|q)[-_]?(\d+)[-_]?(attr|item|opt|best|worst|most|least|\d+)', re.IGNORECASE)
        task_matches = [c for c in col_list if task_pattern.match(c)]
        
        if len(task_matches) > 5:
            return 'wide_by_respondent', f"Detected wide format ({len(task_matches)} task columns)"
        
        return 'unknown', "Could not auto-detect format. Please use manual column mapping."
    
    @staticmethod
    def get_format_description(format_type):
        descriptions = {
            'ready': "Your data is already in the correct format! You can proceed directly to analysis.",
            'qualtrics_wide': "Qualtrics format detected. The converter will reshape columns like Q1_1, Q1_2, Q1_Best into the required format.",
            'long': "Long format detected (one row per item). The converter will pivot this into wide format.",
            'wide_by_respondent': "Wide-by-respondent format detected. The converter will reshape into multiple rows per respondent.",
            'unknown': "Format not recognized. Use the Column Mapper to manually specify your columns."
        }
        return descriptions.get(format_type, "")
    
    @staticmethod
    def convert_qualtrics_wide(df, task_prefix='Q', best_suffix='Best', worst_suffix='Worst', id_column=None):
        if id_column is None:
            id_candidates = [c for c in df.columns if 'response' in c.lower() or 'id' in c.lower()]
            id_column = id_candidates[0] if id_candidates else df.columns[0]
        
        task_pattern = re.compile(
            rf'({task_prefix}\d+)[-_](\d+|{best_suffix}|{worst_suffix}|Best|Worst|Most|Least)', 
            re.IGNORECASE
        )
        
        tasks = defaultdict(dict)
        for col in df.columns:
            match = task_pattern.match(col)
            if match:
                task_name = match.group(1).upper()
                suffix = match.group(2)
                tasks[task_name][suffix] = col
        
        if not tasks:
            raise ValueError(f"No task columns found matching pattern '{task_prefix}N_X'")
        
        rows = []
        for _, respondent in df.iterrows():
            resp_id = respondent[id_column]
            
            for task_name in sorted(tasks.keys()):
                task_cols = tasks[task_name]
                
                attr_cols = {k: v for k, v in task_cols.items() 
                           if k.isdigit() or k.lower() not in ['best', 'worst', 'most', 'least', 
                                                                best_suffix.lower(), worst_suffix.lower()]}
                
                best_col = worst_col = None
                for suffix, col in task_cols.items():
                    if suffix.lower() in ['best', 'most', best_suffix.lower()]:
                        best_col = col
                    elif suffix.lower() in ['worst', 'least', worst_suffix.lower()]:
                        worst_col = col
                
                if not attr_cols or not best_col or not worst_col:
                    continue
                
                row = {'Response ID': resp_id}
                for i, (suffix, col) in enumerate(sorted(attr_cols.items()), 1):
                    row[f'Attribute{i}'] = respondent[col]
                
                row['Most'] = respondent[best_col]
                row['Least'] = respondent[worst_col]
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def convert_long_format(df, id_col, task_col, item_col, selection_col,
                           most_value='Most', least_value='Least'):
        rows = []
        grouped = df.groupby([id_col, task_col])
        
        for (resp_id, task), group in grouped:
            row = {'Response ID': resp_id}
            
            items = group[item_col].tolist()
            for i, item in enumerate(items, 1):
                row[f'Attribute{i}'] = item
            
            most_mask = group[selection_col].astype(str).str.lower().str.contains(most_value.lower(), na=False)
            least_mask = group[selection_col].astype(str).str.lower().str.contains(least_value.lower(), na=False)
            
            most_item = group.loc[most_mask, item_col]
            least_item = group.loc[least_mask, item_col]
            
            row['Most'] = most_item.iloc[0] if len(most_item) > 0 else None
            row['Least'] = least_item.iloc[0] if len(least_item) > 0 else None
            
            rows.append(row)
        
        return pd.DataFrame(rows)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def process_color_input(color):
    if not color:
        return None
    color_str = str(color).strip()
    if color_str.lower() in mcolors.CSS4_COLORS:
        return color_str.lower()
    if color_str.startswith('#'):
        return color_str
    if len(color_str) == 6 and all(c in '0123456789ABCDEFabcdef' for c in color_str):
        return f'#{color_str}'
    try:
        return mcolors.to_hex(color_str)
    except ValueError:
        return None


def detect_terminology(df):
    cols = [c.lower() for c in df.columns]
    if 'most' in cols and 'least' in cols:
        return 'Most/Least'
    elif 'best' in cols and 'worst' in cols:
        return 'Best/Worst'
    return None


def get_column_names(df, input_terminology):
    cols_lower = {c.lower(): c for c in df.columns}
    if input_terminology == 'Most/Least':
        return cols_lower.get('most', 'Most'), cols_lower.get('least', 'Least')
    else:
        return cols_lower.get('best', 'Best'), cols_lower.get('worst', 'Worst')


def check_errors(df, attribute_columns, pos_col, neg_col):
    responses_per_participant = df.groupby('Response ID').size()
    if not (responses_per_participant == responses_per_participant.iloc[0]).all():
        raise ValueError("Inconsistent number of responses per participant")
    if len(attribute_columns) < 3:
        raise ValueError("Less than 3 attributes found")
    columns_to_check = attribute_columns + [pos_col, neg_col]
    if df[columns_to_check].isnull().any().any():
        raise ValueError(f"Missing data in {pos_col}, {neg_col}, or attribute columns")
    for idx, row in df.iterrows():
        displayed_attributes = set(row[attribute_columns])
        if row[pos_col] not in displayed_attributes:
            raise ValueError(f"Row {idx}: Selection '{row[pos_col]}' not in displayed attributes")
        if row[neg_col] not in displayed_attributes:
            raise ValueError(f"Row {idx}: Selection '{row[neg_col]}' not in displayed attributes")


def calculate_observed_percentages(df, attribute_columns, pos_col, neg_col, output_terms):
    pos_label, neg_label = output_terms
    results = []
    unique_items = pd.unique(df[attribute_columns].values.ravel())
    unique_items = [item for item in unique_items if pd.notna(item)]
    
    for item in unique_items:
        item_displays = (df[attribute_columns] == item).sum().sum()
        pos_count = (df[pos_col] == item).sum()
        neg_count = (df[neg_col] == item).sum()
        unselected_count = item_displays - pos_count - neg_count
        results.append({
            'Item': item,
            f'% Selected as {pos_label}': pos_count / item_displays * 100 if item_displays > 0 else 0,
            '% Unselected': unselected_count / item_displays * 100 if item_displays > 0 else 0,
            f'% Selected as {neg_label}': neg_count / item_displays * 100 if item_displays > 0 else 0,
            'Score': (pos_count - neg_count) / item_displays * 100 if item_displays > 0 else 0
        })
    return pd.DataFrame(results).sort_values('Score', ascending=False)


def calculate_scores_no_ci(df, attribute_columns, pos_col, neg_col):
    unique_attributes = pd.unique(df[attribute_columns].values.ravel())
    unique_attributes = [attr for attr in unique_attributes if pd.notna(attr)]
    results = []
    for item in unique_attributes:
        item_displays = (df[attribute_columns] == item).sum().sum()
        pos_count = (df[pos_col] == item).sum()
        neg_count = (df[neg_col] == item).sum()
        results.append({
            'Item': item, 
            'Score': (pos_count - neg_count) / item_displays * 100 if item_displays > 0 else 0
        })
    return pd.DataFrame(results).sort_values('Score', ascending=False)


def perform_maxdiff_analysis(attribute_data, pos_data, neg_data, unique_attributes):
    n_attributes = len(unique_attributes)
    display_count = np.sum(attribute_data[:, :, None] == np.arange(n_attributes), axis=(0, 1))
    pos_count = np.bincount(pos_data, minlength=n_attributes)
    neg_count = np.bincount(neg_data, minlength=n_attributes)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        score = np.where(display_count > 0, 
                        ((pos_count / display_count) - (neg_count / display_count)) * 100, 
                        0)
    
    return pd.DataFrame({'Item': unique_attributes, 'Score': score})


def bootstrap_analysis(df, attribute_columns, unique_attributes, attr_to_index, 
                       pos_col, neg_col, n_iterations=10000, progress_callback=None):
    unique_participants = df['Response ID'].unique()
    n_participants = len(unique_participants)
    n_attributes = len(unique_attributes)
    df_reset = df.reset_index(drop=True)
    attribute_data = df_reset[attribute_columns].replace(attr_to_index).values
    pos_data = df_reset[pos_col].replace(attr_to_index).values
    neg_data = df_reset[neg_col].replace(attr_to_index).values
    participant_indices = df_reset.groupby('Response ID').indices
    observed_results = perform_maxdiff_analysis(attribute_data, pos_data, neg_data, unique_attributes)
    observed_scores = observed_results.set_index('Item')['Score']
    all_scores = np.zeros((n_iterations, n_attributes))
    rng = np.random.default_rng()
    
    for i in range(n_iterations):
        sampled_participants = rng.choice(unique_participants, size=n_participants, replace=True)
        sampled_indices = np.concatenate([participant_indices[p] for p in sampled_participants])
        results = perform_maxdiff_analysis(
            attribute_data[sampled_indices], pos_data[sampled_indices], 
            neg_data[sampled_indices], unique_attributes)
        all_scores[i] = results['Score'].values
        if progress_callback and i % 100 == 0:
            progress_callback(i / n_iterations)
    
    if progress_callback:
        progress_callback(1.0)
    
    percentile_2_5 = np.percentile(all_scores, 2.5, axis=0)
    percentile_97_5 = np.percentile(all_scores, 97.5, axis=0)
    obs_scores_array = observed_scores[unique_attributes].values
    
    return pd.DataFrame({
        'Item': unique_attributes, 
        'Score': obs_scores_array,
        '2.5th Percentile': percentile_2_5, 
        '97.5th Percentile': percentile_97_5,
        'Negative Error': obs_scores_array - percentile_2_5,
        'Positive Error': percentile_97_5 - obs_scores_array
    }).sort_values('Score', ascending=False)


def plot_observed_percentages(df, title, output_terms):
    pos_label, neg_label = output_terms
    pos_col = f'% Selected as {pos_label}'
    neg_col = f'% Selected as {neg_label}'
    df = df.sort_values('Score', ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.4)))
    y_pos = range(len(df))
    ax.barh(y_pos, df[pos_col], color='#FFC000', label=pos_col)
    ax.barh(y_pos, df['% Unselected'], left=df[pos_col], color='#D9D9D9', label='% Unselected')
    ax.barh(y_pos, df[neg_col], left=df[pos_col] + df['% Unselected'], color='#5B9BD5', label=neg_col)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Item'], fontsize=8)
    ax.set_title(title)
    ax.legend(loc='lower right', bbox_to_anchor=(1, -0.1), ncol=3)
    
    for i, (pos, unsel, neg) in enumerate(zip(df[pos_col], df['% Unselected'], df[neg_col])):
        if pos > 5:
            ax.text(pos/2, i, f'{pos:.0f}%', va='center', ha='center', fontsize=8)
        if unsel > 5:
            ax.text(pos + unsel/2, i, f'{unsel:.0f}%', va='center', ha='center', fontsize=8)
        if neg > 5:
            ax.text(100 - neg/2, i, f'{neg:.0f}%', va='center', ha='center', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_scores(results, positive_color, negative_color, error_bar_color, zero_line_color,
                anchor_item_color, anchor_item_error_color, title="MaxDiff Scores", 
                sample_size=None, segment_info=None, anchor_item=None, include_ci=True):
    sorted_results = results.sort_values('Score', ascending=True).reset_index(drop=True)
    num_items = len(sorted_results)
    fig, ax = plt.subplots(figsize=(13, max(0.4 * num_items, 10)), dpi=100)
    wrapped_labels = [textwrap.fill(str(label), width=50) for label in sorted_results['Item']]
    has_ci = include_ci and 'Negative Error' in sorted_results.columns and 'Positive Error' in sorted_results.columns
    
    for i, row in sorted_results.iterrows():
        item = row['Item']
        score = row['Score']
        
        if item == anchor_item:
            point_color, error_color = anchor_item_color, anchor_item_error_color
        else:
            point_color = positive_color if score >= 0 else negative_color
            error_color = error_bar_color
        
        if has_ci:
            neg_err = row['Negative Error']
            pos_err = row['Positive Error']
            ax.errorbar(score, i, xerr=[[neg_err], [pos_err]], fmt='o', capsize=5, 
                       capthick=2, color=point_color, markersize=8, ecolor=error_color, elinewidth=2)
        else:
            ax.plot(score, i, 'o', color=point_color, markersize=8)
    
    if segment_info:
        title += f", {segment_info}"
    if sample_size:
        title += f" (n={sample_size})"
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_yticks(range(len(sorted_results)))
    ax.set_yticklabels(wrapped_labels, fontsize=11)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.axvline(x=0, color=zero_line_color, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig


def calculate_correlation_matrix(df, attribute_columns, pos_col, neg_col):
    unique_items = pd.unique(df[attribute_columns].values.ravel())
    unique_items = [item for item in unique_items if pd.notna(item)]
    count_df = pd.DataFrame(index=df['Response ID'].unique(), columns=unique_items, data=0)
    
    for _, row in df.iterrows():
        count_df.loc[row['Response ID'], row[pos_col]] += 1
        count_df.loc[row['Response ID'], row[neg_col]] -= 1
    
    corr_matrix = count_df.astype(float).corr()
    return corr_matrix


def plot_correlation_matrix(corr_matrix, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    return fig


def save_plot(fig, filename, base_dir):
    for ext in ['png', 'pdf']:
        filepath = base_dir / 'plots' / ext / f'{filename}.{ext}'
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def save_dataframe(df, filename, base_dir, include_index=False):
    for ext in ['csv', 'xlsx']:
        filepath = base_dir / 'data' / ext / f'{filename}.{ext}'
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if ext == 'csv':
            df.to_csv(filepath, index=include_index)
        else:
            df.to_excel(filepath, index=include_index)


def save_results(results, base_dir, prefix='', sample_size=None,
                 positive_color='#F4B400', negative_color='#F4B400',
                 error_bar_color='#a1a1a1', zero_line_color='red',
                 anchor_item_color='#a1a1a1', anchor_item_error_color='#a1a1a1',
                 segment_info=None, anchor_item=None, output_terms=('Most', 'Least'),
                 include_ci=True):
    
    # Map internal result types to cleaner output names
    filename_map = {
        'bootstrap_results': 'net_scores',
        'scores': 'net_scores', 
        'hb_utilities': 'hb_utilities',
        'observed_percentages': 'selection_frequencies'
    }
    
    title_map = {
        'bootstrap_results': 'MaxDiff Net Scores',
        'scores': 'MaxDiff Net Scores',
        'hb_utilities': 'HB Utilities',
        'observed_percentages': 'Selection Frequencies'
    }
    
    for result_type, data in results.items():
        # Get clean filename
        clean_name = filename_map.get(result_type, result_type)
        filename = f"{prefix}_{clean_name}_n{sample_size}" if prefix else f"overall_{clean_name}_n{sample_size}"
        
        save_dataframe(data, filename, base_dir)
        
        if result_type in ['bootstrap_results', 'scores', 'hb_utilities']:
            title = title_map.get(result_type, "MaxDiff Scores")
            if result_type == 'scores' and not include_ci:
                title += " (No CI)"
            if segment_info:
                title += f", {segment_info}"
            fig = plot_scores(data, positive_color, negative_color, error_bar_color, 
                            zero_line_color, anchor_item_color, anchor_item_error_color,
                            title=title, sample_size=sample_size, anchor_item=anchor_item, 
                            include_ci=include_ci)
            save_plot(fig, f"{filename}_plot", base_dir)
            
        elif result_type == 'observed_percentages':
            title = title_map.get(result_type, "Selection Frequencies")
            if segment_info:
                title += f", {segment_info}"
            if sample_size:
                title += f" (n={sample_size})"
            fig = plot_observed_percentages(data, title, output_terms)
            save_plot(fig, f"{filename}_plot", base_dir)


def segment_maxdiff_analysis(maxdiff_df, segment_df, pos_col, neg_col, output_terms,
                              n_iterations=10000, anchor_item=None, include_ci=True,
                              progress_callback=None, log_callback=None,
                              analysis_method='count', hb_settings=None):
    results = {}
    segment_columns = [col for col in segment_df.columns if col != 'Response ID']
    attribute_columns = [col for col in maxdiff_df.columns if col.startswith('Attribute')]
    unique_attributes = pd.unique(maxdiff_df[attribute_columns].values.ravel('K'))
    unique_attributes = unique_attributes[~pd.isnull(unique_attributes)]
    attr_to_index = {attr: i for i, attr in enumerate(unique_attributes)}
    merged_df = pd.merge(maxdiff_df, segment_df[['Response ID'] + segment_columns], on='Response ID', how='left')
    
    total_segments = sum(len(merged_df[col].dropna().unique()) for col in segment_columns)
    current_segment = 0
    
    for column in segment_columns:
        if log_callback:
            log_callback(f"Processing segment column: {column}")
        column_results = {}
        
        for segment in merged_df[column].dropna().unique():
            if log_callback:
                log_callback(f"  Processing segment: {segment}")
            segment_data = merged_df[merged_df[column] == segment]
            
            if len(segment_data) == 0:
                continue
            
            sample_size = segment_data['Response ID'].nunique()
            
            # Calculate display stats for segment
            seg_display_stats, seg_balance = calculate_display_statistics(
                segment_data, attribute_columns, pos_col, neg_col
            )
            
            # Run appropriate analysis
            if analysis_method == 'hb' and HAS_NUMPYRO:
                try:
                    hb_model = HierarchicalBayesMaxDiff(
                        n_iterations=hb_settings.get('iterations', 5000),
                        n_warmup=hb_settings.get('warmup', 2500),
                        n_chains=hb_settings.get('chains', 4)
                    )
                    def seg_progress(p):
                        if progress_callback:
                            progress_callback((current_segment + p) / total_segments)
                    segment_results = hb_model.fit(
                        segment_data, attribute_columns, pos_col, neg_col,
                        progress_callback=seg_progress, log_callback=None
                    )
                    result_key = 'hb_utilities'
                except Exception as e:
                    if log_callback:
                        log_callback(f"    HB failed for segment, falling back to count: {e}")
                    segment_results = calculate_scores_no_ci(segment_data, attribute_columns, pos_col, neg_col)
                    result_key = 'scores'
            elif include_ci:
                def seg_progress(p):
                    if progress_callback:
                        progress_callback((current_segment + p) / total_segments)
                segment_results = bootstrap_analysis(segment_data, attribute_columns, unique_attributes,
                    attr_to_index, pos_col, neg_col, n_iterations, seg_progress)
                result_key = 'bootstrap_results'
            else:
                segment_results = calculate_scores_no_ci(segment_data, attribute_columns, pos_col, neg_col)
                result_key = 'scores'
                if progress_callback:
                    progress_callback((current_segment + 1) / total_segments)
            
            segment_observed = calculate_observed_percentages(segment_data, attribute_columns, pos_col, neg_col, output_terms)
            segment_corr = calculate_correlation_matrix(segment_data, attribute_columns, pos_col, neg_col)
            
            column_results[segment] = {
                'sample_size': sample_size, 
                result_key: segment_results,
                'observed_percentages': segment_observed, 
                'correlation_matrix': segment_corr,
                'display_statistics': seg_display_stats,
                'balance_metrics': seg_balance
            }
            current_segment += 1
        
        if column_results:
            results[column] = column_results
    
    return results


def process_segment_results(segment_results, base_dir, colors, output_terms, anchor_item=None, include_ci=True):
    for column, column_results in segment_results.items():
        for segment, segment_data in column_results.items():
            safe_column = ''.join(c if c.isalnum() else '_' for c in str(column))
            safe_segment = ''.join(c if c.isalnum() else '_' for c in str(segment))
            prefix = f"{safe_column}_{safe_segment}"
            sample_size = segment_data['sample_size']
            segment_info = f"{column}: {segment}"
            
            # Find the result key
            score_key = None
            for key in ['hb_utilities', 'bootstrap_results', 'scores']:
                if key in segment_data:
                    score_key = key
                    break
            
            if score_key:
                save_results(
                    {score_key: segment_data[score_key], 'observed_percentages': segment_data['observed_percentages']},
                    base_dir, prefix, sample_size, **colors, segment_info=segment_info, anchor_item=anchor_item,
                    output_terms=output_terms, include_ci=include_ci
                )
            
            # Save display statistics for segment
            save_dataframe(segment_data['display_statistics'], 
                          f"{prefix}_display_statistics_n{sample_size}", base_dir)
            
            # Save display balance plot for segment
            balance_fig = plot_display_balance(
                segment_data['display_statistics'],
                title=f"Display Balance - {segment_info} (n={sample_size})",
                output_terms=output_terms
            )
            save_plot(balance_fig, f"{prefix}_display_balance_n{sample_size}", base_dir)
            
            corr_matrix = segment_data['correlation_matrix']
            save_dataframe(corr_matrix, f"{prefix}_correlation_matrix_n{sample_size}", base_dir, include_index=True)
            corr_fig = plot_correlation_matrix(corr_matrix, f"Correlation Matrix, {segment_info} (n={sample_size})")
            save_plot(corr_fig, f"{prefix}_correlation_matrix_n{sample_size}_plot", base_dir)


# ============================================================================
# GUI COMPONENTS
# ============================================================================

class ColorButton(ctk.CTkFrame):
    def __init__(self, master, label, default_color, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.color = default_color
        self.label = ctk.CTkLabel(self, text=label, width=150, anchor="w")
        self.label.pack(side="left", padx=(0, 10))
        self.color_preview = ctk.CTkButton(
            self, width=40, height=28, text="",
            fg_color=default_color, hover_color=default_color,
            border_width=2, border_color="#999999", command=self.pick_color
        )
        self.color_preview.pack(side="left", padx=(0, 10))
        self.color_entry = ctk.CTkEntry(self, width=100, placeholder_text="#RRGGBB")
        self.color_entry.insert(0, default_color)
        self.color_entry.pack(side="left")
        self.color_entry.bind("<Return>", self.update_from_entry)
        self.color_entry.bind("<FocusOut>", self.update_from_entry)
    
    def pick_color(self):
        color = colorchooser.askcolor(color=self.color, title="Choose Color")
        if color[1]:
            self.set_color(color[1])
    
    def update_from_entry(self, event=None):
        color = process_color_input(self.color_entry.get())
        if color:
            self.set_color(color)
        else:
            self.color_entry.delete(0, "end")
            self.color_entry.insert(0, self.color)
    
    def set_color(self, color):
        self.color = color
        self.color_preview.configure(fg_color=color, hover_color=color)
        self.color_entry.delete(0, "end")
        self.color_entry.insert(0, color)
    
    def get_color(self):
        return self.color


class CollapsibleFrame(ctk.CTkFrame):
    def __init__(self, master, title, expanded=False, **kwargs):
        super().__init__(master, **kwargs)
        self.expanded = expanded
        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.pack(fill="x", padx=10, pady=(10, 5))
        self.toggle_btn = ctk.CTkButton(
            self.header, text="▼" if expanded else "▶", width=30, height=28,
            command=self.toggle, fg_color="transparent", text_color=("gray20", "gray80"),
            hover_color=("gray80", "gray30")
        )
        self.toggle_btn.pack(side="left")
        self.title_label = ctk.CTkLabel(self.header, text=title, font=ctk.CTkFont(size=14, weight="bold"))
        self.title_label.pack(side="left", padx=5)
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        if expanded:
            self.content.pack(fill="x", padx=15, pady=(0, 10))
    
    def toggle(self):
        if self.expanded:
            self.content.pack_forget()
            self.toggle_btn.configure(text="▶")
        else:
            self.content.pack(fill="x", padx=15, pady=(0, 10))
            self.toggle_btn.configure(text="▼")
        self.expanded = not self.expanded
    
    def get_content_frame(self):
        return self.content


class DataPreviewWindow(ctk.CTkToplevel):
    def __init__(self, master, df, detected_format, format_message, on_convert_callback):
        super().__init__(master)
        self.title("Data Preview & Conversion")
        self.geometry("1000x700")
        self.df = df
        self.detected_format = detected_format
        self.on_convert = on_convert_callback
        self.create_widgets(format_message)
        self.transient(master)
        self.grab_set()
    
    def create_widgets(self, format_message):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20, 10))
        ctk.CTkLabel(
            header, text="📋 Data Preview & Format Conversion",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(anchor="w")
        
        detection_frame = ctk.CTkFrame(self)
        detection_frame.pack(fill="x", padx=20, pady=10)
        status_color = "green" if self.detected_format == 'ready' else "orange"
        ctk.CTkLabel(
            detection_frame, text=f"🔍 {format_message}",
            font=ctk.CTkFont(size=14), text_color=status_color
        ).pack(padx=15, pady=10, anchor="w")
        
        desc_text = DataFormatDetector.get_format_description(self.detected_format)
        ctk.CTkLabel(
            detection_frame, text=desc_text, font=ctk.CTkFont(size=12),
            wraplength=900, justify="left"
        ).pack(padx=15, pady=(0, 10), anchor="w")
        
        preview_frame = ctk.CTkFrame(self)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=10)
        ctk.CTkLabel(
            preview_frame, text="Data Preview (first 10 rows):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        tree_frame = ctk.CTkFrame(preview_frame)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
        x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        y_scroll.config(command=self.tree.yview)
        x_scroll.config(command=self.tree.xview)
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)
        
        self.tree["columns"] = list(self.df.columns)
        self.tree["show"] = "headings"
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, minwidth=50)
        for _, row in self.df.head(10).iterrows():
            self.tree.insert("", "end", values=list(row))
        
        if self.detected_format != 'ready':
            self.create_conversion_options()
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        if self.detected_format == 'ready':
            ctk.CTkButton(
                btn_frame, text="✓ Use This Data", width=150,
                command=self.use_directly
            ).pack(side="right", padx=5)
        else:
            ctk.CTkButton(
                btn_frame, text="🔄 Convert Data", width=150,
                command=self.convert_data
            ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            btn_frame, text="Cancel", width=100, fg_color="gray",
            command=self.destroy
        ).pack(side="right", padx=5)
        ctk.CTkButton(
            btn_frame, text="📖 Column Mapper", width=150,
            command=self.open_column_mapper
        ).pack(side="left", padx=5)
    
    def create_conversion_options(self):
        options_frame = ctk.CTkFrame(self)
        options_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(
            options_frame, text="⚙️ Conversion Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        settings_grid = ctk.CTkFrame(options_frame, fg_color="transparent")
        settings_grid.pack(fill="x", padx=10, pady=10)
        
        if self.detected_format == 'qualtrics_wide':
            row1 = ctk.CTkFrame(settings_grid, fg_color="transparent")
            row1.pack(fill="x", pady=2)
            ctk.CTkLabel(row1, text="Task prefix:", width=200, anchor="w").pack(side="left")
            self.task_prefix_entry = ctk.CTkEntry(row1, width=100)
            self.task_prefix_entry.insert(0, "Q")
            self.task_prefix_entry.pack(side="left")
            
            row2 = ctk.CTkFrame(settings_grid, fg_color="transparent")
            row2.pack(fill="x", pady=2)
            ctk.CTkLabel(row2, text="'Best' suffix:", width=200, anchor="w").pack(side="left")
            self.best_suffix_entry = ctk.CTkEntry(row2, width=100)
            self.best_suffix_entry.insert(0, "Best")
            self.best_suffix_entry.pack(side="left")
            
            row3 = ctk.CTkFrame(settings_grid, fg_color="transparent")
            row3.pack(fill="x", pady=2)
            ctk.CTkLabel(row3, text="'Worst' suffix:", width=200, anchor="w").pack(side="left")
            self.worst_suffix_entry = ctk.CTkEntry(row3, width=100)
            self.worst_suffix_entry.insert(0, "Worst")
            self.worst_suffix_entry.pack(side="left")
            
            row4 = ctk.CTkFrame(settings_grid, fg_color="transparent")
            row4.pack(fill="x", pady=2)
            ctk.CTkLabel(row4, text="ID column:", width=200, anchor="w").pack(side="left")
            self.id_col_menu = ctk.CTkComboBox(row4, values=list(self.df.columns), width=200)
            id_candidates = [c for c in self.df.columns if 'response' in c.lower() or 'id' in c.lower()]
            if id_candidates:
                self.id_col_menu.set(id_candidates[0])
            self.id_col_menu.pack(side="left")
        
        elif self.detected_format == 'long':
            cols = list(self.df.columns)
            for label, attr in [
                ("Response ID column:", "long_id_col"),
                ("Task column:", "long_task_col"),
                ("Item column:", "long_item_col"),
                ("Selection column:", "long_selection_col")
            ]:
                row = ctk.CTkFrame(settings_grid, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=label, width=200, anchor="w").pack(side="left")
                combo = ctk.CTkComboBox(row, values=cols, width=200)
                combo.pack(side="left")
                setattr(self, attr, combo)
    
    def use_directly(self):
        self.on_convert(self.df)
        self.destroy()
    
    def convert_data(self):
        try:
            if self.detected_format == 'qualtrics_wide':
                converted_df = DataFormatDetector.convert_qualtrics_wide(
                    self.df, 
                    self.task_prefix_entry.get(), 
                    self.best_suffix_entry.get(),
                    self.worst_suffix_entry.get(), 
                    self.id_col_menu.get()
                )
            elif self.detected_format == 'long':
                converted_df = DataFormatDetector.convert_long_format(
                    self.df, 
                    self.long_id_col.get(), 
                    self.long_task_col.get(),
                    self.long_item_col.get(), 
                    self.long_selection_col.get()
                )
            else:
                messagebox.showwarning("Warning", "Please use the Column Mapper")
                return
            self.show_converted_preview(converted_df)
        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed:\n{str(e)}")
    
    def show_converted_preview(self, converted_df):
        preview_win = ctk.CTkToplevel(self)
        preview_win.title("Converted Data Preview")
        preview_win.geometry("900x500")
        
        ctk.CTkLabel(
            preview_win, text="✓ Converted Data",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        ctk.CTkLabel(
            preview_win, 
            text=f"Shape: {converted_df.shape[0]} rows × {converted_df.shape[1]} columns"
        ).pack()
        
        tree_frame = ctk.CTkFrame(preview_win)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
        x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")
        tree = ttk.Treeview(tree_frame, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        y_scroll.config(command=tree.yview)
        x_scroll.config(command=tree.xview)
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        
        tree["columns"] = list(converted_df.columns)
        tree["show"] = "headings"
        for col in converted_df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        for _, row in converted_df.head(20).iterrows():
            tree.insert("", "end", values=list(row))
        
        btn_frame = ctk.CTkFrame(preview_win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=10)
        
        def use_converted():
            self.on_convert(converted_df)
            preview_win.destroy()
            self.destroy()
        
        def save_converted():
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")], 
                title="Save Converted Data"
            )
            if filepath:
                converted_df.to_csv(filepath, index=False)
                messagebox.showinfo("Saved", f"Saved to:\n{filepath}")
        
        ctk.CTkButton(btn_frame, text="💾 Save CSV", width=120, command=save_converted).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="✓ Use for Analysis", width=150, command=use_converted).pack(side="right", padx=5)
        ctk.CTkButton(
            btn_frame, text="Cancel", width=100, fg_color="gray",
            command=preview_win.destroy
        ).pack(side="right", padx=5)
    
    def open_column_mapper(self):
        ColumnMapperWindow(self, self.df, self.on_convert)


class ColumnMapperWindow(ctk.CTkToplevel):
    def __init__(self, master, df, on_convert_callback):
        super().__init__(master)
        self.title("Manual Column Mapper")
        self.geometry("700x600")
        self.df = df
        self.on_convert = on_convert_callback
        self.create_widgets()
        self.transient(master)
        self.grab_set()
    
    def create_widgets(self):
        ctk.CTkLabel(
            self, text="🔧 Manual Column Mapping",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=20)
        
        map_frame = ctk.CTkScrollableFrame(self, height=400)
        map_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        cols = ["(None)"] + list(self.df.columns)
        self.mappings = {}
        
        ctk.CTkLabel(map_frame, text="Required:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 5))
        
        for label, key in [("Response ID:", "response_id"), ("Most/Best:", "most"), ("Least/Worst:", "least")]:
            row = ctk.CTkFrame(map_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=label, width=150, anchor="w").pack(side="left")
            combo = ctk.CTkComboBox(row, values=cols, width=250)
            combo.pack(side="left")
            self.mappings[key] = combo
        
        ctk.CTkLabel(map_frame, text="\nAttributes:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 5))
        
        for i in range(1, 8):
            row = ctk.CTkFrame(map_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=f"Attribute{i}:", width=150, anchor="w").pack(side="left")
            combo = ctk.CTkComboBox(row, values=cols, width=250)
            combo.pack(side="left")
            self.mappings[f"attr{i}"] = combo
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=20)
        ctk.CTkButton(btn_frame, text="Cancel", width=100, fg_color="gray", command=self.destroy).pack(side="right", padx=5)
        ctk.CTkButton(btn_frame, text="✓ Apply", width=150, command=self.apply_mapping).pack(side="right", padx=5)
    
    def apply_mapping(self):
        try:
            id_col = self.mappings['response_id'].get()
            most_col = self.mappings['most'].get()
            least_col = self.mappings['least'].get()
            
            if "(None)" in [id_col, most_col, least_col]:
                messagebox.showwarning("Warning", "Map all required columns")
                return
            
            attr_cols = [
                self.mappings[f'attr{i}'].get() 
                for i in range(1, 8) 
                if self.mappings[f'attr{i}'].get() != "(None)"
            ]
            
            if len(attr_cols) < 3:
                messagebox.showwarning("Warning", "Need at least 3 attribute columns")
                return
            
            new_df = pd.DataFrame({'Response ID': self.df[id_col]})
            for i, col in enumerate(attr_cols, 1):
                new_df[f'Attribute{i}'] = self.df[col]
            new_df['Most'] = self.df[most_col]
            new_df['Least'] = self.df[least_col]
            
            self.on_convert(new_df)
            self.master.destroy()
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class MaxDiffGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MaxDiff Analysis Tool")
        self.geometry("750x900")
        self.minsize(700, 750)
        
        self.maxdiff_file = None
        self.maxdiff_df = None
        self.segment_file = None
        self.analysis_thread = None
        self.message_queue = queue.Queue()
        
        self.create_widgets()
        self.check_queue()
    
    def create_widgets(self):
        self.main_frame = ctk.CTkScrollableFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(
            header, text="📊 MaxDiff Analysis Tool",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack()
        ctk.CTkLabel(
            header, text="By Carl J",
            font=ctk.CTkFont(size=14), text_color="gray"
        ).pack(pady=(5, 0))
        
        # Help section
        help_frame = CollapsibleFrame(self.main_frame, "📖 Data Format Guide")
        help_frame.pack(fill="x", pady=(0, 15))
        help_content = help_frame.get_content_frame()
        
        instructions = """REQUIRED FORMAT: Response ID | Attribute1 | Attribute2 | Attribute3 | ... | Most | Least

- Response ID: Participant identifier (can repeat for multiple tasks)
- Attribute columns: Items shown in each choice task (minimum 3)
- Most/Best: The item selected as most preferred
- Least/Worst: The item selected as least preferred

Use "Browse" to auto-detect format and convert if needed."""
        
        ctk.CTkLabel(
            help_content, text=instructions, font=ctk.CTkFont(size=12),
            justify="left", wraplength=650
        ).pack(pady=5, anchor="w")
        
        btn_row = ctk.CTkFrame(help_content, fg_color="transparent")
        btn_row.pack(fill="x", pady=5)
        ctk.CTkButton(
            btn_row, text="📥 Example MaxDiff CSV", width=180,
            command=self.generate_example_maxdiff
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            btn_row, text="📥 Example Segment CSV", width=180,
            command=self.generate_example_segment
        ).pack(side="left", padx=5)
        
        # Data files section
        files_frame = ctk.CTkFrame(self.main_frame)
        files_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            files_frame, text="📁 Data Files",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        maxdiff_row = ctk.CTkFrame(files_frame, fg_color="transparent")
        maxdiff_row.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(maxdiff_row, text="MaxDiff Data:", width=100, anchor="w").pack(side="left")
        self.maxdiff_entry = ctk.CTkEntry(maxdiff_row, width=300, state="disabled")
        self.maxdiff_entry.pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            maxdiff_row, text="Browse", width=100, 
            command=self.load_and_preview_maxdiff,
            fg_color="#2d8a4e", hover_color="#236b3c"
        ).pack(side="left")
        self.maxdiff_status = ctk.CTkLabel(maxdiff_row, text="⚠️ Required", text_color="orange", width=80)
        self.maxdiff_status.pack(side="left", padx=10)
        
        self.data_info_label = ctk.CTkLabel(files_frame, text="", font=ctk.CTkFont(size=12), text_color="gray")
        self.data_info_label.pack(anchor="w", padx=15, pady=(0, 5))
        
        segment_row = ctk.CTkFrame(files_frame, fg_color="transparent")
        segment_row.pack(fill="x", padx=15, pady=(5, 15))
        ctk.CTkLabel(segment_row, text="Segment Data:", width=100, anchor="w").pack(side="left")
        self.segment_entry = ctk.CTkEntry(segment_row, width=300, state="disabled", placeholder_text="Optional")
        self.segment_entry.pack(side="left", padx=(0, 10))
        ctk.CTkButton(segment_row, text="Browse", width=100, command=self.browse_segment).pack(side="left")
        self.segment_status = ctk.CTkLabel(segment_row, text="Optional", text_color="gray", width=80)
        self.segment_status.pack(side="left", padx=10)
        ctk.CTkButton(segment_row, text="✕", width=30, fg_color="gray", command=self.clear_segment).pack(side="left", padx=5)
        
        # Analysis Options (collapsible, default collapsed)
        options_collapsible = CollapsibleFrame(self.main_frame, "⚙️ Analysis Options", expanded=False)
        options_collapsible.pack(fill="x", pady=(0, 15))
        options_content = options_collapsible.get_content_frame()
        
        # Output Labels (inside Analysis Options)
        term_section = ctk.CTkFrame(options_content, fg_color="transparent")
        term_section.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(
            term_section, text="Output Labels:",
            font=ctk.CTkFont(size=14)
        ).pack(anchor="w", pady=(0, 5))
        self.output_term_var = ctk.StringVar(value="Best/Worst")
        ctk.CTkSegmentedButton(
            term_section, values=["Most/Least", "Best/Worst"],
            variable=self.output_term_var, width=250
        ).pack(anchor="w")
        
        # Analysis Method (inside Analysis Options)
        method_section = ctk.CTkFrame(options_content, fg_color="transparent")
        method_section.pack(fill="x", pady=(10, 10))
        ctk.CTkLabel(
            method_section, text="Analysis Method:",
            font=ctk.CTkFont(size=14)
        ).pack(anchor="w", pady=(0, 5))
        
        self.analysis_method_var = ctk.StringVar(value="count")
        
        self.count_radio = ctk.CTkRadioButton(
            method_section, 
            text="Count-Based Analysis (Fast, Recommended)",
            variable=self.analysis_method_var, 
            value="count",
            command=self.update_method_options
        )
        self.count_radio.pack(anchor="w", pady=2)
        
        hb_text = "Hierarchical Bayes (Slow, ~5-20 min)"
        if not HAS_NUMPYRO:
            hb_text += " [Unavailable]"
        
        self.hb_radio = ctk.CTkRadioButton(
            method_section, 
            text=hb_text,
            variable=self.analysis_method_var, 
            value="hb",
            command=self.update_method_options
        )
        self.hb_radio.pack(anchor="w", pady=2)
        
        if not HAS_NUMPYRO:
            self.hb_radio.configure(state="disabled")
        
        self.method_desc_label = ctk.CTkLabel(
            method_section, 
            text="Simple count-based scores with optional bootstrap confidence intervals.",
            font=ctk.CTkFont(size=11), text_color="gray", wraplength=650
        )
        self.method_desc_label.pack(anchor="w", pady=(5, 0))
        
        # Count-based options
        self.count_options_frame = ctk.CTkFrame(options_content, fg_color="transparent")
        self.count_options_frame.pack(fill="x", pady=5)
        
        ci_row = ctk.CTkFrame(self.count_options_frame, fg_color="transparent")
        ci_row.pack(fill="x", pady=2)
        self.include_ci_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            ci_row, text="Include 95% Confidence Intervals (Bootstrap)",
            variable=self.include_ci_var, command=self.toggle_ci_options,
            font=ctk.CTkFont(size=14)
        ).pack(side="left")
        
        iter_row = ctk.CTkFrame(self.count_options_frame, fg_color="transparent")
        iter_row.pack(fill="x", pady=2)
        ctk.CTkLabel(iter_row, text="Bootstrap Iterations:", width=150, anchor="w").pack(side="left")
        self.iterations_var = ctk.StringVar(value="10000")
        self.iterations_entry = ctk.CTkEntry(iter_row, width=120, textvariable=self.iterations_var)
        self.iterations_entry.pack(side="left")
        
        # HB options
        self.hb_options_frame = ctk.CTkFrame(options_content, fg_color="transparent")
        self.hb_options_frame.pack(fill="x", pady=5)
        
        hb_iter_row = ctk.CTkFrame(self.hb_options_frame, fg_color="transparent")
        hb_iter_row.pack(fill="x", pady=2)
        ctk.CTkLabel(hb_iter_row, text="MCMC Iterations:", width=150, anchor="w").pack(side="left")
        self.hb_iterations_var = ctk.StringVar(value="5000")
        self.hb_iterations_entry = ctk.CTkEntry(hb_iter_row, width=120, textvariable=self.hb_iterations_var)
        self.hb_iterations_entry.pack(side="left")
        ctk.CTkLabel(hb_iter_row, text="(per chain, after warmup)", font=ctk.CTkFont(size=11), 
                    text_color="gray").pack(side="left", padx=10)
        
        hb_chains_row = ctk.CTkFrame(self.hb_options_frame, fg_color="transparent")
        hb_chains_row.pack(fill="x", pady=2)
        ctk.CTkLabel(hb_chains_row, text="MCMC Chains:", width=150, anchor="w").pack(side="left")
        self.hb_chains_var = ctk.StringVar(value="4")
        self.hb_chains_entry = ctk.CTkEntry(hb_chains_row, width=120, textvariable=self.hb_chains_var)
        self.hb_chains_entry.pack(side="left")
        
        hb_save_row = ctk.CTkFrame(self.hb_options_frame, fg_color="transparent")
        hb_save_row.pack(fill="x", pady=2)
        self.save_individual_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            hb_save_row, text="Save Individual-Level Utilities",
            variable=self.save_individual_var, font=ctk.CTkFont(size=14)
        ).pack(side="left")
        
        # Anchor item (common)
        anchor_row = ctk.CTkFrame(options_content, fg_color="transparent")
        anchor_row.pack(fill="x", pady=(5, 10))
        ctk.CTkLabel(anchor_row, text="Anchor Item:", width=150, anchor="w").pack(side="left")
        self.anchor_entry = ctk.CTkEntry(anchor_row, width=250, placeholder_text="Optional: item to highlight")
        self.anchor_entry.pack(side="left")
        
        # Initialize method options visibility
        self.update_method_options()
        
        # Colors (inside Analysis Options, always customizable)
        colors_section = ctk.CTkFrame(options_content, fg_color="transparent")
        colors_section.pack(fill="x", pady=(10, 0))
        ctk.CTkLabel(
            colors_section, text="Chart Colors:",
            font=ctk.CTkFont(size=14)
        ).pack(anchor="w", pady=(0, 5))
        
        self.colors_container = ctk.CTkFrame(colors_section, fg_color="transparent")
        self.colors_container.pack(fill="x")
        
        self.color_buttons = []
        color_configs = [
            ("Positive Points:", "#f4b400"),
            ("Negative Points:", "#f4b400"),
            ("Error Bars:", "#a1a1a1"),
            ("Zero Line:", "#ff0000"),
            ("Anchor Point:", "#a1a1a1"),
            ("Anchor Error:", "#a1a1a1")
        ]
        for label, default in color_configs:
            cb = ColorButton(self.colors_container, label, default)
            cb.pack(fill="x", pady=2)
            self.color_buttons.append(cb)
        
        # Run section
        run_frame = ctk.CTkFrame(self.main_frame)
        run_frame.pack(fill="x", pady=(0, 15))
        
        progress_container = ctk.CTkFrame(run_frame, fg_color="transparent")
        progress_container.pack(fill="x", padx=15, pady=15)
        self.progress_label = ctk.CTkLabel(progress_container, text="Ready", font=ctk.CTkFont(size=13))
        self.progress_label.pack(anchor="w")
        self.progress_bar = ctk.CTkProgressBar(progress_container, height=15)
        self.progress_bar.pack(fill="x", pady=(5, 10))
        self.progress_bar.set(0)
        
        btn_container = ctk.CTkFrame(run_frame, fg_color="transparent")
        btn_container.pack(pady=(0, 15))
        self.run_btn = ctk.CTkButton(
            btn_container, text="🚀 Run Analysis",
            font=ctk.CTkFont(size=16, weight="bold"),
            width=200, height=45, command=self.run_analysis
        )
        self.run_btn.pack(side="left", padx=10)
        ctk.CTkButton(
            btn_container, text="📂 Open Results", 
            font=ctk.CTkFont(size=14),
            width=150, height=45, fg_color="gray", 
            command=self.open_results_folder
        ).pack(side="left", padx=10)
        
        # Log
        log_frame = ctk.CTkFrame(self.main_frame)
        log_frame.pack(fill="both", expand=True, pady=(0, 10))
        ctk.CTkLabel(
            log_frame, text="📋 Analysis Log",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        self.log_text = ctk.CTkTextbox(log_frame, height=200, font=ctk.CTkFont(family="Courier", size=11))
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
    
    def update_method_options(self):
        """Show/hide options based on selected analysis method."""
        method = self.analysis_method_var.get()
        
        if method == 'count':
            self.count_options_frame.pack(fill="x", padx=0, pady=5)
            self.hb_options_frame.pack_forget()
            self.method_desc_label.configure(
                text="Transparent count-based scores: (Best - Worst) / Displays. Fast and highly correlated with HB results."
            )
        else:  # hb
            self.count_options_frame.pack_forget()
            self.hb_options_frame.pack(fill="x", padx=0, pady=5)
            self.method_desc_label.configure(
                text="Hierarchical Bayes with MCMC sampling. Provides individual-level utilities for follow-up analyses. Takes ~5-20 minutes."
            )
    
    def generate_example_maxdiff(self):
        """Show example MaxDiff data in a popup window."""
        example_data = [
            ["Response ID", "Attribute1", "Attribute2", "Attribute3", "Attribute4", "Most", "Least"],
            ["R001", "Price", "Quality", "Brand", "Speed", "Quality", "Price"],
            ["R001", "Design", "Support", "Durability", "Features", "Features", "Support"],
            ["R001", "Price", "Brand", "Durability", "Speed", "Price", "Speed"],
            ["R002", "Quality", "Brand", "Support", "Features", "Quality", "Support"],
            ["R002", "Price", "Durability", "Speed", "Design", "Speed", "Price"],
            ["R002", "Design", "Quality", "Brand", "Features", "Quality", "Design"],
            ["R003", "Brand", "Speed", "Design", "Durability", "Design", "Brand"],
            ["R003", "Price", "Quality", "Support", "Features", "Quality", "Price"],
            ["R003", "Support", "Durability", "Speed", "Brand", "Support", "Brand"],
        ]
        
        self._show_example_window(
            "Example MaxDiff Data",
            example_data,
            "example_maxdiff.csv",
            "Each row = one task. Response ID repeats for multiple tasks per respondent.\n"
            "Attribute columns = items shown in that task. Most/Least = respondent's choices."
        )
    
    def generate_example_segment(self):
        """Show example segment data in a popup window."""
        example_data = [
            ["Response ID", "Gender", "Age Group", "Region"],
            ["R001", "Female", "25-34", "North"],
            ["R002", "Male", "35-44", "South"],
            ["R003", "Female", "18-24", "North"],
            ["R004", "Male", "45-54", "East"],
            ["R005", "Female", "25-34", "West"],
            ["R006", "Non-binary", "35-44", "South"],
        ]
        
        self._show_example_window(
            "Example Segment Data", 
            example_data,
            "example_segments.csv",
            "One row per respondent. Response ID must match your MaxDiff data.\n"
            "Add any segment columns you want to analyze (demographics, behaviors, etc.)."
        )
    
    def _show_example_window(self, title, data, default_filename, description):
        """Display example data in a popup with copy/save options."""
        window = ctk.CTkToplevel(self)
        window.title(title)
        window.geometry("700x450")
        window.transient(self)
        window.grab_set()
        
        # Description
        ctk.CTkLabel(
            window, text=description,
            font=ctk.CTkFont(size=12), text_color="gray",
            wraplength=650, justify="left"
        ).pack(padx=20, pady=(15, 10), anchor="w")
        
        # Table frame
        table_frame = ctk.CTkFrame(window)
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create treeview
        columns = data[0]
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        y_scroll.pack(side="right", fill="y")
        x_scroll.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, minwidth=60)
        
        # Add data rows
        for row in data[1:]:
            tree.insert("", "end", values=row)
        
        # Convert to CSV string
        csv_string = "\n".join([",".join(row) for row in data])
        
        # Buttons frame
        btn_frame = ctk.CTkFrame(window, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        def copy_to_clipboard():
            window.clipboard_clear()
            window.clipboard_append(csv_string)
            messagebox.showinfo("Copied", "Example data copied to clipboard!\n\nPaste into Excel or a text editor.")
        
        def save_to_file():
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfilename=default_filename
            )
            if filepath:
                with open(filepath, 'w', newline='') as f:
                    f.write(csv_string)
                messagebox.showinfo("Saved", f"Example saved to:\n{filepath}")
        
        def save_to_current_dir():
            filepath = Path(default_filename)
            with open(filepath, 'w', newline='') as f:
                f.write(csv_string)
            messagebox.showinfo("Saved", f"Example saved to:\n{filepath.absolute()}")
        
        ctk.CTkButton(
            btn_frame, text="📋 Copy to Clipboard", width=150,
            command=copy_to_clipboard
        ).pack(side="left", padx=5)
        
#         ctk.CTkButton(
#             btn_frame, text="💾 Save As...", width=120,
#             command=save_to_file
#         ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame, text="💾 Save Here", width=120,
            command=save_to_current_dir, fg_color="#2d8a4e"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame, text="Close", width=100,
            command=window.destroy, fg_color="gray"
        ).pack(side="right", padx=5)
    
    def load_and_preview_maxdiff(self):
        filename = filedialog.askopenfilename(
            title="Select MaxDiff Data",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("All", "*.*")]
        )
        if not filename:
            return
        try:
            if filename.endswith('.xlsx'):
                df = pd.read_excel(filename)
            else:
                df = pd.read_csv(filename)
            self.maxdiff_file = filename
            detected_format, message = DataFormatDetector.detect_format(df)
            DataPreviewWindow(self, df, detected_format, message, self.set_maxdiff_data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{str(e)}")
    
    def set_maxdiff_data(self, df):
        self.maxdiff_df = df
        self.maxdiff_entry.configure(state="normal")
        self.maxdiff_entry.delete(0, "end")
        self.maxdiff_entry.insert(0, self.maxdiff_file or "Converted data")
        self.maxdiff_entry.configure(state="disabled")
        self.maxdiff_status.configure(text="✓ Ready", text_color="green")
        
        n_participants = df['Response ID'].nunique()
        attr_cols = [c for c in df.columns if c.startswith('Attribute')]
        items = [i for i in pd.unique(df[attr_cols].values.ravel()) if pd.notna(i)]
        
        self.data_info_label.configure(
            text=f"📊 {n_participants} participants, {len(df)} tasks, {len(items)} items, {len(attr_cols)} per task"
        )
        self.log(f"✓ Data loaded: {n_participants} participants, {len(items)} items")
    
    def browse_segment(self):
        filename = filedialog.askopenfilename(
            title="Select Segment Data",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")]
        )
        if filename:
            self.segment_file = filename
            self.segment_entry.configure(state="normal")
            self.segment_entry.delete(0, "end")
            self.segment_entry.insert(0, filename)
            self.segment_entry.configure(state="disabled")
            self.segment_status.configure(text="✓ Loaded", text_color="green")
            self.log(f"✓ Segment file: {filename}")
    
    def clear_segment(self):
        self.segment_file = None
        self.segment_entry.configure(state="normal")
        self.segment_entry.delete(0, "end")
        self.segment_entry.configure(state="disabled")
        self.segment_status.configure(text="Optional", text_color="gray")
    
    def toggle_ci_options(self):
        state = "normal" if self.include_ci_var.get() else "disabled"
        self.iterations_entry.configure(state=state)
    
    def log(self, msg):
        self.message_queue.put(("log", msg))
    
    def update_progress(self, val, msg=None):
        self.message_queue.put(("progress", (val, msg)))
    
    def check_queue(self):
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                if msg_type == "log":
                    self.log_text.insert("end", data + "\n")
                    self.log_text.see("end")
                elif msg_type == "progress":
                    self.progress_bar.set(data[0])
                    if data[1]:
                        self.progress_label.configure(text=data[1])
                elif msg_type == "complete":
                    self.run_btn.configure(state="normal", text="🚀 Run Analysis")
                    self.progress_label.configure(text="✅ Complete!")
                    messagebox.showinfo("Complete", "Analysis finished!\nResults in 'results' folder.")
                elif msg_type == "error":
                    self.run_btn.configure(state="normal", text="🚀 Run Analysis")
                    self.progress_label.configure(text="❌ Error")
                    messagebox.showerror("Error", str(data))
        except queue.Empty:
            pass
        self.after(100, self.check_queue)
    
    def open_results_folder(self):
        import subprocess
        import platform
        p = Path("results")
        if p.exists():
            system = platform.system()
            if system == "Windows":
                subprocess.Popen(f'explorer "{p.absolute()}"')
            elif system == "Darwin":
                subprocess.Popen(["open", str(p.absolute())])
            else:
                subprocess.Popen(["xdg-open", str(p.absolute())])
        else:
            messagebox.showwarning("Warning", "Run analysis first!")
    
    def run_analysis(self):
        if self.maxdiff_df is None:
            messagebox.showwarning("Warning", "Load MaxDiff data first!")
            return
        
        analysis_method = self.analysis_method_var.get()
        
        # Validate settings based on method
        if analysis_method == 'count':
            include_ci = self.include_ci_var.get()
            if include_ci:
                try:
                    n_iterations = int(self.iterations_var.get())
                    if n_iterations < 100:
                        raise ValueError("Iterations must be at least 100")
                except ValueError as e:
                    messagebox.showwarning("Warning", f"Enter valid iterations (min 100): {e}")
                    return
            else:
                n_iterations = 0
            hb_settings = None
        else:  # HB
            include_ci = True  # HB always has credible intervals
            n_iterations = 0
            try:
                hb_iter = int(self.hb_iterations_var.get())
                hb_chains = int(self.hb_chains_var.get())
                if hb_iter < 500:
                    raise ValueError("HB iterations must be at least 500")
                if hb_chains < 1:
                    raise ValueError("Must have at least 1 chain")
            except ValueError as e:
                messagebox.showwarning("Warning", f"Invalid HB settings: {e}")
                return
            
            hb_settings = {
                'iterations': hb_iter,
                'warmup': hb_iter // 2,
                'chains': hb_chains,
                'save_individual': self.save_individual_var.get()
            }
        
        output_terms = ("Most", "Least") if self.output_term_var.get() == "Most/Least" else ("Best", "Worst")
        
        colors = {}
        keys = [
            'positive_color', 'negative_color', 'error_bar_color', 
            'zero_line_color', 'anchor_item_color', 'anchor_item_error_color'
        ]
        for i, key in enumerate(keys):
            colors[key] = self.color_buttons[i].get_color()
        
        anchor_item = self.anchor_entry.get().strip() or None
        
        self.run_btn.configure(state="disabled", text="Running...")
        self.progress_bar.set(0)
        self.log_text.delete("1.0", "end")
        
        self.analysis_thread = threading.Thread(
            target=self.run_analysis_thread,
            args=(n_iterations, colors, anchor_item, output_terms, include_ci, 
                  analysis_method, hb_settings),
            daemon=True
        )
        self.analysis_thread.start()
    
    def run_analysis_thread(self, n_iterations, colors, anchor_item, output_terms, 
                           include_ci, analysis_method, hb_settings):
        try:
            base_dir = Path('results')
            base_dir.mkdir(parents=True, exist_ok=True)
            
            maxdiff_df = self.maxdiff_df.copy()
            
            pos_col, neg_col = 'Most', 'Least'
            if 'Best' in maxdiff_df.columns:
                pos_col, neg_col = 'Best', 'Worst'
            
            attribute_columns = [c for c in maxdiff_df.columns if c.startswith('Attribute')]
            
            self.log("Validating data...")
            check_errors(maxdiff_df, attribute_columns, pos_col, neg_col)
            
            unique_attributes = pd.unique(maxdiff_df[attribute_columns].values.ravel('K'))
            unique_attributes = unique_attributes[~pd.isnull(unique_attributes)]
            attr_to_index = {attr: i for i, attr in enumerate(unique_attributes)}
            
            overall_sample_size = maxdiff_df['Response ID'].nunique()
            self.log(f"✓ {len(unique_attributes)} items, {overall_sample_size} participants")
            self.log(f"Analysis method: {analysis_method.upper()}")
            
            # === DISPLAY STATISTICS ===
            self.log("\n" + "=" * 50)
            self.log("CALCULATING DISPLAY STATISTICS...")
            self.log("=" * 50)
            
            display_stats_df, balance_metrics = calculate_display_statistics(
                maxdiff_df, attribute_columns, pos_col, neg_col
            )
            
            # Log the display report
            display_report = format_display_report(display_stats_df, balance_metrics, output_terms)
            for line in display_report.split('\n'):
                self.log(line)
            
            # Save display statistics
            save_dataframe(display_stats_df, f"overall_display_statistics_n{overall_sample_size}", base_dir)
            
            # Save display balance plot
            balance_fig = plot_display_balance(
                display_stats_df, 
                title=f"Display Balance Overview (n={overall_sample_size})",
                output_terms=output_terms
            )
            save_plot(balance_fig, f"overall_display_balance_n{overall_sample_size}", base_dir)
            
            self.log("\n" + "=" * 50)
            
            # === MAIN ANALYSIS ===
            if analysis_method == 'hb':
                # Hierarchical Bayes Analysis
                self.log(f"\nHierarchical Bayes Analysis")
                self.log(f"  Iterations: {hb_settings['iterations']} (+ {hb_settings['warmup']} warmup)")
                self.log(f"  Chains: {hb_settings['chains']}")
                
                hb_model = HierarchicalBayesMaxDiff(
                    n_iterations=hb_settings['iterations'],
                    n_warmup=hb_settings['warmup'],
                    n_chains=hb_settings['chains']
                )
                
                def hb_progress(v):
                    self.update_progress(v * 0.7, f"HB MCMC: {int(v*100)}%")
                
                overall_results = hb_model.fit(
                    maxdiff_df, attribute_columns, pos_col, neg_col,
                    progress_callback=hb_progress, log_callback=self.log
                )
                result_key = 'hb_utilities'
                
                # Save individual utilities if requested
                if hb_settings.get('save_individual', True):
                    self.log("Saving individual-level utilities...")
                    individual_utils = hb_model.get_individual_utilities()
                    save_dataframe(individual_utils, f"overall_individual_utilities_n{overall_sample_size}", 
                                  base_dir, include_index=True)
                    
                    # Also save preference shares
                    pref_shares = hb_model.get_preference_shares()
                    save_dataframe(pref_shares, f"overall_preference_shares_n{overall_sample_size}", base_dir)
                    
            elif include_ci:
                # Count-based with bootstrap CI
                def progress_cb(v):
                    self.update_progress(v * 0.7, f"Bootstrap: {int(v*100)}%")
                self.log(f"\nBootstrap analysis ({n_iterations:,} iterations)...")
                overall_results = bootstrap_analysis(
                    maxdiff_df, attribute_columns, unique_attributes,
                    attr_to_index, pos_col, neg_col, n_iterations, progress_cb
                )
                result_key = 'bootstrap_results'
            else:
                # Count-based without CI
                self.log("\nCalculating scores...")
                overall_results = calculate_scores_no_ci(maxdiff_df, attribute_columns, pos_col, neg_col)
                result_key = 'scores'
                self.update_progress(0.5, "Processing...")
            
            self.log("Calculating percentages...")
            overall_observed = calculate_observed_percentages(
                maxdiff_df, attribute_columns, pos_col, neg_col, output_terms
            )
            
            self.log("Calculating correlations...")
            overall_corr = calculate_correlation_matrix(maxdiff_df, attribute_columns, pos_col, neg_col)
            
            self.update_progress(0.75, "Saving results...")
            save_results(
                {result_key: overall_results, 'observed_percentages': overall_observed},
                base_dir, sample_size=overall_sample_size, **colors, anchor_item=anchor_item,
                output_terms=output_terms, include_ci=include_ci
            )
            
            save_dataframe(overall_corr, f"overall_correlation_matrix_n{overall_sample_size}", base_dir, include_index=True)
            corr_fig = plot_correlation_matrix(overall_corr, f"Correlation Matrix (n={overall_sample_size})")
            save_plot(corr_fig, f"overall_correlation_matrix_n{overall_sample_size}_plot", base_dir)
            
            # Segment analysis
            if self.segment_file:
                self.log("\nProcessing segments...")
                try:
                    if self.segment_file.endswith('.xlsx'):
                        segment_df = pd.read_excel(self.segment_file)
                    else:
                        segment_df = pd.read_csv(self.segment_file)
                    
                    def seg_progress(v):
                        self.update_progress(0.75 + v * 0.2, f"Segments: {int(v*100)}%")
                    
                    segment_results = segment_maxdiff_analysis(
                        maxdiff_df, segment_df, pos_col, neg_col,
                        output_terms, n_iterations, anchor_item, include_ci, 
                        seg_progress, self.log, analysis_method, hb_settings
                    )
                    process_segment_results(
                        segment_results, base_dir, colors, output_terms, 
                        anchor_item, include_ci
                    )
                except Exception as seg_error:
                    self.log(f"⚠️ Segment analysis error: {seg_error}")
            
            self.update_progress(1.0, "Complete!")
            self.log("\n" + "=" * 50)
            self.log("✅ ANALYSIS COMPLETE!")
            self.log(f"   Method: {analysis_method.upper()}")
            self.log(f"   Results saved to: {base_dir.absolute()}")
            self.log("=" * 50)
            self.message_queue.put(("complete", None))
            
        except Exception as e:
            self.log(f"\n❌ ERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.message_queue.put(("error", str(e)))


if __name__ == "__main__":
    app = MaxDiffGUI()
    app.mainloop()
