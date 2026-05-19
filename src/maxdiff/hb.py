"""Hierarchical Bayes MaxDiff via NumPyro NUTS.

Optional dependency: requires ``jax`` and ``numpyro``. When the
imports fail, :data:`HAS_NUMPYRO` is False and :data:`NUMPYRO_ERROR`
holds the failure message; attempting to instantiate
:class:`HierarchicalBayesMaxDiff` then raises a clear ImportError.

Implementation notes:

* Sum-to-zero parameterization. The MaxDiff likelihood is invariant
  to constant shifts, so an identification constraint is required.
  v3.0.1 uses an orthonormal-basis construction (Stan's
  ``sum_to_zero_vector`` / PyMC's ``ZeroSumNormal``): sample ``n-1``
  free parameters and map to ``n`` item-space utilities via a fixed
  ``n x (n-1)`` matrix ``Q`` with orthonormal columns spanning the
  sum-to-zero subspace of ``R^n``. The result sums to exactly 0 and
  has symmetric marginal priors across all items. Earlier
  parameterizations were asymmetric and inflated the CI of one item
  (the derived one) by roughly ``sqrt(n-1)`` relative to the others;
  see ``_model``.

* Malformed-task handling: a task with fewer recognized items than
  expected raises ``ValueError`` (Phase 6). The legacy implementation
  silently padded with item index 0 here, which would have biased
  item 0's worst-rate upward if any malformed data slipped past the
  GUI's :func:`check_errors` gate.

* Parallel chains: ``numpyro.set_host_device_count(4)`` is set at
  import time but chains are still run sequentially with
  ``num_chains=1`` because the per-chain progress reporting in the
  current API needs the loop. Switching to true parallel chains is
  a future performance win.

* R-hat thresholds: tightened to follow Vehtari et al. 2021
  (<1.01 "Excellent", <1.05 "Good", <1.10 "borderline",
  otherwise "FAILED"). Phase 6 update.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Optional dependency probing
# ----------------------------------------------------------------------

HAS_NUMPYRO = False
NUMPYRO_ERROR: str | None = None

try:
    import jax

    jax.config.update("jax_platform_name", "cpu")  # Force CPU to avoid GPU issues
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax import random
    from numpyro.infer import MCMC, NUTS

    with contextlib.suppress(Exception):
        numpyro.set_host_device_count(4)

    HAS_NUMPYRO = True
except ImportError as e:
    NUMPYRO_ERROR = str(e)
except Exception as e:  # pragma: no cover - JAX init can fail in exotic ways
    NUMPYRO_ERROR = str(e)


def _sum_to_zero_basis(n_items: int):
    """Return an ``(n_items, n_items - 1)`` orthonormal basis of the
    sum-to-zero subspace of ``R^{n_items}``.

    Properties guaranteed by construction:

    * ``Q.T @ ones(n_items) == 0`` — each column sums to zero.
    * ``Q.T @ Q == I_{n_items - 1}`` — columns are orthonormal.
    * ``Q @ Q.T == I - (1/n) ones @ ones.T`` — the centering projector.

    Consequently, for ``y ~ Normal(0, sigma * I_{n_items - 1})``,
    ``u = Q @ y`` has ``sum(u) == 0`` exactly and marginal variance
    ``sigma^2 * (n_items - 1) / n_items`` uniform across all items.
    This is the "sum_to_zero_vector" / ``ZeroSumNormal`` parameterization
    used in Stan and PyMC.

    Implementation note: we obtain ``Q`` via SVD of the centering
    projector for numerical stability. The (n-1) columns of the left
    singular matrix corresponding to unit singular values span the
    sum-zero subspace.
    """
    if not HAS_NUMPYRO:  # pragma: no cover - guarded by class __init__
        raise ImportError("jax is required for _sum_to_zero_basis")
    n = n_items
    projector = jnp.eye(n) - jnp.ones((n, n)) / n
    u_svd, _s, _vt = jnp.linalg.svd(projector)
    return u_svd[:, : n - 1]


class HierarchicalBayesMaxDiff:
    """Hierarchical Bayes MaxDiff estimator (NumPyro NUTS).

    Implements the Displayr/Sawtooth-style "tricked logit"
    likelihood: best and worst within each task contribute two
    independent multinomial choices on the displayed items.
    Individual utilities are drawn from a multivariate-normal
    population with a non-centered parameterization.

    See module docstring for known limitations and Phase 4 work
    items.
    """

    def __init__(
        self,
        n_iterations: int = 5000,
        n_warmup: int = 2500,
        n_chains: int = 4,
        target_accept: float = 0.8,
    ) -> None:
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

    def _prepare_data(
        self,
        df: pd.DataFrame,
        attribute_columns: list[str],
        pos_col: str,
        neg_col: str,
    ) -> None:
        unique_items = pd.unique(df[attribute_columns].values.ravel())
        self.items = np.array([item for item in unique_items if pd.notna(item)])
        self.n_items = len(self.items)
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}

        self.respondents = df["Response ID"].unique()
        self.n_respondents = len(self.respondents)
        resp_to_idx = {r: i for i, r in enumerate(self.respondents)}

        tasks_per_resp = df.groupby("Response ID").size()
        self.n_tasks = int(tasks_per_resp.iloc[0])
        self.n_items_per_task = len(attribute_columns)

        items_in_tasks = np.zeros(
            (self.n_respondents, self.n_tasks, self.n_items_per_task), dtype=np.int32
        )
        best_local_idx = np.zeros((self.n_respondents, self.n_tasks), dtype=np.int32)
        worst_local_idx = np.zeros((self.n_respondents, self.n_tasks), dtype=np.int32)

        for resp_id in self.respondents:
            resp_idx = resp_to_idx[resp_id]
            resp_data = df[df["Response ID"] == resp_id].reset_index(drop=True)

            for task_idx in range(min(len(resp_data), self.n_tasks)):
                row = resp_data.iloc[task_idx]

                task_items: list[int] = []
                for col in attribute_columns:
                    if pd.notna(row[col]) and row[col] in self.item_to_idx:
                        task_items.append(self.item_to_idx[row[col]])

                # The legacy implementation silently padded short tasks
                # with item index 0, which biased item 0's worst rate
                # upward. With check_errors() called upstream this path
                # is unreachable; if a caller bypasses check_errors
                # and reaches this point with malformed data, raise
                # rather than corrupt the analysis.
                if len(task_items) < self.n_items_per_task:
                    raise ValueError(
                        f"Respondent {resp_id} task {task_idx + 1}: only "
                        f"{len(task_items)} attribute cells contained a known "
                        f"item (expected {self.n_items_per_task}). Run "
                        f"check_errors() to surface the malformed rows before "
                        f"fitting HB."
                    )

                items_in_tasks[resp_idx, task_idx, :] = task_items[: self.n_items_per_task]

                if row[pos_col] not in self.item_to_idx:
                    raise ValueError(
                        f"Respondent {resp_id} task {task_idx + 1}: "
                        f"'Most' value {row[pos_col]!r} is not a known item."
                    )
                if row[neg_col] not in self.item_to_idx:
                    raise ValueError(
                        f"Respondent {resp_id} task {task_idx + 1}: "
                        f"'Least' value {row[neg_col]!r} is not a known item."
                    )

                best_global = self.item_to_idx[row[pos_col]]
                worst_global = self.item_to_idx[row[neg_col]]

                try:
                    best_local_idx[resp_idx, task_idx] = task_items.index(best_global)
                except ValueError as e:
                    raise ValueError(
                        f"Respondent {resp_id} task {task_idx + 1}: "
                        f"'Most' item {row[pos_col]!r} not among the displayed "
                        f"attributes for this task."
                    ) from e

                try:
                    worst_local_idx[resp_idx, task_idx] = task_items.index(worst_global)
                except ValueError as e:
                    raise ValueError(
                        f"Respondent {resp_id} task {task_idx + 1}: "
                        f"'Least' item {row[neg_col]!r} not among the displayed "
                        f"attributes for this task."
                    ) from e

        self.items_in_tasks = jnp.array(items_in_tasks)
        self.best_local_idx = jnp.array(best_local_idx)
        self.worst_local_idx = jnp.array(worst_local_idx)

    def _model(
        self,
        items_in_tasks,
        best_local_idx,
        worst_local_idx,
        n_respondents,
        n_items,
        n_tasks,
        n_items_per_task,
    ) -> None:
        """Orthonormal-basis sum-to-zero parameterization (v3.0.1).

        The MaxDiff likelihood is invariant to a constant shift across
        items, so an identification constraint is required. Earlier
        revisions tried two asymmetric schemes:

        * "Last item fixed at 0" (pre-Phase-4): produced recovery bias
          on extreme items because the prior on the reference item
          differed from the others.
        * "Last item = -sum(first n-1)" (Phase 4 through v3.0.0): the
          population mean and the n-th respondent utility were
          deterministic functions of the n-1 free draws. The prior on
          the derived item was the sum of n-1 i.i.d. Normals, with
          variance ``(n-1) x sigma^2`` rather than ``sigma^2``. Point
          estimates still recovered correctly (the posterior mean of
          the derived item is the negative sum of the others, which is
          the correct identified value), but the credible interval on
          one item was visibly wider than the others — see
          ``tests/golden/test_hb_goldens.py::
          test_hb_ci_widths_are_symmetric_across_items``.

        v3.0.1 replaces both with a truly symmetric construction: an
        orthonormal basis ``Q`` of the sum-to-zero subspace of R^n.
        Sample ``n-1`` free parameters in that basis, then map to the
        ``n``-dimensional item space via ``Q``. Because ``Q`` has
        orthonormal columns and each column sums to zero, the resulting
        vector sums to exactly 0 by construction and the implied prior
        marginals are identical across all ``n`` items. This is the
        same "sum_to_zero_vector" construction used in Stan and PyMC's
        ``ZeroSumNormal``.
        """
        n_free = n_items - 1

        # Orthonormal basis of the sum-to-zero subspace of R^{n_items}.
        # Q is (n_items, n_free): each column sums to zero (Q.T @ 1 = 0),
        # columns are orthonormal (Q.T @ Q = I), and Q @ Q.T equals the
        # centering projector (I - 1/n * 11^T).
        Q = _sum_to_zero_basis(n_items)

        # Free-basis prior scales chosen so that the IMPLIED marginal
        # variance on each item-space utility matches the original
        # Normal(0, 2) / HalfNormal(1.5) priors. The Q transform dilates
        # variance by (n-1)/n in each row, so we scale up by sqrt(n/(n-1))
        # in the free basis. For n=20 this is a ~3% adjustment.
        scale_factor = jnp.sqrt(n_items / n_free)
        free_mu_scale = 2.0 * scale_factor
        free_sigma_scale = 1.5 * scale_factor

        # Population-level params in the free basis.
        mu_raw = numpyro.sample("mu_raw", dist.Normal(0.0, free_mu_scale).expand([n_free]))

        # Scalar respondent heterogeneity. Per-dimension sigma in the
        # free basis would NOT be symmetric in item space (because Q
        # mixes dimensions non-uniformly across items), so we use a
        # single scale here — a minor model simplification in exchange
        # for true item-space symmetry.
        sigma = numpyro.sample("sigma", dist.HalfNormal(free_sigma_scale))

        # Non-centered respondent deviations in the free basis.
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_respondents, n_free]))

        # Per-respondent params in the free basis, then mapped to item
        # space via Q. Each row of u sums to 0 by construction.
        u_resp_raw = mu_raw[None, :] + sigma * z  # (n_respondents, n_free)
        u = u_resp_raw @ Q.T  # (n_respondents, n_items)

        # Population mean in item space.
        mu_full = Q @ mu_raw  # (n_items,), sums to 0

        resp_idx = jnp.arange(n_respondents)[:, None, None]
        u_task = u[resp_idx, items_in_tasks]

        log_probs_best = jax.nn.log_softmax(u_task, axis=-1)
        log_probs_worst = jax.nn.log_softmax(-u_task, axis=-1)

        r_idx = jnp.arange(n_respondents)[:, None]
        t_idx = jnp.arange(n_tasks)[None, :]

        ll_best = log_probs_best[r_idx, t_idx, best_local_idx]
        ll_worst = log_probs_worst[r_idx, t_idx, worst_local_idx]

        total_ll = jnp.sum(ll_best) + jnp.sum(ll_worst)
        numpyro.factor("likelihood", total_ll)

        numpyro.deterministic("utilities", u)
        numpyro.deterministic("mu_full", mu_full)

    def fit(
        self,
        df: pd.DataFrame,
        attribute_columns: list[str],
        pos_col: str,
        neg_col: str,
        progress_callback: Callable[[float], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        if log_callback:
            log_callback("Preparing data for HB estimation...")

        self._prepare_data(df, attribute_columns, pos_col, neg_col)

        if log_callback:
            log_callback(f"  {self.n_respondents} respondents, {self.n_items} items")
            log_callback(
                f"  {self.n_tasks} tasks per respondent, {self.n_items_per_task} items per task"
            )
            log_callback(
                f"  MCMC: {self.n_iterations} iterations + {self.n_warmup} warmup per chain"
            )
            log_callback(f"  Running {self.n_chains} chain(s)...")

        if progress_callback:
            progress_callback(0.05)

        nuts_kernel = NUTS(self._model, target_accept_prob=self.target_accept, max_tree_depth=10)

        all_samples = []

        for chain_idx in range(self.n_chains):
            if log_callback:
                log_callback(f"\n  ═══ Chain {chain_idx + 1}/{self.n_chains} ═══")

            if progress_callback:
                chain_start_progress = 0.05 + (0.80 * chain_idx / self.n_chains)
                progress_callback(chain_start_progress)

            mcmc = MCMC(
                nuts_kernel,
                num_warmup=self.n_warmup,
                num_samples=self.n_iterations,
                num_chains=1,
                progress_bar=False,
            )

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
                    n_items_per_task=self.n_items_per_task,
                )

                chain_samples = mcmc.get_samples()
                all_samples.append(chain_samples)

                if log_callback:
                    log_callback(f"  Chain {chain_idx + 1} complete ✓")

            except Exception as e:
                if log_callback:
                    log_callback(f"  Chain {chain_idx + 1} failed: {e!s}")
                raise

        if progress_callback:
            progress_callback(0.85)

        if log_callback:
            log_callback(f"\nAll {self.n_chains} chains complete. Processing results...")

        self.samples = {}
        for key in all_samples[0]:
            self.samples[key] = np.concatenate([s[key] for s in all_samples], axis=0)

        if log_callback:
            total_samples = self.samples["mu_raw"].shape[0]
            log_callback(f"  Total posterior samples: {total_samples}")

        self._check_convergence_across_chains(all_samples, log_callback)

        self._compute_summaries()

        self.fitted = True

        if progress_callback:
            progress_callback(1.0)

        if log_callback:
            log_callback("HB estimation complete!")

        return self.get_population_results()

    def _check_convergence_across_chains(self, all_samples, log_callback=None) -> None:
        if len(all_samples) < 2:
            if log_callback:
                log_callback("  (Single chain - skipping R-hat diagnostic)")
            return

        try:
            mu_by_chain = np.stack([s["mu_raw"] for s in all_samples], axis=0)
            _n_chains, n_samples, _n_params = mu_by_chain.shape

            chain_means = mu_by_chain.mean(axis=1)
            B = n_samples * np.var(chain_means, axis=0, ddof=1)

            chain_vars = np.var(mu_by_chain, axis=1, ddof=1)
            W = np.mean(chain_vars, axis=0)

            var_pooled = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

            r_hat = np.sqrt(var_pooled / (W + 1e-10))
            max_rhat = float(np.max(r_hat))
            mean_rhat = float(np.mean(r_hat))

            # Thresholds follow Vehtari et al. 2021 ("Rank-normalized
            # split-Rhat..."): R-hat below 1.01 is excellent; below
            # 1.05 is acceptable; >= 1.05 is a real warning.
            if log_callback:
                if max_rhat < 1.01:
                    log_callback(
                        f"  Convergence: Excellent (R-hat: mean={mean_rhat:.3f}, max={max_rhat:.3f})"
                    )
                elif max_rhat < 1.05:
                    log_callback(
                        f"  Convergence: Good (R-hat: mean={mean_rhat:.3f}, max={max_rhat:.3f})"
                    )
                elif max_rhat < 1.1:
                    log_callback(
                        f"  ⚠️ Convergence borderline (R-hat: mean={mean_rhat:.3f}, max={max_rhat:.3f}) - "
                        "increase iterations"
                    )
                else:
                    log_callback(
                        f"  ⚠️ Convergence FAILED (R-hat: mean={mean_rhat:.3f}, max={max_rhat:.3f}) - "
                        "results may be unreliable; rerun with more iterations or chains"
                    )

        except Exception as e:
            if log_callback:
                log_callback(f"  Could not compute R-hat: {e}")

    def _compute_summaries(self) -> None:
        mu_samples_raw = np.array(self.samples["mu_full"])
        u_samples_raw = np.array(self.samples["utilities"])

        mu_samples = mu_samples_raw - mu_samples_raw.mean(axis=1, keepdims=True)
        u_samples = u_samples_raw - u_samples_raw.mean(axis=2, keepdims=True)

        self.population_mean = mu_samples.mean(axis=0)
        self.population_std = mu_samples.std(axis=0)
        self.mu_percentile_2_5 = np.percentile(mu_samples, 2.5, axis=0)
        self.mu_percentile_97_5 = np.percentile(mu_samples, 97.5, axis=0)

        self.individual_utilities = u_samples.mean(axis=0)

        self.mu_samples = mu_samples
        self.u_samples = u_samples

    def get_population_results(self, rescale: str = "zero_centered") -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        scores = self.population_mean.copy()
        lower = self.mu_percentile_2_5.copy()
        upper = self.mu_percentile_97_5.copy()

        if rescale == "zero_centered":
            shift = scores.mean()
            scores = scores - shift
            lower = lower - shift
            upper = upper - shift

        elif rescale == "probability":
            exp_mu = np.exp(self.mu_samples - self.mu_samples.max(axis=1, keepdims=True))
            share_samples = exp_mu / exp_mu.sum(axis=1, keepdims=True) * 100
            scores = share_samples.mean(axis=0)
            lower = np.percentile(share_samples, 2.5, axis=0)
            upper = np.percentile(share_samples, 97.5, axis=0)

        return pd.DataFrame(
            {
                "Item": self.items,
                "Score": scores,
                "2.5th Percentile": lower,
                "97.5th Percentile": upper,
                "Negative Error": scores - lower,
                "Positive Error": upper - scores,
            }
        ).sort_values("Score", ascending=False)

    def get_individual_utilities(self, rescale: str = "zero_centered") -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        utilities = self.individual_utilities.copy()

        if rescale == "zero_centered":
            utilities = utilities - utilities.mean(axis=1, keepdims=True)
        elif rescale == "probability":
            exp_u = np.exp(utilities - utilities.max(axis=1, keepdims=True))
            utilities = exp_u / exp_u.sum(axis=1, keepdims=True) * 100

        return pd.DataFrame(utilities, index=self.respondents, columns=self.items)

    def get_preference_shares(self) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        exp_mu = np.exp(self.mu_samples - self.mu_samples.max(axis=1, keepdims=True))
        share_samples = exp_mu / exp_mu.sum(axis=1, keepdims=True) * 100

        return pd.DataFrame(
            {
                "Item": self.items,
                "Share (%)": share_samples.mean(axis=0),
                "2.5th Percentile": np.percentile(share_samples, 2.5, axis=0),
                "97.5th Percentile": np.percentile(share_samples, 97.5, axis=0),
            }
        ).sort_values("Share (%)", ascending=False)
