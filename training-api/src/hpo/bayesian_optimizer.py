"""
Bayesian Optimization for Hyperparameter Tuning.

Uses Gaussian Process surrogate model + Expected Improvement acquisition.
More efficient than random/grid search: finds better params in fewer trials.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    logger.warning(
        "scikit-optimize (skopt) not installed. "
        "BayesianHPOptimizer will fall back to random search. "
        "Install with: pip install scikit-optimize"
    )


# ---------------------------------------------------------------------------
# Default search space — matches HPOConfig in config.py
# ---------------------------------------------------------------------------

SEARCH_SPACE_DEFAULTS: Dict[str, tuple[Any, Any]] = {
    "lr0": (0.0001, 0.01),
    "lrf": (0.01, 1.0),
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0001, 0.001),
    "box": (5.0, 10.0),
    "cls": (0.3, 1.0),
}

# Ordered list of param names to ensure consistent dict ordering
_SEARCH_PARAM_NAMES = list(SEARCH_SPACE_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BayesianHPOptimizer:
    """
    Bayesian Optimization for YOLO hyperparameter search.

    Uses Gaussian Process surrogate model + Expected Improvement acquisition.
    More efficient than random/grid search: finds better params in fewer trials.

    Graceful degradation: falls back to random search when scikit-optimize
    is not installed.
    """

    def __init__(
        self,
        n_trials: int = 30,
        random_state: int = 42,
        param_space: Optional[Dict[str, tuple[Any, Any]]] = None,
    ):
        """
        Args:
            n_trials: Total number of HPO trials to run.
            random_state: Random seed for reproducibility.
            param_space: Override for the search space. Defaults to SEARCH_SPACE_DEFAULTS.
        """
        self.n_trials = n_trials
        self.random_state = random_state
        self.param_space = param_space or SEARCH_SPACE_DEFAULTS
        self.optimizer: Optional["Optimizer"] = None
        self.results: List[Dict[str, Any]] = []
        self._trial_count = 0
        self._best_score: Optional[float] = None
        self._best_params: Optional[Dict[str, float]] = None

        if HAS_SKOPT:
            self.optimizer = Optimizer(
                dimensions=self._build_skopt_space(),
                random_state=random_state,
                n_initial_points=5,  # random exploration before BO kicks in
                acq_func="EI",  # Expected Improvement
                acq_optimizer="sampling",
            )
            logger.info(
                "[BayesianHPOptimizer] Initialized with GP+EI (skopt available). "
                f"n_trials={n_trials}, random_state={random_state}"
            )
        else:
            logger.info(
                "[BayesianHPOptimizer] Running in random-search fallback mode. "
                f"n_trials={n_trials}"
            )

    # ------------------------------------------------------------------
    # skopt space builder
    # ------------------------------------------------------------------

    def _build_skopt_space(self) -> List:
        """Build skopt search space from param definitions."""
        if not HAS_SKOPT:
            return []
        space = []
        for name in _SEARCH_PARAM_NAMES:
            low, high = self.param_space[name]
            space.append(Real(low, high, name=name))
        return space

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self) -> Dict[str, float]:
        """
        Suggest the next set of hyperparameters to evaluate.

        Returns:
            Dict mapping param names to float values.
        """
        self._trial_count += 1

        if HAS_SKOPT and self.optimizer is not None:
            # Bayesian suggestion via skopt
            raw = self.optimizer.ask()
            params = dict(zip(_SEARCH_PARAM_NAMES, raw))
            logger.debug(
                f"[BayesianHPOptimizer] BO suggestion #{self._trial_count}: {params}"
            )
        else:
            # Random search fallback
            params = self._random_suggest()
            logger.debug(
                f"[BayesianHPOptimizer] Random suggestion #{self._trial_count}: {params}"
            )

        return params

    def report(self, params: Dict[str, float], score: float) -> None:
        """
        Report the evaluation result back to the optimizer.

        Args:
            params: The parameters that were evaluated.
            score: The mAP50 (or other metric) achieved. Higher is better.
        """
        self.results.append({"params": params, "score": score})

        # Update best-known result
        if self._best_score is None or score > self._best_score:
            self._best_score = score
            self._best_params = dict(params)
            logger.info(
                f"[BayesianHPOptimizer] New best: score={score:.4f} | params={params}"
            )

        if HAS_SKOPT and self.optimizer is not None:
            # skopt minimises, so negate the score
            self.optimizer.tell(list(params.values()), -score)

    def get_best(self) -> Dict[str, float]:
        """
        Return the best parameters found so far.

        Returns:
            Dict of best param values, or empty dict if no results yet.
        """
        if self._best_params is not None:
            return self._best_params
        if self.results:
            return max(self.results, key=lambda r: r["score"])["params"]
        return {}

    def get_search_space(self) -> List:
        """
        Return the raw skopt space (for diagnostics / introspection).
        Returns an empty list when skopt is unavailable.
        """
        if not HAS_SKOPT:
            return []
        return self._build_skopt_space()

    # ------------------------------------------------------------------
    # Fallback: random search
    # ------------------------------------------------------------------

    def _random_suggest(self) -> Dict[str, float]:
        """Generate a random parameter sample within bounds."""
        rs = random.Random(self.random_state + self._trial_count)
        return {
            name: rs.uniform(low, high)
            for name, (low, high) in self.param_space.items()
        }
