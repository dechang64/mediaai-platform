"""
MediaOptimizer - Bayesian Optimization for Cell Culture Media

Gaussian Process surrogate model using Expected Improvement acquisition function
for efficient media formulation optimization.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class Observation:
    """Single observation of media composition and performance."""
    composition: Dict[str, float]
    performance: Dict[str, float]


@dataclass
class Candidate:
    """Recommended formulation candidate."""
    composition: Dict[str, float]
    predicted_performance: float
    uncertainty: float
    ei_value: float


@dataclass
class RecommendationResult:
    """Optimization result containing multiple candidates."""
    candidates: List[Candidate]
    best_composition: Dict[str, float]
    best_performance: float


class MediaOptimizer:
    """
    Bayesian Optimization Engine for Cell Culture Media.

    Uses Gaussian Process surrogate model with Expected Improvement
    acquisition function to efficiently explore media parameter space.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 0.01,
        ei_threshold: float = 0.01,
    ):
        """
        Initialize optimizer.

        Args:
            length_scale: RBF kernel length scale
            signal_variance: Signal variance (sigma_f^2)
            noise_variance: Noise variance (sigma_n^2)
            ei_threshold: Minimum improvement threshold
        """
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance

        self.observations: List[Observation] = []
        self.dimensions: List[str] = []

        # Kernel hyperparameters
        self.length_scale = length_scale

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Radial Basis Function (Matern) kernel.

        k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        if X1.shape[1] != X2.shape[1]:
            raise ValueError(f"Dimension mismatch: {X1.shape[1]} vs {X2.shape[1]}")

        pairwise_sq_dists = (
            np.sum(X1**2, axis=1, keepdims=True) +
            np.sum(X2**2, axis=1) -
            2 * np.dot(X1, X2.T)
        )

        kernel_matrix = self.signal_variance * np.exp(
            -pairwise_sq_dists / (2 * self.length_scale**2)
        )

        return kernel_matrix

    def _compute_posterior(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GP posterior predictive distribution.

        Returns:
            (mean, variance) of predictive distribution
        """
        # Kernel matrices
        K = self._rbf_kernel(X_train, X_train)
        K_noise = K + self.noise_variance * np.eye(len(X_train))

        K_star = self._rbf_kernel(X_train, X_test)
        K_double_star = self._rbf_kernel(X_test, X_test)

        # Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(K_noise)
        except np.linalg.LinAlgError:
            # Fallback: add small regularization
            K_noise += 1e-6 * np.eye(len(K_noise))
            L = np.linalg.cholesky(K_noise)

        # Solve L^T z = y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        # Predictive mean
        mean = K_star.T @ alpha

        # Predictive variance
        v = np.linalg.solve(L, K_star)
        var = np.diag(K_double_star) - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-6)  # Ensure positive

        return mean, var

    def _expected_improvement(
        self,
        mean: np.ndarray,
        var: np.ndarray,
        y_train: np.ndarray,
        target_metric: str,
        objective: str = "maximize",
    ) -> np.ndarray:
        """
        Compute Expected Improvement acquisition function.

        EI(x) = (mu - f_best) * Phi(Z) + sigma * phi(Z)

        where Z = (mu - f_best) / sigma if maximizing
        """
        if objective == "maximize":
            f_best = np.max(y_train)
            improvement = mean - f_best
        else:
            f_best = np.min(y_train)
            improvement = f_best - mean

        # Avoid division by zero
        std = np.sqrt(var)
        std = np.maximum(std, 1e-6)

        Z = improvement / std

        # Normal CDF and PDF
        Phi = 0.5 * (1 + np.erf(Z / np.sqrt(2)))
        phi = np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)

        # Expected Improvement
        ei = improvement * Phi + std * phi

        # Ensure non-negative
        ei = np.maximum(ei, 0)

        return ei

    def add_observation(
        self,
        composition: Dict[str, float],
        performance: Dict[str, float],
    ) -> bool:
        """
        Add new observation to training data.

        Args:
            composition: Media component concentrations
            performance: Observed performance metrics

        Returns:
            True if added successfully
        """
        obs = Observation(composition=composition, performance=performance)
        self.observations.append(obs)

        # Update dimensions if needed
        for key in composition.keys():
            if key not in self.dimensions:
                self.dimensions.append(key)

        return True

    def recommend_candidates(
        self,
        target_metric: str = "viability",
        objective: str = "maximize",
        n_candidates: int = 5,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> RecommendationResult:
        """
        Recommend next formulation candidates to test.

        Args:
            target_metric: Performance metric to optimize
            objective: "maximize" or "minimize"
            n_candidates: Number of candidates to recommend
            bounds: Optional parameter bounds

        Returns:
            RecommendationResult with ordered candidates
        """
        if len(self.observations) < 2:
            # Not enough data for GP
            raise ValueError(
                "Need at least 2 observations for Bayesian optimization"
            )

        # Extract training data
        dims = list(set().union(*[
            set(obs.composition.keys())
            for obs in self.observations
        ]))

        X_train = np.array([
            [obs.composition.get(d, 0.0) for d in dims]
            for obs in self.observations
        ])

        y_train = np.array([
            obs.performance.get(target_metric, 0.0)
            for obs in self.observations
        ])

        # Generate candidate points
        n_candidates_range = 1000

        if bounds is None:
            # Default bounds
            bounds = {d: (0, 10) for d in dims}

        # Random candidate generation
        candidate_compositions = []
        for dim in dims:
            low, high = bounds.get(dim, (0, 10))
            candidates = np.random.uniform(low, high, n_candidates_range)
            candidate_compositions.append(candidates)

        X_candidates = np.column_stack(candidate_compositions)

        # Compute GP posterior
        mean, var = self._compute_posterior(X_train, y_train, X_candidates)

        # Compute Expected Improvement
        ei = self._expected_improvement(mean, var, y_train, target_metric, objective)

        # Sort by EI and select top candidates
        top_indices = np.argsort(ei)[-n_candidates:]

        candidates_result = []
        for idx in reversed(top_indices):
            comp = {dims[i]: X_candidates[idx, i] for i in range(len(dims))}
            candidate = Candidate(
                composition=comp,
                predicted_performance=float(mean[idx]),
                uncertainty=float(np.sqrt(var[idx])),
                ei_value=float(ei[idx])
            )
            candidates_result.append(candidate)

        # Find best in training data
        if objective == "maximize":
            best_idx = np.argmax(y_train)
        else:
            best_idx = np.argmin(y_train)

        best_composition = self.observations[best_idx].composition
        best_performance = y_train[best_idx]

        return RecommendationResult(
            candidates=candidates_result,
            best_composition=best_composition,
            best_performance=float(best_performance),
        )


class GPParameters:
    """Hyperparameters for Gaussian Process"""

    def __init__(
        self,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 0.01,
        length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),
    ):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.length_scale_bounds = length_scale_bounds


# Export key classes
__all__ = [
    "MediaOptimizer",
    "GPParameters",
    "Observation",
    "Candidate",
    "RecommendationResult",
]