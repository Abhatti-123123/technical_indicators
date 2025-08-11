import numpy as np

class VolatilityScaledSignalSizer:
    """
    Volatility-based leverage scaler. Wraps a base position signal by
    dynamically adjusting exposure via Carver-style target vol:
        target_vol = fast_weight * sigma_fast + slow_weight * sigma_slow
        leverage = target_vol / max(sigma_fast, vol_floor)

    Vols are EWMA-updated per bar using recursive formula, annualized.
    Final position = base_pos * leverage, clipped to ±max_leverage.
    Optional smoothing on the leverage factor.
    """

    def __init__(
        self,
        fast_lambda: float = 0.60,
        slow_lambda: float = 0.97,
        fast_weight: float = 0.7,
        slow_weight: float = 0.3,
        max_leverage: float = 3.0,
        vol_floor: float = 1e-3,
        leverage_smoothing_alpha: float | None = None,  # e.g., 0.2 to smooth
    ):
        self.fast_lambda = fast_lambda
        self.slow_lambda = slow_lambda
        self.fast_weight = fast_weight
        self.slow_weight = slow_weight
        self.max_leverage = max_leverage
        self.vol_floor = vol_floor
        self.leverage_smoothing_alpha = leverage_smoothing_alpha

        # internal state
        self._var_fast = None  # unannualized EWMA variance
        self._var_slow = None
        self._prev_leverage = None

    def _update_ewma_var(self, ret: float, lambda_: float, prev_var: float | None) -> float:
        """Recursive EWMA variance update on squared returns (not annualized)."""
        rt2 = ret ** 2
        if prev_var is None:
            return rt2
        return lambda_ * prev_var + (1 - lambda_) * rt2

    def update_and_scale(self, base_pos: float, ret: float) -> float:
        """
        Call once per bar.
        :param base_pos: signal from score_to_position(proba) (can be in [-1,1] or similar)
        :param ret: latest asset return (e.g., close.pct_change().iloc[t])
        :return: scaled position (with vol-based leverage applied and clipped)
        """
        # update EWMA variances
        self._var_fast = self._update_ewma_var(ret, self.fast_lambda, self._var_fast)
        self._var_slow = self._update_ewma_var(ret, self.slow_lambda, self._var_slow)

        # annualize vol (sqrt of variance * sqrt(252))
        sigma_fast = np.sqrt(self._var_fast) * np.sqrt(252)
        sigma_slow = np.sqrt(self._var_slow) * np.sqrt(252)

        # target volatility (blended)
        target_vol = self.fast_weight * sigma_fast + self.slow_weight * sigma_slow

        # safe denominator
        safe_sigma_fast = max(sigma_fast, self.vol_floor)

        # raw leverage and caps
        raw_leverage = target_vol / safe_sigma_fast
        leverage = min(raw_leverage, self.max_leverage) 

        # optional smoothing
        if self.leverage_smoothing_alpha is not None:
            if self._prev_leverage is None:
                smoothed = leverage
            else:
                alpha = self.leverage_smoothing_alpha
                smoothed = alpha * leverage + (1 - alpha) * self._prev_leverage
            leverage = smoothed

        self._prev_leverage = leverage

        # apply to base position and enforce absolute cap
        scaled = base_pos * leverage
        return max(0, min(self.max_leverage, scaled))
