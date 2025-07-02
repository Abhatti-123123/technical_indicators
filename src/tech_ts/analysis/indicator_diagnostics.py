import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tech_ts.analysis.analysis_utils import (
    plot_series, acf_pacf, stationarity, pairwise_coint, johansen_test
)
from tech_ts.data.fetch_data import download_prices
from tech_ts.indicators.indicators import add_indicator
from tech_ts.data.date_config import START_DATE, END_DATE

# -----------------------------------------------------------------
# 1.  Your own data-fetch / TA code here ➜ df_indicators
#     Assume df_indicators = DataFrame indexed by Date, cols = ['rsi', 'macd_diff', 'pctb']
# -----------------------------------------------------------------
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)
df = download_prices(['SPY'], START_DATE, END_DATE)
df = df.dropna()
close = df.iloc[:, 0]  # e.g., SPY by default
indicator_list = ['RSI', 'MACD', 'BollingerBands', 'Volatility']
df_indicators = close.copy()
close_df = close.to_frame(name='Close')
for ind in indicator_list:
    df_indicators = add_indicator(close_df, ind)
# Basic overlay
indicators = df_indicators.drop('Close', axis=1)
# ── 1. Raw overlay (all indicators, crazy scales and all) ───────
fig = plot_series(df_indicators.drop(columns="Close"),
                  title="Indicator levels – raw")
fig.tight_layout()
fig.savefig(OUTDIR / "overlay_raw.png", dpi=150)
plt.close(fig)

# ── 2. Z-scored overlay so scales are comparable ────────────────
z = df_indicators.drop(columns="Close").apply(
        lambda s: (s - s.mean()) / s.std()
    )
fig = plot_series(z, title="Indicator levels – z-score")
fig.tight_layout()
fig.savefig(OUTDIR / "overlay_zscore.png", dpi=150)
plt.close(fig)

# ── 3. One-axis per indicator (small multiples) ─────────────────
cols = z.columns
fig, axes = plt.subplots(len(cols), 1,
                         figsize=(12, 2.5 * len(cols)),
                         sharex=True)
for c, ax in zip(cols, axes):
    z[c].plot(ax=ax)
    ax.set_title(c)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTDIR / "overlay_small_multiples.png", dpi=150)
plt.close(fig)

# ── 4. ACF / PACF for each indicator ────────────────────────────
for c in cols:
    fig = acf_pacf(df_indicators[c], lags=60,
                   title=f"ACF / PACF – {c}")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"acf_pacf_{c.lower()}.png", dpi=150)
    plt.close(fig)

# ── 5. Rolling mean & std (quick stationarity sanity-check) ─────
WINDOW = 252   # 1 trading year
for c in cols:
    s = df_indicators[c]
    fig, ax = plt.subplots(figsize=(12, 4))
    s.plot(ax=ax, label='level')
    s.rolling(WINDOW).mean().plot(ax=ax, label=f'{WINDOW}-day mean')
    s.rolling(WINDOW).std().plot(ax=ax, label=f'{WINDOW}-day std')
    ax.set_title(f"{c} – rolling mean & std")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"rolling_stats_{c.lower()}.png", dpi=150)
    plt.close(fig)

# ── 6. Correlation heat-map between indicators ──────────────────
import numpy as np
corr = df_indicators.drop(columns="Close").corr()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
ax.set_title("Correlation matrix")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(OUTDIR / "corr_heatmap.png", dpi=150)
plt.close(fig)