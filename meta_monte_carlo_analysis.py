import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('META_5Y_1MONTH_FROM_PERPLEXITY.csv')

df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
df = df.sort_values("date").reset_index(drop=True)

S0 = float(df["close"].iloc[-1])
last_date = df["date"].iloc[-1]

log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
sigma_monthly_hist = float(log_returns.std(ddof=1))
sigma_annual_hist = sigma_monthly_hist * np.sqrt(12)

print(f"last date: {last_date.date()}, цена S0 = {S0:.2f}")
print(f"σ (month., historical.) = {sigma_monthly_hist:.4f}")
print(f"σ (annual, historical.) = {sigma_annual_hist:.2%}")

TARGET_BASE = 640.0
TARGET_BULL = 680.0
TARGET_BEAR = 445.0

W_BASE = 0.60
W_BULL = 0.25
W_BEAR = 0.15

N_MONTHS = 12
dt = 1 / 12
N_SIMS = 20000

rng = np.random.default_rng(42)

mu_base_annual = np.log(TARGET_BASE / S0)
mu_bull_annual = np.log(TARGET_BULL / S0)
mu_bear_annual = np.log(TARGET_BEAR / S0)

sigma_base_annual = sigma_annual_hist
sigma_bull_annual = sigma_annual_hist * 0.95
sigma_bear_annual = sigma_annual_hist * 1.25

regimes = rng.choice(
    ["base", "bull", "bear"],
    size=N_SIMS,
    p=[W_BASE, W_BULL, W_BEAR],
)

mu_annual = np.select(
    [regimes == "base", regimes == "bull", regimes == "bear"],
    [mu_base_annual, mu_bull_annual, mu_bear_annual],
).astype(float)

sigma_annual = np.select(
    [regimes == "base", regimes == "bull", regimes == "bear"],
    [sigma_base_annual, sigma_bull_annual, sigma_bear_annual],
).astype(float)

Z = rng.standard_normal((N_SIMS, N_MONTHS))

jump_prob_annual = 0.18
jump_mean = -0.18
jump_std = 0.10

jump_flag = rng.random((N_SIMS, N_MONTHS)) < (jump_prob_annual / 12)
jumps = jump_flag * rng.normal(jump_mean, jump_std, size=(N_SIMS, N_MONTHS))

increments = (
    (mu_annual[:, None] - 0.5 * sigma_annual[:, None] ** 2) * dt
    + sigma_annual[:, None] * np.sqrt(dt) * Z
    + jumps
)

cum_log_returns = np.cumsum(increments, axis=1)

paths = np.zeros((N_SIMS, N_MONTHS + 1))
paths[:, 0] = S0
paths[:, 1:] = S0 * np.exp(cum_log_returns)

final = paths[:, -1]

q05, q25, q50, q75, q95 = np.percentile(final, [5, 25, 50, 75, 95])
mean_final = final.mean()

prob_up = (final > S0).mean()
prob_hit_580 = (final >= 580).mean()
prob_hit_640 = (final >= 640).mean()
prob_hit_680 = (final >= 680).mean()
prob_bear_510 = (final <= 510).mean()
prob_bear_445 = (final <= 445).mean()

print("\n--- Prediction 12 months ---")
print(f"Mean: {mean_final:.2f}")
print(f"Median: {q50:.2f}")
print(f"5%–95% quantiles:  [{q05:.2f}, {q95:.2f}]")
print(f"25%–75% quantiles: [{q25:.2f}, {q75:.2f}]")
print(f"P(increase):         {prob_up:.1%}")
print(f"P(price >= 580): {prob_hit_580:.1%}")
print(f"P(price >= 640): {prob_hit_640:.1%}")
print(f"P(price >= 680): {prob_hit_680:.1%}")
print(f"P(price <= 510): {prob_bear_510:.1%}")
print(f"P(price <= 445): {prob_bear_445:.1%}")

# freq="ME" вместо устаревшего "M"
future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthEnd(1),
    periods=N_MONTHS,
    freq="ME",
)
plot_dates = pd.DatetimeIndex([last_date]).append(future_dates)

quantile_paths = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

fig, ax = plt.subplots(figsize=(14, 8))

hist_tail = df.tail(24)
ax.plot(hist_tail["date"], hist_tail["close"],
        color="#1f2a44", lw=2.2, label="История (close)")
ax.scatter(hist_tail["date"], hist_tail["close"],
           color="#1f2a44", s=20, zorder=3)

sample_idx = rng.choice(N_SIMS, size=200, replace=False)
for i in sample_idx:
    ax.plot(plot_dates, paths[i], color="#4a90e2", lw=0.5, alpha=0.06)

ax.fill_between(plot_dates, quantile_paths[0], quantile_paths[4],
                color="#4a90e2", alpha=0.18, label="5%–95% интервал")
ax.fill_between(plot_dates, quantile_paths[1], quantile_paths[3],
                color="#4a90e2", alpha=0.30, label="25%–75% интервал")
ax.plot(plot_dates, quantile_paths[2],
        color="#c0392b", lw=2.5, label="Медианный прогноз")

# Thesis anchors
for level, color, name in [
    (580, "#7f8c8d", "Base floor 580"),
    (640, "#27ae60", "PT 640"),
    (680, "#8e44ad", "Bull midpoint 680"),
    (445, "#c0392b", "Bear midpoint 445"),
]:
    ax.axhline(level, color=color, ls="--", lw=1.2, alpha=0.8)
    ax.text(plot_dates[-1], level, f"  {name}",
            color=color, va="center", fontsize=9)

ax.axvline(last_date, color="gray", ls="--", lw=1, alpha=0.7)
ax.set_title(
    f"META — 12M Scenario Monte Carlo ({N_SIMS:,} симуляций)".replace(",", " "),
    fontsize=14, fontweight="bold",
)
ax.set_xlabel("Дата")
ax.set_ylabel("Цена, $")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", framealpha=0.95)

footer = (
    f"S₀ = ${S0:.2f} | "
    f"σ annual = {sigma_annual_hist:.1%} | "
    f"P(>=640) = {prob_hit_640:.0%} | "
    f"P(<=445) = {prob_bear_445:.0%}"
)
fig.text(0.5, 0.01, footer, ha="center", fontsize=10,
         color="#555", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1])

out = "META_scenario_MC_12M.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nГрафик сохранён: {out}")