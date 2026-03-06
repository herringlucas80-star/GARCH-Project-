 
import numpy as np
import pandas as pd 
import yfinance as yf
from arch import arch_model 
ticker = "SPY" 
start = "2000-01-01"
end = None 
px = yf.download(ticker, start=start, end=end, auto_adjust=True)["Close"].dropna()
r = np.log(px).diff().dropna()
r_pct = 100 * r 
am = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="normal") 
res = am.fit(disp="off") 
print(res.summary()) 
params = res.params 
omega = params["omega"] 
alpha = params["alpha[1]"] 
beta = params["beta[1]"]
mu = params.get("mu", np.nan) 
print("\nKey params:") 
print(f"mu = {mu:.6f}") 
print(f"omega = {omega:.6f}") 
print(f"alpha = {alpha:.6f}") 
print(f"beta = {beta:.6f}") 
print(f"alpha + beta = {alpha + beta:.6f}")
stationary = (alpha + beta) < 1 
print(f"\nStationary (alpha+beta<1)? {stationary}") 
if stationary: 
    unc_var = omega / (1 - alpha - beta) 
    unc_vol = np.sqrt(unc_var) 
    print(f"Unconditional variance (%^2): {unc_var:.6f}") 
    print(f"Unconditional volatility (%): {unc_vol:.6f}") 
if (alpha + beta) > 0 and (alpha + beta) < 1: 
    half_life = np.log(0.5) / np.log(alpha + beta) 
    print(f"Half-life of volatility shocks (days): {half_life:.2f}") 
H = 21 
fcst = res.forecast(horizon=H, reindex=False) 
var_path = fcst.variance.values[-1, :] 
cum_var = var_path.sum() 
cum_vol = np.sqrt(cum_var) 
print(f"\nCumulative (integrated) variance forecast over {H} days (%^2): {cum_var:.6f}") 
print(f"Cumulative volatility over {H} days (%): {cum_vol:.6f}") 
daily_vol_last = np.sqrt(var_path[0]) 
ann_vol_last = daily_vol_last * np.sqrt(252) 
print(f"1-day ahead forecast volatility (%): {daily_vol_last:.6f}") 
print(f"1-day ahead forecast annualized vol (%): {ann_vol_last:.6f}") 









import os 
import matplotlib.pyplot as plt 
print("ABOUT TO PLOT...") 
print("Saving plots in:", os.getcwd()) 
cond_var = res.conditional_volatility 
cond_vol = cond_var 
unc_vol = (omega / (1 - alpha - beta)) ** 0.5 

plt.figure(figsize=(12,6)) 
plt.plot(r_pct.index, r_pct, color='gray', alpha=0.6) 
plt.title("Daily Log Returns (%)") 
plt.ylabel("Daily Log Return (%)") 
plt.tight_layout() 
plt.savefig("1_returns.png", dpi=200)
plt.close()

plt.figure(figsize=(12,6)) 
plt.plot(r_pct.index, r_pct**2, color='darkred', alpha=0.6) 
plt.title("Squared Returns (Volatility Clustering)") 
plt.ylabel("Squared Daily Returns (%)^2") 
plt.tight_layout()
plt.savefig("2_squared_returns.png", dpi=200)
plt.close()

plt.figure(figsize=(12,6)) 
plt.plot(r_pct.index, cond_vol, label="Conditional Volatility (%)") 
plt.axhline(unc_vol, color='red', linestyle='--', label="Unconditional Volatility (%)") 
plt.legend()
plt.title("Estimated Garch(1,1) Conditional Volatility") 
plt.ylabel("Conditional Daily Volatility (%)") 
plt.tight_layout()
plt.savefig("3_conditional_vol.png", dpi=200) 
plt.close()

print("\n--- HALF-LIFE BLOCK DEBUG ---")
print("cwd:", os.getcwd())
phi = alpha + beta
print("alpha:", alpha, "beta:", beta, "phi:", phi)
print("phi condition (0<phi<1):", (0 < phi < 1))
print("--- END DEBUG ---\n")

plt.figure(figsize=(12, 6))
h = np.arange(0, 200)

if 0 < phi < 1:
    half_life = np.log(0.5) / np.log(phi)
    decay = phi ** h
    plt.plot(h, decay, label=f"Decay: (alpha+beta)^h, phi={phi:.4f}")
    plt.axhline(0.5, linestyle="--", label="50% level")
    plt.axvline(half_life, linestyle="--", label=f"Half-life ≈ {half_life:.1f} days")
    plt.title("GARCH(1,1) Volatility Shock Decay (Half-Life Visualization)")
    print(f"\nHalf-life (days): {half_life:.2f}")
else:
    plt.plot(h, np.zeros_like(h), alpha=0)  # empty axes for annotation
    plt.text(0.5, 0.6, f"Half-life undefined: phi = alpha+beta = {phi:.6f}",
             transform=plt.gca().transAxes, fontsize=12, ha="center")
    plt.text(0.5, 0.45, "Stationarity requires 0 < phi < 1", transform=plt.gca().transAxes, ha="center")
    plt.title("Half-life: Not defined (non-stationary or unstable)")
    print(f"\nphi = alpha+beta = {phi:.6f}. Half-life not defined (non-stationary/unstable).")

plt.xlabel("Days After Shock")
plt.ylabel("Remaining Shock Impact (normalized)")
plt.legend(loc='upper right')

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
os.makedirs(script_dir, exist_ok=True)
fname = os.path.join(script_dir, "4_half_life_decay.png")

fig = plt.gcf()
fig.savefig(fname, dpi=200)
plt.close(fig)

if os.path.exists(fname):
    print("Saved half-life plot to:", fname)   
    try:
        os.startfile(fname) 
    except Exception: 
        pass 
else: 
    print("Failed to save half-life plot. Checked path:", fname) 