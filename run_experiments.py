import importlib.util, subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
for _mod, _pip in [("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),("sklearn","scikit-learn"),("matplotlib","matplotlib"),("reportlab","reportlab"),("PIL","pillow")]:
    if importlib.util.find_spec(_mod) is None:
        print("Installing", _pip, "..."); subprocess.check_call([sys.executable, "-m", "pip", "install", _pip])
def _find(*names):
    for _n in names:
        for _d in (".", "data"):
            _p = os.path.join(_d, _n)
            if os.path.exists(_p): return _p
    raise FileNotFoundError("Put noise_data_csv_1.csv in the same folder as this script.")
import csv, io, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)
BL, OR, GR, RD = "#1f4e8c", "#e07b00", "#6b6b6b", "#b3201a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
CSVFILE = _find("noise_data_csv_1.csv", "noise_data_csv.csv")
LIMIT = {"Day": 65.0, "Night": 55.0}
MON = {m: i for i, m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}
rows = list(csv.reader(io.StringIO(open(CSVFILE, encoding="latin1").read())))
recs = []
for r in rows[2:]:
    if len(r) < 12 or "-" not in r[0]: continue
    a, b = r[0].strip().split("-")
    if a not in MON or not r[2].strip(): continue
    d = pd.Timestamp(2000 + int(b), MON[a], 1)
    recs.append(dict(month=d, period="Day", leq=float(r[2]), lmin=float(r[3]), lmax=float(r[4]), pct=r[5].strip()))
    recs.append(dict(month=d, period="Night", leq=float(r[7]), lmin=float(r[8]), lmax=float(r[9]), pct=r[10].strip()))
df = pd.DataFrame(recs)
df["limit"] = df.period.map(LIMIT); df["exceed"] = (df.leq > df.limit).astype(int); df["margin"] = df.leq - df.limit
df["pct_flag"] = (df.pct.str.lower() != "within limit").astype(int)
df["flag_lmin"] = df.lmin.round(1) == 21.6; df["flag_lmax"] = df.lmax <= df.leq + 1e-9
df["flag"] = df.flag_lmin | df.flag_lmax
df["midx"] = (df.month.dt.year - 2019) * 12 + df.month.dt.month - 4; df["mon"] = df.month.dt.month
df["night"] = (df.period == "Night").astype(int)
df["msin"] = np.sin(2*np.pi*df.mon/12); df["mcos"] = np.cos(2*np.pi*df.mon/12)
df["festival"] = df.mon.isin([10, 11]).astype(int)
df = df.sort_values(["period", "midx"]).reset_index(drop=True)
df["lag"] = df.groupby("period").leq.shift(1); df["lag_exc"] = df.groupby("period").exceed.shift(1); df["lag_margin"] = df.lag - df.limit
df.to_csv("cleaned_noise_dataset.csv", index=False)
N = df.midx.nunique()
R = {"n_months": int(N), "n_rows": len(df), "label_agree": int((df.pct_flag == df.exceed).sum())}
d = df.pivot(index="midx", columns="period", values="leq")
R.update(day_mean=d.Day.mean(), night_mean=d.Night.mean(), diff_mean=(d.Day-d.Night).mean(), wilcoxon_p=float(stats.wilcoxon(d.Day, d.Night).pvalue),
         exc_day=int(df[df.period=="Day"].exceed.sum()), exc_night=int(df[df.period=="Night"].exceed.sum()), exc_total=int(df.exceed.sum()),
         n_flag_lmin=int(df.flag_lmin.sum()), n_flag_lmax=int(df.flag_lmax.sum()))
R["mad_day"] = float(d.Day.diff().abs().mean()); R["mad_night"] = float(d.Night.diff().abs().mean())
fest = df[df.festival==1]; non = df[df.festival==0]
R.update(fest_margin=fest.margin.mean(), non_margin=non.margin.mean(), fest_rate=fest.exceed.mean(), non_rate=non.exceed.mean())
R["spearman_lag"] = float(stats.spearmanr(df.dropna().leq, df.dropna().lag)[0])
AF = ["night", "msin", "mcos", "festival"]; BF = AF + ["lag"]; CBF = AF + ["lag_margin"]
def regs(k=3, depth=3, alpha=1.0):
    return {"OLS": lambda: Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
            "Ridge": lambda: Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=alpha))]),
            "Decision tree": lambda: DecisionTreeRegressor(max_depth=depth, min_samples_leaf=2, random_state=42),
            "k-NN": lambda: Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=k, weights="distance"))])}
def clfs(k=3, depth=3, C=1.0):
    return {"Logistic reg.": lambda: Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=C, max_iter=1000, random_state=42))]),
            "Decision tree": lambda: DecisionTreeClassifier(max_depth=depth, min_samples_leaf=2, random_state=42),
            "k-NN": lambda: Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=k))]),
            "Gauss. NB": lambda: Pipeline([("s", StandardScaler()), ("m", GaussianNB())])}
def rm(y, p): return dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=r2_score(y, p))
def cm(y, yh, pr=None):
    r = dict(Acc=accuracy_score(y, yh), Prec=precision_score(y, yh, zero_division=0), Rec=recall_score(y, yh, zero_division=0), F1=f1_score(y, yh, zero_division=0))
    r["AUC"] = float(roc_auc_score(y, pr)) if (pr is not None and len(set(y)) == 2) else None
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel(); r.update(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)); return r
# ---- Protocol A: leave-one-month-out
LA = df.copy()
for n in ["Period mean"] + list(regs()): LA[n] = np.nan
for n in ["Period majority"] + list(clfs()) + ["OLS>limit"]: LA["c_"+n] = np.nan
for n in clfs(): LA["p_"+n] = np.nan
def loo(**kw):
    out = LA.copy()
    for m in range(N):
        tr, te = df[df.midx != m], df[df.midx == m]
        out.loc[te.index, "Period mean"] = te.period.map(tr.groupby("period").leq.mean()).values
        for n, mk in regs(**kw.get("reg", {})).items(): out.loc[te.index, n] = mk().fit(tr[AF], tr.leq).predict(te[AF])
        out.loc[te.index, "c_Period majority"] = (te.period.map(tr.groupby("period").exceed.mean()) >= .5).astype(int).values
        for n, mk in clfs(**kw.get("clf", {})).items():
            mm = mk().fit(tr[AF], tr.exceed); out.loc[te.index, "c_"+n] = mm.predict(te[AF]); out.loc[te.index, "p_"+n] = mm.predict_proba(te[AF])[:, 1]
        out.loc[te.index, "c_OLS>limit"] = (regs()["OLS"]().fit(tr[AF], tr.leq).predict(te[AF]) > te.limit.values).astype(int)
    return out
LA = loo()
RA = ["Period mean", "OLS", "Ridge", "Decision tree", "k-NN"]
R["A_reg"] = {n: rm(LA.leq, LA[n]) for n in RA}
CA = ["Period majority", "OLS>limit", "Logistic reg.", "Decision tree", "k-NN", "Gauss. NB"]
R["A_clf"] = {n: cm(LA.exceed, LA["c_"+n].astype(int), LA.get("p_"+n) if n in clfs() else None) for n in CA}
bestA = min(RA[1:], key=lambda n: R["A_reg"][n]["RMSE"]); R["bestA"] = bestA
bs = {k: [] for k in ["Period mean", bestA]}
for _ in range(5000):
    q = pd.concat([LA[LA.midx == m] for m in RNG.choice(N, N, replace=True)])
    for k in bs: bs[k].append(np.sqrt(np.mean((q.leq - q[k])**2)))
R["A_boot"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
X = np.column_stack([np.ones(len(df))] + [df[c].values for c in AF]); y = df.leq.values
beta = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ beta; dof = len(y) - X.shape[1]; s2 = res @ res / dof
se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))); tc = stats.t.ppf(.975, dof)
R["ols"] = {n: dict(b=float(b), lo=float(b-tc*e), hi=float(b+tc*e), p=float(2*(1-stats.t.cdf(abs(b/e), dof)))) for n, b, e in zip(["const"]+AF, beta, se)}
R["ols_r2"] = float(1 - res@res/((y-y.mean())**2).sum()); R["ols_dof"] = int(dof)
lg = LogisticRegression(C=1.0, max_iter=1000).fit(df[AF], df.exceed); R["or"] = dict(zip(AF, map(float, np.exp(lg.coef_[0]))))
R["exc_by_month"] = df.groupby("mon").exceed.sum().to_dict()
# ---- Protocol B: rolling origin (test Oct-19..Mar-20)
use = df.dropna(subset=["lag"]); TM = list(range(6, N))
def roll(drop=None, kw={}):
    P, C = [], []
    for t in TM:
        tr = use[use.midx < t]; te = use[use.midx == t]
        if drop is not None: tr = tr[~((tr.period == drop[0]) & (tr.midx == drop[1]))]
        o = te[["midx","month","period","leq","limit","exceed","lag","lag_exc"]].copy()
        o["Persistence"] = te.lag.values; o["Period mean"] = te.period.map(tr.groupby("period").leq.mean()).values
        for n, mk in regs(**kw).items(): o[n] = mk().fit(tr[BF], tr.leq).predict(te[BF])
        P.append(o)
        c = te[["midx","month","period","leq","limit","exceed"]].copy()
        c["Persistence"] = te.lag_exc.values.astype(int); c["Period majority"] = (te.period.map(tr.groupby("period").exceed.mean()) >= .5).astype(int).values
        for n, mk in clfs().items():
            if tr.exceed.nunique() < 2: c[n] = int(tr.exceed.iloc[0]); c["p_"+n] = float(tr.exceed.iloc[0]); continue
            mm = mk().fit(tr[CBF], tr.exceed); c[n] = mm.predict(te[CBF]); c["p_"+n] = mm.predict_proba(te[CBF])[:, 1]
        c["OLS>limit"] = (regs()["OLS"]().fit(tr[BF], tr.leq).predict(te[BF]) > te.limit.values).astype(int)
        C.append(c)
    return pd.concat(P), pd.concat(C)
PB, CB = roll()
RB = ["Persistence", "Period mean", "OLS", "Ridge", "Decision tree", "k-NN"]
R["B_reg"] = {n: rm(PB.leq, PB[n]) for n in RB}; R["B_n"] = len(PB)
CBN = ["Persistence", "Period majority", "OLS>limit", "Logistic reg.", "Decision tree", "k-NN", "Gauss. NB"]
R["B_clf"] = {n: cm(CB.exceed, CB[n].astype(int), CB.get("p_"+n) if n in clfs() else None) for n in CBN}
R["B_exc_rate"] = float(CB.exceed.mean()); R["B_train_rate"] = float(use[use.midx < 6].exceed.mean())
bestB = min(RB[2:], key=lambda n: R["B_reg"][n]["RMSE"]); R["bestB"] = bestB
sens = {}
for k in (2, 3, 5): sens[f"k-NN k={k}"] = rm(*(lambda p: (p.leq, p["k-NN"]))(roll(kw=dict(k=k))[0]))["RMSE"]
for dp in (2, 3, 4): sens[f"Tree depth={dp}"] = rm(*(lambda p: (p.leq, p["Decision tree"]))(roll(kw=dict(depth=dp))[0]))["RMSE"]
for a in (0.1, 1, 10): sens[f"Ridge alpha={a}"] = rm(*(lambda p: (p.leq, p["Ridge"]))(roll(kw=dict(alpha=a))[0]))["RMSE"]
p2 = roll(drop=("Day", 3))[0]
for n in ["OLS", "Ridge", "Decision tree", "k-NN"]: sens[f"{n} (Jul-19 day row dropped)"] = rm(p2.leq, p2[n])["RMSE"]
R["sens"] = sens
json.dump(R, open("results.json", "w"), indent=1, default=str)
LA.to_csv("loo_predictions.csv", index=False); PB.to_csv("rolling_regression_predictions.csv", index=False); CB.to_csv("rolling_classification_predictions.csv", index=False)
def tab(dct, sub, proto, cls=False):
    o = []
    for m, v in dct.items():
        o.append(dict(Protocol=proto, Model=m, **({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})))
    return o
pd.DataFrame(tab(R["A_reg"], "", "A leave-one-month-out") + tab(R["B_reg"], "", "B rolling forecast")).to_csv("regression_results.csv", index=False)
pd.DataFrame(tab(R["A_clf"], "", "A leave-one-month-out") + tab(R["B_clf"], "", "B rolling forecast")).to_csv("classification_results.csv", index=False)
# ---- figures
lab = [m.strftime("%b\n%y") for m in sorted(df.month.unique())]; x = np.arange(N)
fig, ax = plt.subplots(figsize=(3.4, 2.9))
for m in (6, 7): ax.axvspan(m-.5, m+.5, color="#fff0cc", lw=0)
ax.plot(x, d.Day.values, "-o", ms=3.2, color=BL, label="Day Leq (06–22 h)"); ax.plot(x, d.Night.values, "-s", ms=3.2, color=OR, label="Night Leq (22–06 h)")
ax.axhline(65, color=BL, ls="--", lw=.9); ax.axhline(55, color=OR, ls="--", lw=.9)
ax.text(11.4, 65.7, "65 dB(A) day limit", ha="right", fontsize=5.5, color=BL); ax.text(11.4, 52.6, "55 dB(A) night limit", ha="right", fontsize=5.5, color=OR)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=5.5); ax.set_ylabel("Monthly Leq, dB(A)"); ax.set_ylim(48, 82); ax.set_xlim(-.5, 11.5)
h, l = ax.get_legend_handles_labels(); ax.legend(h+[Patch(color="#fff0cc")], l+["Festival season (Oct–Nov)"], loc="upper center", bbox_to_anchor=(.5, -.2), frameon=False, fontsize=5.8, ncol=2, columnspacing=.8, handlelength=1.4)
fig.tight_layout(); fig.savefig("fig1_timeseries.png", dpi=300); plt.close()
fig, ax = plt.subplots(figsize=(3.4, 1.5))
M = np.vstack([df[df.period == p].sort_values("midx").margin.values for p in ("Day", "Night")])
im = ax.imshow(M, cmap="RdBu_r", vmin=-20, vmax=20, aspect="auto"); ax.set_yticks([0, 1]); ax.set_yticklabels(["Day", "Night"], fontsize=6)
ax.set_xticks(range(12)); ax.set_xticklabels(["Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar"], fontsize=5.5)
for i in range(2):
    for j in range(12): ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center", fontsize=4.6, color="white" if abs(M[i,j]) > 12 else "black")
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.01); cb.set_label("Leq − limit (dB)", fontsize=5.5); cb.ax.tick_params(labelsize=5)
fig.tight_layout(); fig.savefig("fig2_margin_heatmap.png", dpi=300); plt.close()
fig, ax = plt.subplots(figsize=(3.3, 3.7)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 16)
steps = [("1  Raw data", "CSV: Apr-2019 to Mar-2020\n12 months × (day, night) = 24 rows"), ("2  Cleaning", "drop footnote rows, parse months\nflag Lmin/Lmax anomalies"),
         ("3  Features", "period, month sin/cos, festival,\nprevious-month value (protocol B)"), ("4  Two protocols", "A: leave-one-month-out (12 folds)\nB: rolling origin, Oct-19 … Mar-20"),
         ("5  Models [scaler inside]", "OLS, ridge, tree, k-NN | logistic,\ntree, k-NN, NB vs baselines"), ("6  Evaluation", "RMSE, MAE, R² | acc., P, R, F1, AUC\nmonth-level bootstrap interval")]
for i, (t, s) in enumerate(steps):
    y0 = 15 - i*2.6; ax.add_patch(plt.Rectangle((.2, y0-2.0), 9.6, 2.15, fc="#f2f5fa", ec=BL, lw=.8)); ax.text(.5, y0-.35, t, fontsize=7.2, fontweight="bold", va="top", color=BL); ax.text(.5, y0-.95, s, fontsize=6, va="top")
    if i < 5: ax.annotate("", xy=(5, y0-2.2), xytext=(5, y0-2.0), arrowprops=dict(arrowstyle="-|>", color=BL, lw=.8))
fig.tight_layout(pad=.2); fig.savefig("fig3_pipeline.png", dpi=300); plt.close()
fig, ax = plt.subplots(figsize=(3.3, 3.0))
for per, mk, col in (("Day", "o", BL), ("Night", "s", OR)):
    q = LA[LA.period == per]; ax.scatter(q.leq, q[bestA], marker=mk, s=16, color=col, label=per, alpha=.85)
lo, hi = df.leq.min()-3, df.leq.max()+3; ax.plot([lo, hi], [lo, hi], color=GR, lw=.8); ax.set_xlabel("Observed Leq, dB(A)"); ax.set_ylabel("Predicted (leave-one-month-out)")
ax.set_title(f"{bestA}: RMSE {R['A_reg'][bestA]['RMSE']:.2f}, R² {R['A_reg'][bestA]['R2']:.2f}", fontsize=7.5); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("fig4_lomo_scatter.png", dpi=300); plt.close()
fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.7))
for ax, per in zip(axs, ("Day", "Night")):
    q = PB[PB.period == per].sort_values("midx"); xx = np.arange(len(q))
    ax.plot(xx, q.leq, "-o", color="black", ms=3.2, lw=1.3, label="Observed"); ax.plot(xx, q["Persistence"], ":", color=GR, lw=1.2, label="Persistence")
    ax.plot(xx, q[bestB], "-s", color=OR if per == "Night" else BL, ms=2.8, lw=1, label=bestB); ax.axhline(LIMIT[per], color=RD, ls="--", lw=.8)
    ax.set_title(f"{per}-time (limit {LIMIT[per]:.0f} dB(A))", fontsize=8); ax.set_xticks(xx); ax.set_xticklabels([m.strftime("%b") for m in q.month], fontsize=6.5); ax.set_ylabel("Leq, dB(A)"); ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig("fig5_predictions.png", dpi=300); plt.close()

# ---------- export every paper table as CSV ----------
os.makedirs("tables", exist_ok=True); os.makedirs("figures", exist_ok=True)
import glob, shutil
dsc = []
for per in ("Day", "Night"):
    q = df[df.period == per].sort_values("midx")
    dsc.append(dict(Period=per, Mean=round(q.leq.mean(), 2), Min=q.leq.min(), Max=q.leq.max(), Months_over_limit=int(q.exceed.sum()), Mean_margin=round(q.margin.mean(), 2), Mean_abs_monthly_change=round(q.leq.diff().abs().mean(), 2)))
pd.DataFrame(dsc).to_csv("tables/Table3_descriptive.csv", index=False)
pd.DataFrame([dict(Model=m, **{k: round(v, 3) for k, v in x.items()}) for m, x in R["A_reg"].items()]).to_csv("tables/Table4_regression_leave_one_month_out.csv", index=False)
pd.DataFrame([dict(Term=k, Coef=round(v["b"], 3), CI_low=round(v["lo"], 3), CI_high=round(v["hi"], 3), p=round(v["p"], 4)) for k, v in R["ols"].items()]).to_csv("tables/Table5_OLS_coefficients.csv", index=False)
pd.DataFrame([dict(Model=m, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}) for m, x in R["A_clf"].items()]).to_csv("tables/Table6_classification_leave_one_month_out.csv", index=False)
pd.DataFrame([dict(Model=m, **{k: round(v, 3) for k, v in x.items()}) for m, x in R["B_reg"].items()]).to_csv("tables/Table7_forecast_regression.csv", index=False)
pd.DataFrame([dict(Model=m, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}) for m, x in R["B_clf"].items()]).to_csv("tables/Table8_forecast_classification.csv", index=False)
pd.DataFrame([dict(Setting=k, Forecast_RMSE=round(v, 3)) for k, v in R["sens"].items()]).to_csv("tables/Table9_sensitivity.csv", index=False)
df[["month","period","leq","lmin","lmax","limit","exceed","margin","flag"]].sort_values(["month","period"]).to_csv("tables/TableA1_monthly_data.csv", index=False)
for fpng in glob.glob("fig*.png"): shutil.copy(fpng, os.path.join("figures", fpng))
print("Experiments done. See figures/ and tables/ folders, results.json and *_results.csv.")

