"""
Noise-level project - ONE-FILE VERSION (12-month file: noise_data_csv_1.csv).
Open this folder in VS Code, open main.py and press the Run (triangle) button.
It installs missing libraries, runs all experiments, makes figures + CSV tables and builds
Noise_Level_Journal_Submission.pdf in this same folder.
"""
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
    raise FileNotFoundError("Put noise_data_csv_1.csv in the same folder as main.py.")
print("STEP 1/2: running experiments and making figures ...")
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
print("Experiments done.")


print("STEP 2/2: building the PDF ...")
import json, numpy as np, pandas as pd
import json, numpy as np, pandas as pd
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                Image, KeepTogether, FrameBreak, NextPageTemplate, PageBreak)

import matplotlib
if os.path.exists("fonts/LiberationSerif-Regular.ttf"):
    _fm = {"Serif":"fonts/LiberationSerif-Regular.ttf","Serif-B":"fonts/LiberationSerif-Bold.ttf","Serif-I":"fonts/LiberationSerif-Italic.ttf","Serif-BI":"fonts/LiberationSerif-BoldItalic.ttf",
           "Sans":"fonts/LiberationSans-Regular.ttf","Sans-B":"fonts/LiberationSans-Bold.ttf","Sans-I":"fonts/LiberationSans-Italic.ttf","Sans-BI":"fonts/LiberationSans-BoldItalic.ttf"}
else:   # fallback: fonts that ship with matplotlib (layout may run to 6 pages)
    _t = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    _fm = {"Serif":"DejaVuSerif.ttf","Serif-B":"DejaVuSerif-Bold.ttf","Serif-I":"DejaVuSerif-Italic.ttf","Serif-BI":"DejaVuSerif-BoldItalic.ttf",
           "Sans":"DejaVuSans.ttf","Sans-B":"DejaVuSans-Bold.ttf","Sans-I":"DejaVuSans-Oblique.ttf","Sans-BI":"DejaVuSans-BoldOblique.ttf"}
    _fm = {k: os.path.join(_t, v) for k, v in _fm.items()}
for n, path in _fm.items():
    pdfmetrics.registerFont(TTFont(n, path))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-BI")

NAVY = colors.HexColor("#1f4e8c"); GREY = colors.HexColor("#555555")
W, H = A4; M = 40; GAP = 16; CW = (W - 2*M - GAP) / 2
body = ParagraphStyle("body", fontName="Serif", fontSize=8.1, leading=9.7, alignment=TA_JUSTIFY, spaceAfter=3.2)
h1 = ParagraphStyle("h1", fontName="Sans-B", fontSize=9.2, leading=11, spaceBefore=5, spaceAfter=2.5, textColor=colors.black, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="Sans-BI", fontSize=8, leading=9.6, spaceBefore=3, spaceAfter=1.5, keepWithNext=1)
cap = ParagraphStyle("cap", fontName="Sans", fontSize=6.6, leading=8, alignment=TA_JUSTIFY, spaceBefore=1.5, spaceAfter=5)
tcell = ParagraphStyle("tc", fontName="Sans", fontSize=6.3, leading=7.4)
tcellb = ParagraphStyle("tcb", parent=tcell, fontName="Sans-B")
title = ParagraphStyle("title", fontName="Sans-B", fontSize=14.5, leading=17.5, alignment=TA_CENTER, spaceAfter=5)
auth = ParagraphStyle("auth", fontName="Sans-B", fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=1)
aff = ParagraphStyle("aff", fontName="Sans", fontSize=6.7, leading=8.2, alignment=TA_CENTER, textColor=GREY, spaceAfter=1)
abst = ParagraphStyle("abst", fontName="Serif", fontSize=8.1, leading=9.8, alignment=TA_JUSTIFY, spaceAfter=3)
refs = ParagraphStyle("refs", parent=body, fontName="Sans", fontSize=6.5, leading=7.9, leftIndent=11, firstLineIndent=-11, alignment=TA_LEFT, spaceAfter=1.8)
eqs = ParagraphStyle("eq", fontName="Serif", fontSize=8.1, leading=10, alignment=TA_CENTER)
eqn = ParagraphStyle("eqn", fontName="Serif", fontSize=8.1, leading=10, alignment=2)

def P(t, s=body): return Paragraph(t, s)
def f(x, n=2): return f"{x:.{n}f}"
def sub(t): return f"<sub>{t}</sub>"
LEQ = "<i>L</i>" + sub("eq")
def eq(t, n):
    tb = Table([[P(t, eqs), P(f"({n})", eqn)]], colWidths=[CW-24, 24]); tb.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),1),("BOTTOMPADDING",(0,0),(-1,-1),2)])); return tb

def table(rows, widths, bold_rows=(), caption=None, align_left_cols=(0,), hdr_rows=1, span=None):
    data = []
    for i, r in enumerate(rows):
        st = tcellb if (i < hdr_rows or i in bold_rows) else tcell
        data.append([P(str(c), ParagraphStyle("x", parent=st, alignment=(TA_LEFT if j in align_left_cols else TA_CENTER))) for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=hdr_rows)
    s = [("LINEABOVE",(0,0),(-1,0),.8,colors.black),("LINEBELOW",(0,hdr_rows-1),(-1,hdr_rows-1),.5,colors.black),("LINEBELOW",(0,-1),(-1,-1),.8,colors.black),
         ("TOPPADDING",(0,0),(-1,-1),1.1),("BOTTOMPADDING",(0,0),(-1,-1),1.1),("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),("VALIGN",(0,0),(-1,-1),"MIDDLE")]
    t.setStyle(TableStyle(s))
    items = []
    if caption: items.append(P(caption, cap))
    items.append(t); items.append(Spacer(1, 6))
    return KeepTogether(items)

def fig(path, caption, w=CW):
    im = PILImage.open(path); h = w * im.size[1] / im.size[0]
    return KeepTogether([Image(path, width=w, height=h), P(caption, cap)])

R = json.load(open("results.json")); D = pd.read_csv("cleaned_noise_dataset.csv", parse_dates=["month"])
Ra, Ca, Rb, Cb, E = R["A_reg"], R["A_clf"], R["B_reg"], R["B_clf"], R["ols"]
pm = lambda x: "&lt;0.001" if x < 0.001 else f(x, 3)
drows = []
for per in ("Day", "Night"):
    q = D[D.period == per].sort_values("midx"); drows.append((per, q.leq.mean(), q.leq.min(), q.leq.max(), int(q.exceed.sum()), q.margin.mean(), q.leq.diff().abs().mean()))
S = []
S.append(P("SHORT COMMUNICATION", ParagraphStyle("sc", fontName="Sans-B", fontSize=7.5, textColor=NAVY, alignment=TA_CENTER, spaceAfter=3)))
S.append(P("Day- and Night-time Noise Levels at a Commercial-Area Monitoring Station: Limit-Exceedance Classification and Regression with Classical Machine-Learning Models", title))
S.append(P("[Student Name]", auth))
S.append(P("Department of Computer Science, [College Name], University of Delhi, New Delhi, India | Roll No.: [xxxx] | E-mail: [name@college.du.ac.in]", aff))
S.append(P("Course: Machine Learning (B.Sc. (H) Computer Science, Semester VI) | Supervisor: [Name] | Problem No. 53 (urban noise-level modelling) from ML_Application_Problems.xlsx | Article type: Short Communication (single-station exploratory study)", aff))
S.append(Spacer(1, 4))
S.append(P(f"<b>Abstract.</b> Traffic noise above prescribed limits harms sleep and cardiovascular health, and India sets separate day (65 dB(A)) and night (55 dB(A)) limits for commercial areas. This exploratory study asks whether classical machine-learning models can (i) explain monthly day- and night-time equivalent noise level ({LEQ}), (ii) classify whether a month exceeds its limit, and (iii) forecast next month’s level. We use one open CPCB/KSPCB file of twelve monthly summaries (April 2019–March 2020; 12 months × day/night = 24 observations), screen it for quality problems, and compare linear, ridge, decision-tree and k-NN regressors and logistic-regression, tree, k-NN and naive-Bayes classifiers with trivial baselines under leave-one-month-out and rolling-origin evaluation. The limit was exceeded in {R['exc_night']} of 12 night-time months and {R['exc_day']} of 12 day-time months. Night-time {LEQ} was only {f(R['diff_mean'],1)} dB(A) below day-time (Wilcoxon p = {f(R['wilcoxon_p'],2)}) although its limit is 10 dB(A) lower. No learned model beat the period-mean baseline when explaining {LEQ} (baseline RMSE {f(Ra['Period mean']['RMSE'])}; ridge {f(Ra['Ridge']['RMSE'])}, R² = {f(Ra['Ridge']['R2'])}), and the majority rule (F1 = {f(Ca['Period majority']['F1'],3)}) matched or beat every classifier (logistic regression F1 = {f(Ca['Logistic reg.']['F1'],3)}, AUC = {f(Ca['Logistic reg.']['AUC'],2)}). In a six-month rolling forecast, ridge regression had the lowest RMSE ({f(Rb['Ridge']['RMSE'])} versus {f(Rb['Persistence']['RMSE'])} for persistence) but the ranking changed with the penalty strength and rests on 12 predictions. Twelve monthly values are too few for machine learning to add value; the robust findings are persistent non-compliance and data-quality anomalies.", abst))
S.append(P("<b>Keywords:</b> noise pollution; equivalent noise level; CPCB limits; regression; logistic regression; k-NN; limit exceedance; time-series validation; SDG 3; SDG 11", ParagraphStyle("kw", parent=abst, fontSize=7.4, leading=9)))
S.append(FrameBreak())
S.append(P("1 Introduction", h1))
S.append(P("Environmental noise is a recognised health risk: reviews link long-term exposure, especially at night, with sleep disturbance, annoyance and cardiovascular disease [3, 4], and the WHO gives separate guideline values for road-traffic noise [2]. In India the Noise Pollution (Regulation and Control) Rules, 2000 set ambient limits as A-weighted equivalent levels; for commercial areas these are 65 dB(A) by day (06:00–22:00) and 55 dB(A) by night (22:00–06:00) [1]. Monthly summaries from continuous noise monitors are published on the open-government-data portal [5] but are rarely modelled."))
S.append(P(f"We use one commercial-area station to ask three questions. <b>Q1</b>: how do day and night {LEQ} depend on the period of day, the month and the festival season? <b>Q2</b>: can a month’s exceedance of the prescribed limit be classified from the same information? <b>Q3</b>: can last month’s level forecast next month’s? We use the classical models of an undergraduate course and always compare them with trivial baselines, because with so few observations a benefit must be demonstrated rather than assumed. Our contributions are: (i) a cleaned, quality-screened dataset; (ii) two leakage-aware evaluation protocols; (iii) an explicit, mostly negative, result; and (iv) a statement of what these data cannot answer (weekday effects). Section 2 reviews related work, Section 3 the data, Sections 4–5 the methods, Section 6 the results and Section 7 the discussion."))
S.append(P("2 Related Work", h1))
S.append(P("Basner et al. [3] and Münzel et al. [4] review the auditory and non-auditory health effects of environmental noise, which motivates night-time limits. For time-ordered data, Bergmeir and Benítez [6] discuss when ordinary cross-validation is valid; we therefore pair a leave-one-month-out analysis (to explain levels) with forward-chaining evaluation (to forecast). The national COVID-19 lockdown began on 25 March 2020, in the last week of our data, so its effect cannot be separated here. All models were implemented with scikit-learn [7]; the algorithms follow [8]."))
S.append(P("3 Dataset and Preprocessing", h1))
S.append(P("3.1 Source and description", h2))
S.append(P(f"The data are monthly noise summaries from a CPCB/KSPCB commercial-area station on the Open Government Data (OGD) Platform India [5], supplied as <i>noise_data_csv.csv</i> (April 2019–March 2020). Each row is a month with, for day and night separately, the limit, {LEQ}, minimum (L{sub('min')}) and maximum (L{sub('max')}) levels in dB(A), a “% Increase” over the limit and the number of days (Table 1); day is 06:00–22:00 and night 22:00–06:00. After removing the header and footnote rows we obtained 12 months × 2 periods = 24 observations with no missing values."))
S.append(table([["Variable","Unit","Missing","Use in this study"],["Month","month","0 %","split, season (sin/cos)"],["Limit (day / night)","dB(A)","0 %","exceedance label (65 / 55)"],
  [f"{LEQ} day, night","dB(A)","0 %","regression target; label source"],[f"L{sub('min')}, L{sub('max')} (day, night)","dB(A)","0 %","quality screening only"],
  ["% Increase","%","0 %","not used (derived from target: leakage)"],["No. of days","days","0 %","not used"]],[CW*0.31,CW*0.12,CW*0.12,CW*0.45],align_left_cols=(0,3),
  caption="<b>Table 1.</b> Raw variables in the CSV file (n = 12 months, each with a day and a night value)."))
S.append(P("3.2 Data quality", h2))
S.append(P(f"The published “% Increase” column agrees with the rule {LEQ} &gt; limit in all {R['label_agree']} of 24 observations, so labels were derived from {LEQ} and the limit directly. Two anomalies were found. First, {R['n_flag_lmin']} of the 12 day-time rows (April–December 2019) report an identical L{sub('min')} of 21.6 dB(A), a physically implausible constant for a commercial area that suggests a stuck or default reading; one of them (July 2019) also has L{sub('max')} equal to {LEQ} (78.5 dB(A)). Second, the series is erratic: the mean absolute month-to-month change in {LEQ} is {f(R['mad_day'],1)} dB (day) and {f(R['mad_night'],1)} dB (night). We cannot tell whether this reflects real activity or measurement problems. Flagged rows were kept (the target itself shows no direct error) and tested in a sensitivity analysis (Section 6.5)."))
S.append(fig("fig1_timeseries.png", f"<b>Fig. 1.</b> Monthly day and night {LEQ}, April 2019–March 2020, with the 65 and 55 dB(A) limits (dashed). Yellow: festival season (October–November)."))
S.append(P("3.3 Labels and features", h2))
S.append(P(f"The label is <i>c</i> = 1 if {LEQ} &gt; the limit of its period and 0 otherwise (night {LEQ} of exactly 55.0 dB(A) in September 2019 is “within limit”). Overall {R['exc_total']} of 24 observations ({f(100*R['exc_total']/24,0)}%) exceed the limit (Fig. 2). Features (Table 2) are the period (night = 1), the month as sin/cos, a festival-season flag (October–November, Dasara–Deepavali) and, for forecasting, the previous month’s value. Features were standardised inside each pipeline using training data only."))
S.append(P("<b>Weekday effects could not be modelled.</b> The task mentions weekday analysis, but the file contains monthly aggregates only. We state this limitation rather than invent a proxy; daily or hourly records from the CPCB noise monitoring network would be needed."))
S.append(table([["Feature","Definition","Used in"],["night","1 if night-time row, else 0","A, B"],["month sin, cos","sin / cos(2π·month/12)","A, B"],["festival","1 for October and November","A, B"],
  ["lag",f"previous month’s {LEQ} of the same period (relative to the limit for classifiers)","B"]],[CW*0.22,CW*0.64,CW*0.14],align_left_cols=(0,1),
  caption="<b>Table 2.</b> Engineered features; A = leave-one-month-out protocol, B = forecasting protocol."))
S.append(fig("fig2_margin_heatmap.png", f"<b>Fig. 2.</b> Margin {LEQ} − limit (dB) by month and period. Red = exceedance, blue = within limit."))
S.append(P("4 Methodology", h1))
S.append(P("Fig. 3 summarises the workflow. The equivalent level is the energy average of the A-weighted sound pressure <i>p</i> over a period <i>T</i> (<i>p</i><sub>0</sub> = 20 µPa):"))
S.append(eq(f"{LEQ} = 10 log<sub>10</sub> [ (1/<i>T</i>) ∫ (<i>p</i>(<i>t</i>)/<i>p</i><sub>0</sub>)² d<i>t</i> ],     <i>c</i> = 1 if {LEQ} &gt; <i>L</i><sub>lim</sub>", 1))
S.append(fig("fig3_pipeline.png", "<b>Fig. 3.</b> Methodology pipeline (square brackets: scaler fitted inside the training folds)."))
S.append(P("4.1 Regression models", h2))
S.append(P("<b>Multiple linear regression (OLS)</b> fits ŷ = <b>w</b><super>T</super><b>x</b> + <i>b</i> by least squares. <b>Ridge</b> adds an L2 penalty against collinearity:"))
S.append(eq("<i>J</i>(<b>w</b>) = Σ<sub>i</sub> (<i>y</i><sub>i</sub> − ŷ<sub>i</sub>)² + λ Σ<sub>j</sub> <i>w</i><sub>j</sub>²", 2))
S.append(P("A <b>regression tree</b> (depth 3, at least 2 samples per leaf) splits the features to minimise within-node squared error; <b>k-NN regression</b> (k = 3, distance-weighted, Euclidean distance) averages the targets of the nearest training months. <b>Baselines:</b> the training mean per period and, for forecasting, <i>persistence</i> (next month = this month)."))
S.append(P("4.2 Classification models", h2))
S.append(P(f"<b>Logistic regression</b> models <i>P</i>(<i>c</i> = 1 | <b>x</b>) = σ(<b>w</b><super>T</super><b>x</b> + <i>b</i>) with σ(<i>z</i>) = 1/(1 + e<super>−<i>z</i></super>) and an L2 penalty (C = 1). We also use a depth-3 <b>decision tree</b>, <b>k-NN</b> (k = 3) and <b>Gaussian naive Bayes</b>. Baselines are the majority class per period, persistence of last month’s label, and “regression-then-threshold” (OLS predicts {LEQ}; flag a month if it exceeds the limit)."))
S.append(P("4.3 Evaluation metrics", h2))
S.append(P("Regression: MAE, RMSE and R² = 1 − Σ(<i>y</i> − ŷ)² / Σ(<i>y</i> − ȳ)². R² is negative whenever a model predicts worse than the mean of the evaluation data, which is common with few, noisy observations. Classification: accuracy, precision, recall, F1 and ROC-AUC; recall and F1 are emphasised because a missed exceedance is costlier than a false alarm. Uncertainty is shown by a month-level bootstrap (5,000 resamples)."))
S.append(P("5 Experimental Setup", h1))
S.append(P(f"<b>Protocol A (explanatory): leave-one-month-out.</b> Each of the 12 months is held out in turn (day and night rows together) and the model is trained on the other 11, giving 24 out-of-fold predictions. This is interpolation, not forecasting. <b>Protocol B (forecasting): rolling origin.</b> For each month <i>t</i> from October 2019 to March 2020, models are trained only on earlier months that have a lag value (5 to 10 months) and predict both periods of month <i>t</i> ({R['B_n']} predictions). <b>Hyper-parameters</b> were fixed in advance (ridge λ = 1; tree depth 3; k = 3; C = 1) because 12 months are far too few for nested tuning; Section 6.5 shows their sensitivity. Preprocessing sits inside a scikit-learn Pipeline. <b>Environment:</b> Python 3.12, scikit-learn, pandas, NumPy, SciPy, Matplotlib; random seed 42. The accompanying code regenerates every number and figure."))
S.append(P("6 Results", h1))
S.append(P("6.1 Descriptive findings", h2))
rows = [["Period","Mean","Min–max","Over limit","Mean margin","|Δ| m-o-m"]]
for per, m, lo, hi, ex, mg, ma in drows: rows.append([per, f(m,1), f"{lo:.1f}–{hi:.1f}", f"{ex}/12", f"{mg:+.1f}", f(ma,1)])
S.append(table(rows,[CW*0.14,CW*0.14,CW*0.24,CW*0.16,CW*0.18,CW*0.14],caption=f"<b>Table 3.</b> Descriptive statistics of monthly {LEQ} (dB(A)); margin = {LEQ} − limit; |Δ| m-o-m = mean absolute month-to-month change."))
S.append(P(f"The night-time limit was exceeded in {R['exc_night']} of 12 months and the day-time limit in {R['exc_day']} of 12 (Table 3, Figs 1–2); only September 2019 had no exceedance in either period. Night {LEQ} was {f(R['diff_mean'],1)} dB(A) below day {LEQ} on average, a difference that is not significant (Wilcoxon signed-rank p = {f(R['wilcoxon_p'],2)}); because the night limit is 10 dB(A) lower, the night margin ({drows[1][5]:+.1f} dB) is larger than the day margin ({drows[0][5]:+.1f} dB). Festival months (October–November) were not noisier in monthly averages: the exceedance rate was {f(100*R['fest_rate'],0)}% in both festival and other months and the mean margin was {R['fest_margin']:+.1f} versus {R['non_margin']:+.1f} dB, although L{sub('max')} reached 104.4 dB(A) (day) in October 2019. Monthly {LEQ} averages dilute such short events."))
S.append(P("6.2 Explaining Leq (Protocol A)", h2))
names = ["Period mean","OLS","Ridge","Decision tree","k-NN"]
rows = [["Model","MAE","RMSE","R²"]] + [[n, f(Ra[n]["MAE"]), f(Ra[n]["RMSE"]), f(Ra[n]["R2"])] for n in names]
S.append(table(rows,[CW*0.46,CW*0.18,CW*0.18,CW*0.18],bold_rows=(1,),caption=f"<b>Table 4.</b> Leave-one-month-out regression of {LEQ} (dB(A); 24 out-of-fold predictions). Best RMSE in bold."))
S.append(P(f"No learned model beat the period-mean baseline (RMSE {f(Ra['Period mean']['RMSE'])}): ridge reached {f(Ra['Ridge']['RMSE'])} (R² {f(Ra['Ridge']['R2'])}), OLS {f(Ra['OLS']['RMSE'])}, k-NN {f(Ra['k-NN']['RMSE'])} and the tree {f(Ra['Decision tree']['RMSE'])} (Table 4, Fig. 4). All R² values are negative: out of sample, month and festival information predicts worse than the period mean. The bootstrap 95% intervals of ridge ({f(R['A_boot']['Ridge'][0])}–{f(R['A_boot']['Ridge'][1])}) and the baseline ({f(R['A_boot']['Period mean'][0])}–{f(R['A_boot']['Period mean'][1])}) overlap almost completely."))
S.append(fig("fig4_lomo_scatter.png", "<b>Fig. 4.</b> Observed versus leave-one-month-out predicted Leq for ridge regression; grey line = perfect prediction."))
rows = [["Term","Coef. (dB)","95% CI","p"]]
for k, lab in [("const","Intercept (day)"),("night","Night"),("festival","Festival season"),("msin","Month sin"),("mcos","Month cos")]:
    e = E[k]; rows.append([lab, f"{e['b']:+.2f}", f"{e['lo']:.1f} to {e['hi']:.1f}", pm(e['p']) if e['p'] < .001 else f(e['p'],3)])
S.append(table(rows,[CW*0.40,CW*0.17,CW*0.27,CW*0.16],caption=f"<b>Table 5.</b> OLS coefficients on all 24 observations (R² = {f(R['ols_r2'])}, {R['ols_dof']} residual d.f.). Indicative only: monthly residuals may be autocorrelated."))
S.append(P(f"In the descriptive OLS fit (Table 5) every interval for night, festival and the seasonal terms includes zero; the data are consistent with, but do not demonstrate, such effects. The in-sample R² of {f(R['ols_r2'])} is not borne out out-of-sample (Table 4)."))
S.append(P("6.3 Classifying limit exceedance (Protocol A)", h2))
cn = ["Period majority","OLS>limit","Logistic reg.","Decision tree","k-NN","Gauss. NB"]
rows = [["Model","Acc.","Prec.","Rec.","F1","AUC"]] + [[n.replace("OLS>limit","OLS then threshold"), f(Ca[n]['Acc'],3), f(Ca[n]['Prec'],3), f(Ca[n]['Rec'],3), f(Ca[n]['F1'],3), "–" if Ca[n]['AUC'] is None else f(Ca[n]['AUC'],3)] for n in cn]
S.append(table(rows,[CW*0.34,CW*0.13,CW*0.13,CW*0.13,CW*0.13,CW*0.14],bold_rows=(1,),caption="<b>Table 6.</b> Leave-one-month-out classification of limit exceedance (18 of 24 observations exceed). Best F1 in bold."))
S.append(P(f"The majority-by-period rule (day: within limit; night: exceeds) achieved the best F1 ({f(Ca['Period majority']['F1'],3)}); logistic regression reached F1 {f(Ca['Logistic reg.']['F1'],3)} with accuracy {f(Ca['Logistic reg.']['Acc'],3)} but an AUC of only {f(Ca['Logistic reg.']['AUC'],2)}, i.e. almost no ability to rank months. The tree and k-NN were worse (F1 {f(Ca['Decision tree']['F1'],3)} and {f(Ca['k-NN']['F1'],3)}), and thresholding the regression output was worse than classifying directly (F1 {f(Ca['OLS>limit']['F1'],3)}). The odds ratio for night in the logistic model is {f(R['or']['night'],1)}, but it is estimated from only 24 rows."))
S.append(P("6.4 One-month-ahead forecasting (Protocol B)", h2))
fn = ["Persistence","Period mean","OLS","Ridge","Decision tree","k-NN"]
rows = [["Model","MAE","RMSE","R²"]] + [[n, f(Rb[n]['MAE']), f(Rb[n]['RMSE']), f(Rb[n]['R2'])] for n in fn]
S.append(table(rows,[CW*0.46,CW*0.18,CW*0.18,CW*0.18],bold_rows=(4,),caption=f"<b>Table 7.</b> Rolling-origin forecast of {LEQ} (dB(A)), October 2019–March 2020; 12 predictions. Best RMSE in bold."))
S.append(P(f"Ridge regression had the lowest forecast RMSE ({f(Rb['Ridge']['RMSE'])}, R² {f(Rb['Ridge']['R2'])}), ahead of the tree ({f(Rb['Decision tree']['RMSE'])}), k-NN ({f(Rb['k-NN']['RMSE'])}), the period mean ({f(Rb['Period mean']['RMSE'])}) and persistence ({f(Rb['Persistence']['RMSE'])}); OLS was worst ({f(Rb['OLS']['RMSE'])}). Persistence is poor because consecutive months are weakly related (Spearman ρ = {f(R['spearman_lag'],2)} between {LEQ} and its previous-month value). With only 12 predictions, however, differences of 1–2 dB are not statistically meaningful, and every R² is still negative (Fig. 5)."))
S.append(fig("fig5_predictions.png", f"<b>Fig. 5.</b> Rolling-origin forecasts of monthly {LEQ} for day (top) and night (bottom): observed, persistence and ridge. Red dashed: limit."))
cn2 = ["Persistence","Period majority","OLS>limit","Logistic reg.","Decision tree","k-NN","Gauss. NB"]
rows = [["Model","Acc.","Prec.","Rec.","F1"]] + [[n.replace("OLS>limit","OLS then threshold"), f(Cb[n]['Acc'],3), f(Cb[n]['Prec'],3), f(Cb[n]['Rec'],3), f(Cb[n]['F1'],3)] for n in cn2]
S.append(table(rows,[CW*0.38,CW*0.155,CW*0.155,CW*0.155,CW*0.155],bold_rows=(2,),caption="<b>Table 8.</b> Rolling-origin classification of exceedance (12 predictions; 11 exceedances)."))
S.append(P(f"For exceedance, the majority rule is again best (F1 {f(Cb['Period majority']['F1'],3)}) and no learned classifier improves on it (Table 8). {f(100*R['B_exc_rate'],0)}% of the forecast observations exceeded the limit against {f(100*R['B_train_rate'],0)}% in the early training months, a base-rate shift that favours rules that always predict the common class."))
S.append(P("6.5 Sensitivity", h2))
sn = R["sens"]
rows = [["Setting","Forecast RMSE (B)"],["k-NN, k = 2 / 3 / 5",f"{f(sn['k-NN k=2'])} / {f(sn['k-NN k=3'])} / {f(sn['k-NN k=5'])}"],["Tree depth 2 / 3 / 4",f"{f(sn['Tree depth=2'])} / {f(sn['Tree depth=3'])} / {f(sn['Tree depth=4'])}"],
  ["Ridge λ = 0.1 / 1 / 10",f"{f(sn['Ridge alpha=0.1'])} / {f(sn['Ridge alpha=1'])} / {f(sn['Ridge alpha=10'])}"],
  ["July-2019 day row dropped: OLS / ridge / tree / k-NN",f"{f(sn['OLS (Jul-19 day row dropped)'])} / {f(sn['Ridge (Jul-19 day row dropped)'])} / {f(sn['Decision tree (Jul-19 day row dropped)'])} / {f(sn['k-NN (Jul-19 day row dropped)'])}"],["Persistence (reference)",f(Rb['Persistence']['RMSE'])]]
S.append(table(rows,[CW*0.6,CW*0.4],caption="<b>Table 9.</b> Sensitivity of the forecasting RMSE to hyper-parameters and to the one row with L<sub>max</sub> = L<sub>eq</sub>."))
S.append(P(f"The forecasting ranking is unstable: ridge ranges from {f(sn['Ridge alpha=10'])} (λ = 10) to {f(sn['Ridge alpha=0.1'])} (λ = 0.1), the latter worse than persistence, and k-NN from {f(sn['k-NN k=5'])} to {f(sn['k-NN k=2'])}. We therefore do not claim that any learned model forecasts better than the baselines. Only the July-2019 day row could be dropped as a quality test, since the nine rows with L<sub>min</sub> = 21.6 dB(A) make up three quarters of the day-time data."))
S.append(P("7 Discussion", h1))
S.append(P(f"<b>What the data show.</b> The clearest result is regulatory rather than predictive: the 55 dB(A) night limit was exceeded in {R['exc_night']} of 12 months and the 65 dB(A) day limit in {R['exc_day']} of 12. Since night limits protect sleep [2–4], persistent night-time non-compliance in a commercial area is the finding most relevant to a city administration. <b>What did not work.</b> With 24 observations and only month, period and festival information, learned models did not beat a period mean or a majority rule in explanation or classification, and their forecasting advantage was not stable. This is a statement about the data size, not about the algorithms."))
S.append(P("<b>Limitations.</b> (i) Only twelve monthly values from one station; all intervals are wide and p-values are indicative. (ii) No weekday analysis is possible from monthly aggregates. (iii) Nine day-time rows carry a constant L<sub>min</sub>, so measurement quality is uncertain. (iv) The festival flag (October–November) is coarse. (v) The data end as the 2020 lockdown began, so its effect cannot be studied. (vi) No traffic, weather or event data were available. Extending the series with the following year’s monthly file would be the most valuable next step."))
S.append(P("<b>Ethics and SDG impact.</b> The data are aggregate environmental measurements with no personal information. The main risk is over-interpretation of a very small sample, so we report baselines, intervals and failures. Better use of open noise data supports SDG 11 (sustainable cities, including target 11.6) and SDG 3 (health and well-being)."))
S.append(P("8 Conclusion and Future Work", h1))
S.append(P(f"Using one open CPCB/KSPCB monthly file for a commercial-area station (April 2019–March 2020), we found that night-time noise exceeded the prescribed limit in {R['exc_night']} of 12 months and day-time noise in {R['exc_day']} of 12, and that classical models (ridge R² = {f(Ra['Ridge']['R2'])}; logistic regression F1 = {f(Ca['Logistic reg.']['F1'],2)}, AUC = {f(Ca['Logistic reg.']['AUC'],2)}) do not outperform trivial baselines on such a short series. Future work should use daily or hourly data (enabling weekday and festival-day analysis), longer records and several stations, add traffic and weather covariates, and verify the instrument anomalies with the monitoring agency."))
S.append(P("Declarations", h1))
S.append(P("<b>Data availability.</b> The file was obtained from the OGD Platform India resource “Noise levels measured at CAAQMS of CPCB at BWSSB (Commercial Area)” [5]; the cleaned table, predictions and all code (main.py) are in the assignment folder [student to add GitHub link]."))
S.append(P("<b>Conflict of interest.</b> The author declares no conflict of interest. <b>Funding.</b> None."))
S.append(P("<b>Use of AI tools.</b> [Student to edit and confirm.] An AI assistant (Claude, Anthropic) was used to write the analysis code, execute the experiments and draft this report from the supplied data and template; every number in the paper is produced by the accompanying code. The author must state precisely what was done by them, verify all content and take full responsibility for it."))
S.append(P("References", h1))
for r in ["[1] Ministry of Environment and Forests, Government of India, “The Noise Pollution (Regulation and Control) Rules, 2000,” S.O. 123(E), New Delhi, 14 Feb. 2000 (as amended).",
"[2] World Health Organization Regional Office for Europe, <i>Environmental Noise Guidelines for the European Region</i>. Copenhagen, Denmark, 2018.",
"[3] M. Basner et al., “Auditory and non-auditory effects of noise on health,” <i>The Lancet</i>, vol. 383, no. 9925, pp. 1325–1332, 2014, doi: 10.1016/S0140-6736(13)61613-X.",
"[4] T. Münzel et al., “Cardiovascular effects of environmental noise exposure,” <i>European Heart Journal</i>, vol. 35, no. 13, pp. 829–836, 2014, doi: 10.1093/eurheartj/ehu030.",
"[5] CPCB / KSPCB, “Noise levels measured at CAAQMS of CPCB at BWSSB (Commercial Area),” Open Government Data (OGD) Platform India. [Online]. Available: https://www.data.gov.in/resource/noise-levels-measured-caaqms-cpcb-bwssb-commercial-area (accessed [date]).",
"[6] C. Bergmeir and J. M. Benítez, “On the use of cross-validation for time series predictor evaluation,” <i>Information Sciences</i>, vol. 191, pp. 192–213, 2012, doi: 10.1016/j.ins.2011.12.028.",
"[7] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825–2830, 2011.",
"[8] T. Hastie, R. Tibshirani and J. Friedman, <i>The Elements of Statistical Learning</i>, 2nd ed. New York, NY, USA: Springer, 2009."]: S.append(P(r, refs))
S.append(P("Appendix A. Monthly data used", h1))
dd = D.pivot(index="midx", columns="period", values=["leq","lmin","lmax"]); mo = D[D.period=="Day"].sort_values("midx").month.dt.strftime("%b-%y").values
rows = [["Month","Day Leq","Lmin","Lmax","Night Leq","Lmin","Lmax","Over (D/N)"]]
for i in range(12):
    ex = D[D.midx==i].set_index("period").exceed
    rows.append([mo[i], f(dd.leq.Day[i],1), f(dd.lmin.Day[i],1), f(dd.lmax.Day[i],1), f(dd.leq.Night[i],1), f(dd.lmin.Night[i],1), f(dd.lmax.Night[i],1), f"{'Y' if ex['Day'] else 'N'} / {'Y' if ex['Night'] else 'N'}"])
S.append(table(rows,[CW*0.14,CW*0.12,CW*0.11,CW*0.11,CW*0.14,CW*0.11,CW*0.11,CW*0.16],caption="<b>Table A1.</b> Monthly values (dB(A)); Over = exceeds limit (Y/N) for day / night. Day L<sub>min</sub> = 21.6 for Apr–Dec 2019 is flagged (Section 3.2)."))
def deco(c, doc):
    c.saveState(); c.setFont("Sans-B", 6.6); c.setFillColor(NAVY)
    c.drawString(M, H-24, "SHORT COMMUNICATION — ML Application Assignment, B.Sc. (H) Computer Science, Sem VI (University of Delhi)"); c.drawRightString(W-M, H-24, "Journal-format submission")
    c.setStrokeColor(NAVY); c.setLineWidth(.6); c.line(M, H-28, W-M, H-28); c.setFillColor(GREY); c.setFont("Sans", 6.3); c.line(M, 30, W-M, 30)
    c.drawString(M, 21, "Problem No. 53 — Day/night noise-level modelling at a CPCB/KSPCB commercial-area station. Real data, real results (see Data availability)."); c.drawRightString(W-M, 21, f"Page {doc.page}"); c.restoreState()
top = H - 36; bot = 38
_fb = type(FrameBreak()); _idx = next(i for i, x in enumerate(S) if isinstance(x, _fb))
TH = sum(x.wrap(W - 2*M, 2000)[1] + x.getSpaceBefore() + x.getSpaceAfter() for x in S[:_idx] if hasattr(x, "wrap")) + 6
kw = dict(leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
first = PageTemplate("first", frames=[Frame(M, top-TH, W-2*M, TH, id="t", **kw), Frame(M, bot, CW, top-TH-bot, id="c1", **kw), Frame(M+CW+GAP, bot, CW, top-TH-bot, id="c2", **kw)], onPage=deco)
later = PageTemplate("later", frames=[Frame(M, bot, CW, top-bot, id="l1", **kw), Frame(M+CW+GAP, bot, CW, top-bot, id="l2", **kw)], onPage=deco)
doc = BaseDocTemplate("Noise_Level_Journal_Submission.pdf", pagesize=A4, pageTemplates=[first, later], title="Day- and Night-time Noise Levels at a Commercial-Area Monitoring Station", author="[Student Name]")
S.insert(0, NextPageTemplate("later")); doc.build(S)
print("built")

print("\nALL DONE -> open Noise_Level_Journal_Submission.pdf (figures, CSV tables and results.json are in this folder too).")
