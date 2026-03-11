import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading
import datetime

# Librerie opzionali
try:
    import yfinance as yf  # type: ignore[import]
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import requests
    REQ_AVAILABLE = True
except ImportError:
    REQ_AVAILABLE = False

try:
    import pandas as pd  # type: ignore[import]
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

# ── Costanti fiscali danesi ────────────────────────────────────────────────────
ASK_DEPOSIT_LIMIT = 174200
ASK_TAX           = 0.17
KAPITAL_TAX       = 0.40
DEFAULT_YEARS     = 4
ASK_LIMIT_GROWTH  = 0.025   # crescita annua limite ASK (~2.5%)

# ── Temi ──────────────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "BG": "#0d1117", "BG2": "#161b22", "BG3": "#1c2128",
        "BORDER": "#30363d", "TEXT": "#e6edf3", "TEXT_DIM": "#7d8590",
        "TEXT_MID": "#adbac7",
    },
    "light": {
        "BG": "#f6f8fa", "BG2": "#ffffff", "BG3": "#eaeef2",
        "BORDER": "#d0d7de", "TEXT": "#1f2328", "TEXT_DIM": "#656d76",
        "TEXT_MID": "#424a53",
    },
}
_theme = "dark"

def _T(key):
    return THEMES[_theme][key]

# Colori fissi (uguali in entrambi i temi)
BLUE   = "#58a6ff"
ORANGE = "#f0883e"
GREEN  = "#3fb950"
PURPLE = "#bc8cff"
RED    = "#ff7b72"
YELLOW = "#e3b341"
CYAN   = "#39d353"
PINK   = "#f778ba"
TEAL   = "#56d364"

# Font
F_MONO    = "Consolas"
F_TITLE   = ("Consolas", 13, "bold")
F_HEADER  = ("Consolas", 10, "bold")
F_BODY    = ("Consolas", 9)
F_SMALL   = ("Consolas", 8)
F_KPI     = ("Consolas", 12, "bold")
F_KPI_LBL = ("Consolas", 8)

# Shortcut proprietà tema
def BG():     return _T("BG")
def BG2():    return _T("BG2")
def BG3():    return _T("BG3")
def BORDER(): return _T("BORDER")
def TEXT():   return _T("TEXT")
def TEXT_DIM(): return _T("TEXT_DIM")
def TEXT_MID(): return _T("TEXT_MID")

# ── ETF lista ABIS/SKAT ────────────────────────────────────────────────────────
DEFAULT_ETF_LIST = [
    ("CSPX.L",  "IE00B5BMR087", "iShares Core S&P 500 Acc",     "S&P500"),
    ("VUSA.AS", "IE00B3XXRP09", "Vanguard S&P 500 Acc",         "S&P500"),
    ("SWDA.L",  "IE00B4L5Y983", "iShares Core MSCI World Acc",  "World"),
    ("EUNL.DE", "IE00B4L5Y983", "iShares Core MSCI World (DE)", "World"),
    ("VWCE.DE", "IE00BK5BQT80", "Vanguard FTSE All-World Acc",  "All-World"),
    ("IUSQ.DE", "IE00B4L5Y983", "iShares MSCI World Acc",       "World"),
    ("SPPW.DE", "IE00BYML9W36", "SPDR MSCI World Acc",          "World"),
    ("MEUD.PA", "LU1681044480", "Amundi MSCI Europe Acc",       "Europe"),
    ("QDVE.DE", "IE00B3XXRP09", "iShares S&P 500 IT Acc",       "Sector-IT"),
    ("IQQQ.DE", "IE00B3RBWM25", "iShares Nasdaq 100 Acc",       "Nasdaq"),
    ("VFEM.AS", "IE00B3VVMM84", "Vanguard FTSE EM Acc",         "EM"),
    ("AGGH.L",  "IE00BDBRDM35", "iShares Global Agg Bond Acc",  "Bond"),
]

_etf_cache = {}
_DKK_TO_EUR_FALLBACK = 0.1340

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def fmt(n):
    return f"{int(round(n)):,} kr".replace(",", ".")

def fmt_pct(v):
    return f"{v*100:.1f}%"

def fetch_dkk_eur_rate() -> float:
    if YF_AVAILABLE:
        try:
            ticker = yf.Ticker("DKKEUR=X")
            hist   = ticker.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
    if REQ_AVAILABLE:
        try:
            r = requests.get("https://open.er-api.com/v6/latest/DKK", timeout=5)
            if r.status_code == 200:
                return float(r.json()["rates"]["EUR"])
        except Exception:
            pass
    return _DKK_TO_EUR_FALLBACK

# ══════════════════════════════════════════════════════════════════════════════
# ETF DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_etf_data(ticker: str, years_back: int = 5) -> dict:
    if ticker in _etf_cache:
        return _etf_cache[ticker]
    if not YF_AVAILABLE:
        return _fallback_etf(ticker)
    try:
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=years_back * 365)
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(start=start.isoformat(),
                                  end=end.isoformat(), interval="1mo")
        if hist.empty or len(hist) < 6:
            return _fallback_etf(ticker)
        prices = hist["Close"].dropna().values
        n_years = len(prices) / 12
        cagr    = (prices[-1] / prices[0]) ** (1 / n_years) - 1
        monthly_returns = np.diff(prices) / prices[:-1]
        vol_annual = np.std(monthly_returns) * np.sqrt(12)
        rf_monthly = 0.03 / 12
        sharpe = ((np.mean(monthly_returns) - rf_monthly)
                  / np.std(monthly_returns) * np.sqrt(12)
                  if np.std(monthly_returns) > 0 else 0)
        cumulative  = np.cumprod(1 + monthly_returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown    = (cumulative - rolling_max) / rolling_max
        max_dd      = float(np.min(drawdown))
        ytd = 0.0
        try:
            hist_ytd = ticker_obj.history(
                start=f"{end.year}-01-01", end=end.isoformat())
            if not hist_ytd.empty:
                p   = hist_ytd["Close"].dropna().values
                ytd = (p[-1] / p[0]) - 1 if len(p) > 1 else 0.0
        except Exception:
            pass
        result = {
            "ticker": ticker, "annual_return": round(cagr, 4),
            "volatility": round(vol_annual, 4), "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4), "ytd": round(ytd, 4),
            "prices": prices.tolist(), "monthly_ret": monthly_returns.tolist(),
            "last_price": float(prices[-1]), "n_months": len(prices), "status": "ok",
        }
        _etf_cache[ticker] = result
        return result
    except Exception as e:
        return _fallback_etf(ticker, str(e))


def _fallback_etf(ticker: str, error: str = "no data") -> dict:
    defaults = {
        "CSPX.L":  {"annual_return": 0.112, "volatility": 0.175, "sharpe": 0.65},
        "VUSA.AS": {"annual_return": 0.112, "volatility": 0.175, "sharpe": 0.65},
        "SWDA.L":  {"annual_return": 0.095, "volatility": 0.155, "sharpe": 0.55},
        "VWCE.DE": {"annual_return": 0.090, "volatility": 0.150, "sharpe": 0.52},
        "IQQQ.DE": {"annual_return": 0.150, "volatility": 0.240, "sharpe": 0.60},
        "AGGH.L":  {"annual_return": 0.020, "volatility": 0.060, "sharpe": 0.10},
    }
    base = defaults.get(ticker, {"annual_return": 0.08, "volatility": 0.18, "sharpe": 0.40})
    return {**base, "ticker": ticker, "max_drawdown": -0.35,
            "ytd": 0.0, "prices": [], "monthly_ret": [],
            "last_price": 0.0, "n_months": 0, "status": f"fallback ({error})"}


def score_etf(data: dict) -> float:
    sharpe = min(max(data.get("sharpe", 0), -1), 3)
    cagr   = min(max(data.get("annual_return", 0), -0.5), 0.5)
    mdd    = max(data.get("max_drawdown", -1), -1)
    vol    = min(data.get("volatility", 1), 1)
    return (0.40 * sharpe / 3 + 0.30 * cagr / 0.5
            + 0.20 * (1 + mdd) + 0.10 * (1 - vol))

# ══════════════════════════════════════════════════════════════════════════════
# SIMULAZIONE con PAL-skat annuale reale + Inflazione
# ══════════════════════════════════════════════════════════════════════════════

def simulate_monthly(params, events=None):
    months        = params["years"] * 12
    cur_gross     = params["grossSalary"]
    cur_itax      = params["incomeTax"]
    cur_expenses  = params["monthlyExpenses"]
    cur_sav_r     = params["savingsReturn"]
    portfolio     = params.get("portfolio", [])
    inflation     = params.get("inflation", 0.0)
    pal_annual    = params.get("palAnnual", True)
    ask_lim_grow  = params.get("askLimitGrowth", 0.0)

    monthly_overrides = {}
    if events:
        for ev in events:
            m = ev["month"]
            if m not in monthly_overrides:
                monthly_overrides[m] = {}
            monthly_overrides[m][ev["field"]] = ev["value"]

    etf_states = []
    for etf in portfolio:
        etf_states.append({
            "ticker":        etf["ticker"],
            "pct":           etf["pct"],
            "r_annual":      etf["annual_return"],
            "deposited":     0.0,
            "balance":       0.0,
            "ask_dep":       0.0,
            "balance_jan1":  0.0,
        })

    ops_balance       = 0.0
    total_deposited   = 0.0
    ask_deposited     = 0.0
    current_ask_limit = ASK_DEPOSIT_LIMIT

    data = []
    net0 = cur_gross * (1 - cur_itax)
    data.append({
        "month": 0, "year_frac": 0.0,
        "etf_balances": {e["ticker"]: 0.0 for e in portfolio},
        "ask_balance": 0.0, "ops_balance": 0.0, "total": 0.0,
        "deposited": 0.0, "net_gain": 0.0, "net_monthly": net0,
        "sav_monthly": net0 - cur_expenses, "ask_deposited": 0.0,
        "ask_full": False, "real_total": 0.0,
        "ask_limit": current_ask_limit,
    })

    for m in range(1, months + 1):
        if m % 12 == 1:
            current_ask_limit *= (1 + ask_lim_grow)
            for es in etf_states:
                es["balance_jan1"] = es["balance"]

        if m in monthly_overrides:
            ov = monthly_overrides[m]
            cur_gross    = ov.get("grossSalary",     cur_gross)
            cur_itax     = ov.get("incomeTax",       cur_itax)
            cur_expenses = ov.get("monthlyExpenses", cur_expenses)
            cur_sav_r    = ov.get("savingsReturn",   cur_sav_r)
            for es in etf_states:
                key = f"etfReturn_{es['ticker']}"
                if key in ov:
                    es["r_annual"] = ov[key]

        net_monthly = cur_gross * (1 - cur_itax)
        sav_monthly = net_monthly - cur_expenses
        ask_full    = ask_deposited >= current_ask_limit

        if sav_monthly > 0:
            remaining_sav = sav_monthly
            for es in etf_states:
                if ask_full:
                    dep = 0.0
                else:
                    desired  = sav_monthly * es["pct"]
                    ask_room = max(0.0, current_ask_limit - ask_deposited)
                    dep      = min(desired, ask_room)
                    ask_deposited += dep
                    es["ask_dep"]  += dep
                es["deposited"] += dep
                es["balance"]   += dep
                remaining_sav   -= dep
            ops_balance     += max(0.0, remaining_sav)
            total_deposited += sav_monthly

        ask_balance_total = 0.0
        etf_balances_snap = {}
        for es in etf_states:
            r_monthly = es["r_annual"] / 12
            gain      = es["balance"] * r_monthly
            if pal_annual:
                es["balance"] += gain
            else:
                es["balance"] += gain * (1 - ASK_TAX)
            etf_balances_snap[es["ticker"]] = es["balance"]
            ask_balance_total += es["balance"]

        if pal_annual and m % 12 == 0:
            for es in etf_states:
                annual_gain = es["balance"] - es["balance_jan1"]
                if annual_gain > 0:
                    tax = annual_gain * ASK_TAX
                    es["balance"] -= tax
            ask_balance_total = sum(es["balance"] for es in etf_states)
            etf_balances_snap = {es["ticker"]: es["balance"] for es in etf_states}

        ops_gain    = ops_balance * (cur_sav_r / 12)
        ops_balance += ops_gain * (1 - KAPITAL_TAX)

        total    = ask_balance_total + ops_balance
        net_gain = total - total_deposited

        inflation_factor = (1 + inflation) ** (m / 12)
        real_total = total / inflation_factor

        data.append({
            "month": m, "year_frac": m / 12,
            "etf_balances": etf_balances_snap,
            "ask_balance": ask_balance_total, "ops_balance": ops_balance,
            "total": total, "deposited": total_deposited, "net_gain": net_gain,
            "net_monthly": net_monthly, "sav_monthly": sav_monthly,
            "ask_deposited": ask_deposited, "ask_full": ask_full,
            "real_total": real_total,
            "ask_limit": current_ask_limit,
        })

    return {
        "data":            data,
        "net_monthly":     data[-1]["net_monthly"],
        "sav_monthly":     data[-1]["sav_monthly"],
        "ask_deposited":   ask_deposited,
        "ask_full":        data[-1]["ask_full"],
        "final_ask":       data[-1]["ask_balance"],
        "final_ops":       data[-1]["ops_balance"],
        "final_total":     data[-1]["total"],
        "final_real":      data[-1]["real_total"],
        "final_deposited": data[-1]["deposited"],
        "final_net_gain":  data[-1]["net_gain"],
        "etf_states":      etf_states,
    }


def find_optimal_monthly(params, events=None):
    if not params.get("portfolio"):
        return {"pct": 0.0, "total": simulate_monthly(params, events)["final_total"]}
    best = {"pct": 0.0, "total": 0.0}
    base_portfolio = params["portfolio"]
    total_etf_pct  = sum(e["pct"] for e in base_portfolio)
    if total_etf_pct == 0:
        return {"pct": 0.0, "total": simulate_monthly(params, events)["final_total"]}
    for scale in range(0, 101, 5):
        factor = scale / 100 / total_etf_pct if total_etf_pct > 0 else 0
        scaled = [{**e, "pct": e["pct"] * factor} for e in base_portfolio]
        res = simulate_monthly({**params, "portfolio": scaled}, events)
        if res["final_total"] > best["total"]:
            best = {"pct": scale / 100, "total": res["final_total"]}
    return best


def simulate_monte_carlo_monthly(params, events=None, n_simulations=300):
    months     = params["years"] * 12
    all_totals = np.zeros((n_simulations, months + 1))
    portfolio  = params.get("portfolio", [])
    sav_mean   = params["savingsReturn"]
    sav_sigma  = 0.01
    for i in range(n_simulations):
        mc_events = list(events) if events else []
        for y in range(params["years"]):
            for etf in portfolio:
                mu    = etf["annual_return"]
                sigma = etf.get("volatility", 0.18)
                r     = float(np.random.lognormal(
                    mean=np.log(1 + mu) - 0.5 * sigma**2, sigma=sigma)) - 1
                for mo in range(y * 12 + 1, (y + 1) * 12 + 1):
                    mc_events.append({"month": mo,
                                      "field": f"etfReturn_{etf['ticker']}", "value": r})
            r_sav = max(0.0, float(np.random.normal(sav_mean, sav_sigma)))
            for mo in range(y * 12 + 1, (y + 1) * 12 + 1):
                mc_events.append({"month": mo, "field": "savingsReturn", "value": r_sav})
        res = simulate_monthly(params, mc_events)
        for mo in range(months + 1):
            all_totals[i, mo] = res["data"][mo]["total"]
    return {
        "p5":  np.percentile(all_totals, 5,  axis=0),
        "p25": np.percentile(all_totals, 25, axis=0),
        "p50": np.percentile(all_totals, 50, axis=0),
        "p75": np.percentile(all_totals, 75, axis=0),
        "p95": np.percentile(all_totals, 95, axis=0),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SCROLL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def make_scrollable(parent, bg_color, width=None):
    outer = tk.Frame(parent, bg=bg_color)
    if width:
        outer.configure(width=width)
    cv = tk.Canvas(outer, bg=bg_color, highlightthickness=0, borderwidth=0)
    sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
    inner = tk.Frame(cv, bg=bg_color)
    inner.bind("<Configure>",
               lambda e: cv.configure(scrollregion=cv.bbox("all")))
    win_id = cv.create_window((0, 0), window=inner, anchor="nw")
    cv.configure(yscrollcommand=sb.set)
    def _on_resize(e):
        cv.itemconfig(win_id, width=e.width)
    cv.bind("<Configure>", _on_resize)
    cv.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    def _scroll(e):
        cv.yview_scroll(int(-1 * (e.delta / 120)), "units")
    cv.bind("<MouseWheel>", _scroll)
    inner.bind("<MouseWheel>", _scroll)
    return outer, inner, cv

# ══════════════════════════════════════════════════════════════════════════════
# TOOLTIP
# ══════════════════════════════════════════════════════════════════════════════

class ChartTooltip:
    """Tooltip interattivo su grafico matplotlib."""
    def __init__(self, fig, canvas, ax_getter, data_getter, fmt_fn):
        self.fig        = fig
        self.canvas     = canvas
        self.ax_getter  = ax_getter   # callable → ax corrente
        self.data_getter = data_getter # callable → lista dict dati
        self.fmt_fn     = fmt_fn
        self._annot     = None
        self._cid       = canvas.mpl_connect("motion_notify_event", self._on_move)

    def _on_move(self, event):
        ax = self.ax_getter()
        if ax is None or event.inaxes != ax:
            if self._annot:
                self._annot.set_visible(False)
                self.canvas.draw_idle()
            return
        data = self.data_getter()
        if not data:
            return
        x = event.xdata
        if x is None:
            return
        # Trova il punto più vicino sull'asse x
        xs = [d.get("year_frac", d.get("month", 0) / 12) for d in data]
        idx = int(np.argmin(np.abs(np.array(xs) - x)))
        d   = data[idx]
        total     = d.get("total", 0)
        ask_bal   = d.get("ask_balance", 0)
        ops_bal   = d.get("ops_balance", 0)
        deposited = d.get("deposited", 0)
        month     = d.get("month", 0)
        txt = (f"Mese {month}  (anno {month/12:.1f})\n"
               f"Totale:   {self.fmt_fn(total)}\n"
               f"ASK:      {self.fmt_fn(ask_bal)}\n"
               f"Opspar:   {self.fmt_fn(ops_bal)}\n"
               f"Versato:  {self.fmt_fn(deposited)}\n"
               f"Guadagno: {self.fmt_fn(max(0, total - deposited))}")
        if self._annot is None:
            self._annot = ax.annotate(
                txt, xy=(xs[idx], total),
                xytext=(20, 20), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5",
                          fc=THEMES[_theme]["BG2"],
                          ec=BLUE, lw=1.2, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1),
                color=THEMES[_theme]["TEXT"],
                fontfamily=F_MONO, fontsize=8,
                zorder=10)
        else:
            self._annot.set_text(txt)
            self._annot.xy = (xs[idx], total)
            self._annot.set_visible(True)
            self._annot.get_bbox_patch().set(
                fc=THEMES[_theme]["BG2"],
                ec=BLUE)
            self._annot.set_color(THEMES[_theme]["TEXT"])
        self.canvas.draw_idle()

    def disconnect(self):
        self.canvas.mpl_disconnect(self._cid)
        self._annot = None

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("📈 Piano di Accumulo — ASK · Opspar")
        self.resizable(True, True)
        self.minsize(1400, 860)
        self.geometry("1560x920")

        self._theme = "dark"
        self._apply_theme()

        # ttk style
        self._style = ttk.Style(self)
        self._style.theme_use("clam")
        self._apply_ttk_style()

        # Variabili principali
        self.gross_salary = tk.StringVar(value="36000")
        self.income_tax   = tk.StringVar(value="37")
        self.monthly_exp  = tk.StringVar(value="12550")
        self.sav_return   = tk.StringVar(value="3")
        self.years        = tk.StringVar(value=str(DEFAULT_YEARS))
        self.inflation    = tk.StringVar(value="2.5")
        self.pal_annual   = tk.BooleanVar(value=True)
        self.ask_lim_grow = tk.BooleanVar(value=True)
        self.tab          = tk.StringVar(value="proiezione")
        self.currency     = tk.StringVar(value="DKK")
        self.dkk_eur_rate = _DKK_TO_EUR_FALLBACK

        self.portfolio      = []
        self.sel_ticker_var = tk.StringVar(value="CSPX.L")
        self.sel_pct_var    = tk.StringVar(value="60")
        self.etf_data_cache = {}
        self.loading        = False

        self.events       = []
        self.ev_month_var = tk.StringVar(value="13")
        self.ev_field_var = tk.StringVar(value="grossSalary")
        self.ev_value_var = tk.StringVar(value="40000")

        self._refresh_job = None
        self._last_data   = None
        self._last_res    = None
        self._last_params = None
        self._tooltip     = None
        self._current_ax  = None

        for v in (self.gross_salary, self.income_tax, self.monthly_exp,
                  self.sav_return, self.years, self.inflation):
            v.trace_add("write", lambda *_: self._schedule_refresh())
        self.pal_annual.trace_add("write",   lambda *_: self._schedule_refresh())
        self.ask_lim_grow.trace_add("write", lambda *_: self._schedule_refresh())

        self._build_ui()
        self._refresh()

    # ── Tema ──────────────────────────────────────────────────────────────────
    def _apply_theme(self):
        global _theme
        _theme = self._theme
        self.configure(bg=BG())

    def _apply_ttk_style(self):
        self._style.configure("TScrollbar",
            troughcolor=BG2(), background=BORDER(),
            arrowcolor=TEXT_DIM(), bordercolor=BG2(),
            lightcolor=BORDER(), darkcolor=BORDER())
        self._style.configure("TCombobox",
            fieldbackground=BG3(), background=BG3(),
            foreground=TEXT(), selectbackground=BG3(),
            selectforeground=TEXT(), bordercolor=BORDER(),
            arrowcolor=TEXT_DIM(), font=F_BODY)
        self._style.map("TCombobox",
            fieldbackground=[("readonly", BG3())],
            foreground=[("readonly", TEXT())])

    def _toggle_theme(self):
        self._theme = "light" if self._theme == "dark" else "dark"
        self._apply_theme()
        # Ricostruisci UI
        for w in self.winfo_children():
            w.destroy()
        self._tooltip = None
        self._apply_ttk_style()
        self._build_ui()
        self._refresh()

    # ── Utility ───────────────────────────────────────────────────────────────
    def _schedule_refresh(self):
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
        self._refresh_job = self.after(350, self._refresh)

    def _parse(self, var, fallback, lo=None, hi=None):
        try:
            v = float(str(var.get()).replace(",", ".").strip())
            if lo is not None and v < lo: return lo
            if hi is not None and v > hi: return hi
            return v
        except (ValueError, tk.TclError):
            return fallback

    def _params(self):
        return {
            "grossSalary":     self._parse(self.gross_salary, 36000, 0),
            "incomeTax":       self._parse(self.income_tax,   37, 0, 99) / 100,
            "monthlyExpenses": self._parse(self.monthly_exp,  12550, 0),
            "savingsReturn":   self._parse(self.sav_return,   3, 0, 50) / 100,
            "inflation":       self._parse(self.inflation,    2.5, 0, 30) / 100,
            "askTax":          ASK_TAX, "kapitalTax": KAPITAL_TAX,
            "years":           max(1, int(self._parse(self.years, DEFAULT_YEARS, 1, 30))),
            "portfolio":       self.portfolio,
            "palAnnual":       self.pal_annual.get(),
            "askLimitGrowth":  ASK_LIMIT_GROWTH if self.ask_lim_grow.get() else 0.0,
        }

    def _fmt(self, n: float) -> str:
        if self.currency.get() == "EUR":
            v = n * self.dkk_eur_rate
            return f"€{int(round(v)):,}".replace(",", ".")
        return fmt(n)

    def _darken(self, hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = int(r * 0.20); g = int(g * 0.20); b = int(b * 0.20)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr_inner = tk.Frame(self, bg=BG2(), padx=20, pady=10)
        hdr_inner.pack(fill="x")

        left_hdr = tk.Frame(hdr_inner, bg=BG2())
        left_hdr.pack(side="left", fill="y")
        tk.Label(left_hdr, text="📈  Piano di Accumulo",
                 bg=BG2(), fg=TEXT(), font=("Consolas", 16, "bold")).pack(anchor="w")
        tk.Label(left_hdr,
                 text="Aktiesparekonto (ASK) · Opsparingskonto · Danimarca",
                 bg=BG2(), fg=TEXT_DIM(), font=F_SMALL).pack(anchor="w")

        right_hdr = tk.Frame(hdr_inner, bg=BG2())
        right_hdr.pack(side="right", fill="y")

        # Badge fiscale
        badge = tk.Frame(right_hdr, bg=BG3(), padx=12, pady=6)
        badge.pack(side="left", padx=(0, 10))
        tk.Label(badge, text="ASK 17%  ·  Opspar 40%  ·  Limite 174.200 kr",
                 bg=BG3(), fg=YELLOW, font=F_SMALL).pack()

        # Tasso cambio
        self.rate_lbl = tk.Label(right_hdr,
            text=f"1 DKK = {self.dkk_eur_rate:.4f} EUR",
            bg=BG2(), fg=TEXT_DIM(), font=F_SMALL)
        self.rate_lbl.pack(side="left", padx=(0, 8))

        # Pulsante valuta
        self.currency_btn = tk.Button(
            right_hdr, text="💱  DKK → EUR",
            bg="#1a3a1a", fg=GREEN, relief="flat",
            font=("Consolas", 9, "bold"), cursor="hand2",
            padx=12, pady=6, bd=0,
            activebackground="#1f4a1f", activeforeground=GREEN,
            command=self._toggle_currency)
        self.currency_btn.pack(side="left", padx=(0, 6))

        # Pulsante tema
        theme_icon = "☀️" if self._theme == "dark" else "🌙"
        tk.Button(right_hdr,
                  text=f"{theme_icon}  Tema",
                  bg=BG3(), fg=TEXT_DIM(), relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2",
                  padx=10, pady=6, bd=0,
                  activebackground=BORDER(), activeforeground=TEXT(),
                  command=self._toggle_theme).pack(side="left", padx=(0, 6))

        # Export
        tk.Button(right_hdr, text="📥  Export",
                  bg=self._darken(BLUE), fg=BLUE, relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2",
                  padx=10, pady=6, bd=0,
                  activebackground=BLUE, activeforeground=BG(),
                  command=self._export_dialog).pack(side="left")

        # ── PanedWindow body ──────────────────────────────────────────────────
        self._paned = tk.PanedWindow(self, orient="horizontal",
                                      bg=BORDER(), sashwidth=6,
                                      sashrelief="flat", sashpad=2)
        self._paned.pack(fill="both", expand=True, padx=6, pady=6)

        # Pannello sinistro
        left_outer, self.left_frame, self._scroll_cv = make_scrollable(
            self._paned, bg_color=BG())
        self.left_frame.bind("<MouseWheel>",
            lambda e: self._scroll_cv.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._paned.add(left_outer, minsize=340, width=420)

        # Pannello destro
        right = tk.Frame(self._paned, bg=BG())
        self._paned.add(right, minsize=600)

        self._build_controls(self.left_frame)
        self._build_portfolio_panel(self.left_frame)
        self._build_events_panel(self.left_frame)
        self._build_strategy_panel(self.left_frame)
        self._build_charts(right)

    # ── Card / input helpers ──────────────────────────────────────────────────
    def _card(self, parent, title=None, color=None, pady_bottom=10):
        if color is None: color = BLUE
        sep = tk.Frame(parent, bg=BORDER(), height=1)
        sep.pack(fill="x", pady=(8, 0))
        outer = tk.Frame(parent, bg=BG2())
        outer.pack(fill="x", pady=(0, pady_bottom))
        inner = tk.Frame(outer, bg=BG2(), padx=16, pady=14)
        inner.pack(fill="both")
        if title:
            title_row = tk.Frame(inner, bg=BG2())
            title_row.pack(fill="x", pady=(0, 12))
            tk.Label(title_row, text=title, bg=BG2(), fg=color,
                     font=F_HEADER).pack(side="left")
            tk.Frame(title_row, bg=color, height=2).pack(
                side="left", fill="x", expand=True, padx=(8, 0), pady=(1, 0))
        return inner

    def _input_row(self, parent, label, var, unit="", color=None, tip=""):
        if color is None: color = BLUE
        row = tk.Frame(parent, bg=BG2())
        row.pack(fill="x", pady=(0, 10))
        lbl_row = tk.Frame(row, bg=BG2())
        lbl_row.pack(fill="x")
        tk.Label(lbl_row, text=label, bg=BG2(), fg=TEXT_MID(),
                 font=F_SMALL).pack(side="left")
        if tip:
            tk.Label(lbl_row, text=f"  {tip}", bg=BG2(), fg=TEXT_DIM(),
                     font=("Consolas", 7)).pack(side="left")
        entry_row = tk.Frame(row, bg=BG2())
        entry_row.pack(fill="x")
        entry_bg = tk.Frame(entry_row, bg=BORDER(), padx=1, pady=1)
        entry_bg.pack(side="left")
        e = tk.Entry(entry_bg, textvariable=var,
                     bg=BG3(), fg=color, insertbackground=color,
                     font=("Consolas", 11, "bold"),
                     relief="flat", bd=0, width=14)
        e.pack(ipady=6, padx=4)
        if unit:
            tk.Label(entry_row, text=unit, bg=BG2(), fg=TEXT_DIM(),
                     font=F_SMALL).pack(side="left", padx=(6, 0))
        return e

    def _btn(self, parent, text, color, command,
             side=None, fill=None, expand=False, pady=0):
        bg_dark = self._darken(color)
        b = tk.Button(parent, text=text, bg=bg_dark, fg=color,
                      relief="flat", font=("Consolas", 9, "bold"),
                      cursor="hand2", padx=12, pady=6, bd=0,
                      activebackground=color, activeforeground=BG(),
                      command=command)
        kw = {"pady": pady}
        if side:   kw["side"] = side
        if fill:   kw["fill"] = fill
        if expand: kw["expand"] = True
        b.pack(**kw)
        return b

    def _checkbox(self, parent, text, var, color=None):
        if color is None: color = TEXT_MID()
        tk.Checkbutton(parent, text=text, variable=var,
                       bg=BG2(), fg=color, selectcolor=BG3(),
                       activebackground=BG2(), activeforeground=color,
                       font=F_SMALL).pack(anchor="w", pady=2)

    # ── Parametri ─────────────────────────────────────────────────────────────
    def _build_controls(self, parent):
        card = self._card(parent, "📊  PARAMETRI BASE", BLUE)
        fields = [
            ("Stipendio lordo / mese",  self.gross_salary, "DKK",   BLUE,   "es. 36000"),
            ("Tassa sul reddito",       self.income_tax,   "%",     ORANGE, "es. 37"),
            ("Spese mensili totali",    self.monthly_exp,  "DKK",   GREEN,  "affitto+vitto"),
            ("Rendimento conto risp.", self.sav_return,   "% ann", BLUE,   "es. 3"),
            ("Orizzonte temporale",    self.years,        "anni",  ORANGE, "1–30"),
            ("Inflazione annua",       self.inflation,    "%",     RED,    "es. 2.5"),
        ]
        for lbl, var, unit, col, tip in fields:
            self._input_row(card, lbl, var, unit, col, tip)

        # Opzioni fiscali
        opt_frame = tk.Frame(card, bg=BG2())
        opt_frame.pack(fill="x", pady=(4, 0))
        self._checkbox(opt_frame, "PAL-skat annuale (reale, applicata a dicembre)",
                       self.pal_annual, GREEN)
        self._checkbox(opt_frame, "Aggiorna limite ASK ogni anno (+2.5%)",
                       self.ask_lim_grow, YELLOW)

        # KPI grid — rimosso FIRE
        kpi_grid = tk.Frame(parent, bg=BG())
        kpi_grid.pack(fill="x", pady=(0, 4))
        for c in range(2):
            kpi_grid.columnconfigure(c, weight=1)

        self.kpi_labels = {}
        kpis = [
            ("net",        "Netto / mese",      GREEN,  0, 0),
            ("sav",        "Risparmio / mese",  ORANGE, 0, 1),
            ("ask_dep",    "ASK versato",        PURPLE, 1, 0),
            ("tot",        "Patrimonio finale", BLUE,   1, 1),
            ("gain",       "Guadagno netto",    GREEN,  2, 0),
            ("ask_months", "Mesi → ASK pieno",  YELLOW, 2, 1),
            ("real_tot",   "Patrimonio reale",  CYAN,   3, 0),
        ]
        for key, lbl, col, r, c in kpis:
            cell = tk.Frame(kpi_grid, bg=BG2(), padx=12, pady=10)
            cell.grid(row=r, column=c, sticky="ew",
                      padx=(0 if c else 1, 1 if c else 0), pady=1)
            tk.Label(cell, text=lbl, bg=BG2(), fg=TEXT_DIM(), font=F_KPI_LBL).pack(anchor="w")
            v = tk.Label(cell, text="—", bg=BG2(), fg=col, font=F_KPI)
            v.pack(anchor="w")
            self.kpi_labels[key] = v

    # ── Portafoglio ───────────────────────────────────────────────────────────
    def _build_portfolio_panel(self, parent):
        card = self._card(parent, "📦  PORTAFOGLIO ETF  (ASK)", PURPLE)

        tk.Label(card,
                 text="ETF lista ABIS/SKAT · lagerbeskatning 17%\n"
                      "% = quota del risparmio mensile allocata in ASK.",
                 bg=BG2(), fg=TEXT_DIM(), font=F_SMALL,
                 wraplength=360, justify="left").pack(anchor="w", pady=(0, 10))

        # Selezione ETF
        sel_lbl_row = tk.Frame(card, bg=BG2())
        sel_lbl_row.pack(fill="x", pady=(0, 4))
        tk.Label(sel_lbl_row, text="ETF", bg=BG2(), fg=TEXT_MID(), font=F_SMALL).pack(side="left")

        ticker_names = [f"{t}  —  {n}" for t, _, n, _ in DEFAULT_ETF_LIST]
        combo_bg = tk.Frame(card, bg=BORDER(), pady=1, padx=1)
        combo_bg.pack(fill="x", pady=(0, 8))
        self._etf_combo = ttk.Combobox(combo_bg,
                                        textvariable=self.sel_ticker_var,
                                        values=ticker_names,
                                        state="readonly",
                                        font=("Consolas", 9))
        self._etf_combo.pack(fill="x")
        self._etf_combo.bind("<<ComboboxSelected>>", self._on_etf_selected)

        # Preview
        self.etf_preview_lbl = tk.Label(card,
                                         text="Clicca 'Aggiorna dati' per statistiche reali.",
                                         bg=BG2(), fg=TEXT_DIM(), font=F_SMALL,
                                         wraplength=360, justify="left")
        self.etf_preview_lbl.pack(anchor="w", pady=(0, 8))

        # % risparmio
        pct_row = tk.Frame(card, bg=BG2())
        pct_row.pack(fill="x", pady=(0, 8))
        tk.Label(pct_row, text="% risparmio  →  ASK",
                 bg=BG2(), fg=TEXT_MID(), font=F_SMALL).pack(side="left")
        pct_bg = tk.Frame(pct_row, bg=BORDER(), padx=1, pady=1)
        pct_bg.pack(side="right")
        tk.Entry(pct_bg, textvariable=self.sel_pct_var,
                 width=6, bg=BG3(), fg=PURPLE, insertbackground=PURPLE,
                 font=("Consolas", 11, "bold"), relief="flat", bd=0).pack(ipady=5, padx=4)

        # Pulsanti
        btn_row = tk.Frame(card, bg=BG2())
        btn_row.pack(fill="x", pady=(0, 6))
        self._btn(btn_row, "➕  Aggiungi ETF",  BLUE,  self._add_etf,         side="left", expand=True)
        tk.Frame(btn_row, bg=BG(), width=6).pack(side="left")
        self._btn(btn_row, "🔄  Aggiorna dati", GREEN, self._fetch_all_async, side="left", expand=True)

        # Status
        self.load_status_lbl = tk.Label(card, text="", bg=BG2(), fg=CYAN, font=F_SMALL)
        self.load_status_lbl.pack(anchor="w", pady=(0, 6))

        # Lista portafoglio scrollabile
        list_outer = tk.Frame(card, bg=BG3(), pady=1, padx=1)
        list_outer.pack(fill="x", pady=(0, 8))
        self.portfolio_list_frame = tk.Frame(list_outer, bg=BG3())
        self.portfolio_list_frame.pack(fill="x")

        # Allocazione totale
        self.alloc_lbl = tk.Label(card, text="Allocazione: 0% ASK + 100% Opspar",
                                   bg=BG2(), fg=ORANGE,
                                   font=("Consolas", 9, "bold"))
        self.alloc_lbl.pack(anchor="w", pady=(4, 8))

        # Raccomandazioni
        self._btn(card, "⭐  Raccomandazioni SKAT (top ETF ora)",
                  PURPLE, self._show_recommendations, fill="x", pady=4)

        self._refresh_portfolio_list()

    def _on_etf_selected(self, event=None):
        raw    = self.sel_ticker_var.get()
        ticker = raw.split("  —  ")[0].strip() if "  —  " in raw else raw.strip()
        if ticker in self.etf_data_cache:
            self._update_etf_preview(ticker, self.etf_data_cache[ticker])

    def _update_etf_preview(self, ticker, d):
        status_icon = "✅" if d.get("status") == "ok" else "⚠️"
        self.etf_preview_lbl.config(
            text=(f"{status_icon}  CAGR {d['annual_return']*100:.1f}%  ·  "
                  f"Vol {d['volatility']*100:.1f}%  ·  "
                  f"Sharpe {d['sharpe']:.2f}  ·  "
                  f"MaxDD {d['max_drawdown']*100:.1f}%  ·  "
                  f"YTD {d['ytd']*100:.1f}%"),
            fg=GREEN if d.get("status") == "ok" else YELLOW)

    def _add_etf(self):
        raw    = self.sel_ticker_var.get()
        ticker = raw.split("  —  ")[0].strip() if "  —  " in raw else raw.strip()
        try:
            pct = float(self.sel_pct_var.get().replace(",", "."))
        except ValueError:
            pct = 60.0
        pct = max(1.0, min(100.0, pct))
        current_total = sum(e["pct"] for e in self.portfolio)
        if current_total + pct > 100.0:
            pct = max(0.0, 100.0 - current_total)
            if pct < 1.0:
                self.load_status_lbl.config(text="⚠️ Allocazione totale già al 100%!", fg=RED)
                return
        name = ticker
        for t, _, n, typ in DEFAULT_ETF_LIST:
            if t == ticker:
                name = f"{n} ({typ})"; break
        d = self.etf_data_cache.get(ticker, _fallback_etf(ticker))
        self.portfolio.append({
            "ticker": ticker, "name": name, "pct": pct / 100,
            "annual_return": d["annual_return"],
            "volatility":    d.get("volatility", 0.18),
            "sharpe":        d.get("sharpe", 0.4),
            "status":        d.get("status", "fallback"),
        })
        self._refresh_portfolio_list()
        self.load_status_lbl.config(text=f"✅ {ticker} aggiunto ({pct:.0f}%)", fg=GREEN)
        self._refresh()

    def _remove_etf(self, idx):
        self.portfolio.pop(idx)
        self._refresh_portfolio_list()
        self._refresh()

    def _refresh_portfolio_list(self):
        for w in self.portfolio_list_frame.winfo_children():
            w.destroy()
        if not self.portfolio:
            tk.Label(self.portfolio_list_frame,
                     text="  Nessun ETF — il 100% va in Opspar.",
                     bg=BG3(), fg=TEXT_DIM(), font=F_SMALL, pady=8).pack(anchor="w")
            self.alloc_lbl.config(text="Allocazione: 0% ASK + 100% Opspar")
            return
        hdr = tk.Frame(self.portfolio_list_frame, bg=BORDER())
        hdr.pack(fill="x")
        for col, w in zip(["Ticker", "%", "CAGR", "Vol", "Sharpe", ""], [9,5,7,6,7,3]):
            tk.Label(hdr, text=col, bg=BORDER(), fg=TEXT_DIM(),
                     font=("Consolas", 7, "bold"),
                     width=w, anchor="w", padx=4, pady=4).pack(side="left")
        total_ask_pct = 0.0
        for i, etf in enumerate(self.portfolio):
            total_ask_pct += etf["pct"]
            d   = self.etf_data_cache.get(etf["ticker"], etf)
            row = tk.Frame(self.portfolio_list_frame,
                           bg=BG2() if i % 2 == 0 else BG3())
            row.pack(fill="x")
            sc = GREEN if etf.get("status") == "ok" else YELLOW
            for val, col, ww in [
                (etf["ticker"],                      BLUE,     9),
                (f"{etf['pct']*100:.0f}%",            PURPLE,   5),
                (f"{d['annual_return']*100:.1f}%",    sc,       7),
                (f"{d.get('volatility',0)*100:.0f}%", TEXT_DIM(), 6),
                (f"{d.get('sharpe',0):.2f}",          CYAN,     7),
            ]:
                tk.Label(row, text=val, bg=row.cget("bg"), fg=col,
                         font=F_SMALL, width=ww,
                         anchor="w", padx=4, pady=5).pack(side="left")
            tk.Button(row, text="✕", bg=row.cget("bg"), fg=RED,
                      relief="flat", font=("Consolas", 7),
                      cursor="hand2", width=2, pady=2,
                      command=lambda i=i: self._remove_etf(i)).pack(side="left")
        ops_pct = max(0.0, 1.0 - total_ask_pct)
        self.alloc_lbl.config(
            text=(f"ASK: {total_ask_pct*100:.0f}%  +  Opspar: {ops_pct*100:.0f}%  "
                  f"{'✅' if abs(total_ask_pct + ops_pct - 1) < 0.01 else '⚠️'}"),
            fg=GREEN if total_ask_pct > 0 else ORANGE)

    def _fetch_all_async(self):
        if self.loading: return
        self.loading = True
        self.load_status_lbl.config(text="⏳ Download Yahoo Finance...", fg=CYAN)
        def _worker():
            tickers = [t for t, _, _, _ in DEFAULT_ETF_LIST]
            for i, ticker in enumerate(tickers):
                self.after(0, lambda t=ticker, i=i:
                    self.load_status_lbl.config(
                        text=f"⏳ [{i+1}/{len(tickers)}] {t}...", fg=CYAN))
                d = fetch_etf_data(ticker)
                self.etf_data_cache[ticker] = d
                for etf in self.portfolio:
                    if etf["ticker"] == ticker:
                        etf["annual_return"] = d["annual_return"]
                        etf["volatility"]    = d.get("volatility", 0.18)
                        etf["sharpe"]        = d.get("sharpe", 0.4)
                        etf["status"]        = d.get("status", "fallback")
            self.after(0, self._on_fetch_complete)
        threading.Thread(target=_worker, daemon=True).start()

    def _on_fetch_complete(self):
        self.loading = False
        ok_count = sum(1 for d in self.etf_data_cache.values() if d.get("status") == "ok")
        self.load_status_lbl.config(
            text=f"✅ Dati aggiornati: {ok_count}/{len(DEFAULT_ETF_LIST)} ETF OK", fg=GREEN)
        self._refresh_portfolio_list()
        self._refresh()
        raw    = self.sel_ticker_var.get()
        ticker = raw.split("  —  ")[0].strip() if "  —  " in raw else raw.strip()
        if ticker in self.etf_data_cache:
            self._update_etf_preview(ticker, self.etf_data_cache[ticker])

    def _show_recommendations(self):
        if not self.etf_data_cache:
            self.load_status_lbl.config(text="⚠️ Prima clicca 'Aggiorna dati'!", fg=YELLOW)
            return
        win = tk.Toplevel(self)
        win.title("⭐ Raccomandazioni ETF — Lista ABIS/SKAT")
        win.configure(bg=BG())
        win.geometry("780x480")
        tk.Label(win, text="⭐  RANKING ETF — Lista ABIS/SKAT",
                 bg=BG(), fg=BLUE, font=F_TITLE, padx=20, pady=14).pack(anchor="w")
        tk.Label(win,
                 text="Score = 40% Sharpe + 30% CAGR + 20% (1+MaxDD) + 10% (1-Vol)",
                 bg=BG(), fg=TEXT_DIM(), font=F_SMALL, padx=20).pack(anchor="w")
        ranked = []
        for ticker, d in self.etf_data_cache.items():
            name = ticker
            for t, _, n, typ in DEFAULT_ETF_LIST:
                if t == ticker: name = f"{n} ({typ})"; break
            ranked.append((score_etf(d), ticker, name, d))
        ranked.sort(reverse=True)
        frame = tk.Frame(win, bg=BG(), padx=20)
        frame.pack(fill="both", expand=True)
        cols   = ["#", "Ticker", "Nome", "CAGR", "Vol", "Sharpe", "MaxDD", "YTD", "Score"]
        widths = [3, 10, 28, 7, 6, 8, 8, 7, 7]
        hdr = tk.Frame(frame, bg=BORDER())
        hdr.pack(fill="x", pady=(10, 2))
        for col, w in zip(cols, widths):
            tk.Label(hdr, text=col, bg=BORDER(), fg=TEXT_DIM(),
                     font=("Consolas", 8, "bold"),
                     width=w, anchor="w", padx=4, pady=4).pack(side="left", padx=1)
        for rank, (sc, ticker, name, d) in enumerate(ranked, 1):
            color = [GREEN, CYAN, BLUE, TEXT(), TEXT_DIM()][min(rank-1, 4)]
            row   = tk.Frame(frame, bg=BG2() if rank % 2 == 0 else BG3())
            row.pack(fill="x", pady=1)
            for (val, col2), w in zip([
                (f"#{rank}", color), (ticker, BLUE), (name[:26], TEXT()),
                (f"{d['annual_return']*100:.1f}%", GREEN),
                (f"{d.get('volatility',0)*100:.0f}%", TEXT_DIM()),
                (f"{d.get('sharpe',0):.2f}", CYAN),
                (f"{d.get('max_drawdown',0)*100:.1f}%", RED),
                (f"{d.get('ytd',0)*100:.1f}%", YELLOW),
                (f"{sc:.3f}", color),
            ], widths):
                tk.Label(row, text=val, bg=row.cget("bg"), fg=col2,
                         font=F_SMALL, width=w, anchor="w", padx=4, pady=5).pack(side="left")
            tk.Button(row, text="➕", bg=row.cget("bg"), fg=GREEN,
                      relief="flat", font=F_SMALL, cursor="hand2",
                      command=lambda t=ticker: (self.sel_ticker_var.set(t),
                                                self._add_etf())).pack(side="right", padx=4)

    # ── Events Panel ──────────────────────────────────────────────────────────
    def _build_events_panel(self, parent):
        card = self._card(parent, "📅  VARIAZIONI NEL TEMPO", YELLOW)
        tk.Label(card,
                 text="Cambia stipendio, spese o tasse a partire da un mese specifico.",
                 bg=BG2(), fg=TEXT_DIM(), font=F_SMALL,
                 wraplength=360, justify="left").pack(anchor="w", pady=(0, 10))
        field_labels = {
            "grossSalary":     "💼 Stipendio lordo (DKK/mese)",
            "incomeTax":       "🏛️ Tassa reddito (es. 37 = 37%)",
            "monthlyExpenses": "🛒 Spese mensili (DKK)",
            "savingsReturn":   "🏦 Rendimento risparmio (es. 3 = 3%)",
        }
        r1 = tk.Frame(card, bg=BG2())
        r1.pack(fill="x", pady=(0, 6))
        tk.Label(r1, text="Campo:", bg=BG2(), fg=TEXT_MID(), font=F_SMALL).pack(side="left")
        combo_bg = tk.Frame(r1, bg=BORDER(), pady=1, padx=1)
        combo_bg.pack(side="right")
        ttk.Combobox(combo_bg, textvariable=self.ev_field_var,
                     values=list(field_labels.keys()),
                     state="readonly", font=F_SMALL, width=22).pack()
        self.ev_field_lbl = tk.Label(card, text="", bg=BG2(), fg=YELLOW, font=F_SMALL)
        self.ev_field_lbl.pack(anchor="w", pady=(0, 6))
        def on_fc(*_):
            self.ev_field_lbl.config(text=field_labels.get(self.ev_field_var.get(), ""))
        self.ev_field_var.trace_add("write", on_fc); on_fc()
        r2 = tk.Frame(card, bg=BG2())
        r2.pack(fill="x", pady=(0, 8))
        for lbl, var, col in [("Mese:", self.ev_month_var, YELLOW),
                               ("  Valore:", self.ev_value_var, YELLOW)]:
            tk.Label(r2, text=lbl, bg=BG2(), fg=TEXT_MID(), font=F_SMALL).pack(side="left")
            eb = tk.Frame(r2, bg=BORDER(), padx=1, pady=1)
            eb.pack(side="left", padx=(4, 0))
            w = 5 if "Mese" in lbl else 10
            tk.Entry(eb, textvariable=var, width=w, bg=BG3(), fg=col,
                     insertbackground=col, font=("Consolas", 10),
                     relief="flat", bd=0).pack(ipady=5, padx=4)
        self._btn(card, "➕  Aggiungi Evento", GREEN, self._add_event,
                  fill="x", pady=(0, 6))
        self.events_list_frame = tk.Frame(card, bg=BG3())
        self.events_list_frame.pack(fill="x")
        self._refresh_events_list()

    def _add_event(self):
        try:
            field = self.ev_field_var.get()
            month = int(self.ev_month_var.get())
            value = float(self.ev_value_var.get().replace(",", "."))
            max_m = max(1, int(self._parse(self.years, DEFAULT_YEARS, 1, 30))) * 12
            if month < 1 or month > max_m: return
            internal = value
            if field in ("incomeTax", "savingsReturn") and value > 1:
                internal = value / 100
            labels = {
                "grossSalary":     f"Stipendio → {fmt(value)}",
                "incomeTax":       f"Tassa → {internal*100:.0f}%",
                "monthlyExpenses": f"Spese → {fmt(value)}",
                "savingsReturn":   f"Risp → {internal*100:.1f}%",
            }
            self.events.append({
                "month": month, "field": field, "value": internal,
                "label": f"M{month}: {labels.get(field, field)}"
            })
            self.events.sort(key=lambda e: e["month"])
            self._refresh_events_list()
            self._refresh()
        except (ValueError, tk.TclError):
            pass

    def _remove_event(self, idx):
        self.events.pop(idx)
        self._refresh_events_list()
        self._refresh()

    def _refresh_events_list(self):
        for w in self.events_list_frame.winfo_children():
            w.destroy()
        if not self.events:
            tk.Label(self.events_list_frame, text="  Nessun evento.",
                     bg=BG3(), fg=TEXT_DIM(), font=F_SMALL, pady=6).pack(anchor="w")
            return
        for i, ev in enumerate(self.events):
            row = tk.Frame(self.events_list_frame, bg=BG2() if i%2==0 else BG3())
            row.pack(fill="x")
            tk.Label(row, text=ev["label"], bg=row.cget("bg"),
                     fg=YELLOW, font=F_SMALL, pady=5, padx=6).pack(side="left")
            tk.Button(row, text="✕", bg=row.cget("bg"), fg=RED,
                      relief="flat", font=F_SMALL, cursor="hand2",
                      command=lambda i=i: self._remove_event(i)).pack(side="right", padx=4)

    # ── Strategy Panel ────────────────────────────────────────────────────────
    def _build_strategy_panel(self, parent):
        card = self._card(parent, "🎯  CONFRONTO STRATEGIE", CYAN)
        tk.Label(card, text="Confronto automatico strategie principali:",
                 bg=BG2(), fg=TEXT_DIM(), font=F_SMALL).pack(anchor="w", pady=(0, 8))
        self.strategy_frame = tk.Frame(card, bg=BG2())
        self.strategy_frame.pack(fill="x")

    def _update_strategy_panel(self, params):
        for w in self.strategy_frame.winfo_children():
            w.destroy()
        strategies = []
        if self.portfolio:
            strategies.append(("Portafoglio corrente", params["portfolio"]))
        cspx = self.etf_data_cache.get("CSPX.L", _fallback_etf("CSPX.L"))
        strategies.append(("100% S&P500 (CSPX.L)", [
            {"ticker": "CSPX.L", "name": "S&P500", "pct": 1.0,
             "annual_return": cspx["annual_return"],
             "volatility": cspx.get("volatility", 0.175)}
        ]))
        strategies.append(("100% Opspar", []))
        widths = [20, 14, 13]
        hdr = tk.Frame(self.strategy_frame, bg=BORDER())
        hdr.pack(fill="x")
        for col, w in zip(["Strategia", "Patrimonio", "Guadagno"], widths):
            tk.Label(hdr, text=col, bg=BORDER(), fg=TEXT_DIM(),
                     font=("Consolas", 8, "bold"),
                     width=w, anchor="w", padx=4, pady=4).pack(side="left", padx=1)
        for i, (name, portfolio) in enumerate(strategies):
            test_params = {**params, "portfolio": portfolio}
            res   = simulate_monthly(test_params, self.events)
            color = GREEN if i == 0 and self.portfolio else TEXT()
            row   = tk.Frame(self.strategy_frame, bg=BG2() if i%2==0 else BG3())
            row.pack(fill="x", pady=1)
            for val, w in zip([name, self._fmt(res["final_total"]),
                                self._fmt(res["final_net_gain"])], widths):
                tk.Label(row, text=val, bg=row.cget("bg"), fg=color,
                         font=F_SMALL, width=w, anchor="w",
                         padx=4, pady=5).pack(side="left")

    # ── Charts ────────────────────────────────────────────────────────────────
    def _build_charts(self, parent):
        tab_bar = tk.Frame(parent, bg=BG2())
        tab_bar.pack(fill="x")
        self.tab_btns = {}
        # Tab FIRE rimosso
        tabs = [
            ("proiezione",   "📈  PROIEZIONE"),
            ("portafoglio",  "📊  PORTAFOGLIO"),
            ("strategie",    "🎯  STRATEGIE"),
            ("montecarlo",   "🎲  RISCHIO MC"),
            ("composizione", "🍰  COMPOSIZIONE"),
            ("ask_fill",     "🟣  ASK FILL"),
        ]
        for t, lbl in tabs:
            b = tk.Button(tab_bar, text=lbl, bg=BG2(), fg=TEXT_DIM(),
                          relief="flat", font=("Consolas", 9, "bold"),
                          cursor="hand2", padx=14, pady=10, bd=0,
                          activebackground=BG3(), activeforeground=BLUE,
                          command=lambda x=t: self._set_tab(x))
            b.pack(side="left")
            self.tab_btns[t] = b
        self._highlight_tab(self.tab.get())

        chart_frame = tk.Frame(parent, bg=BG2())
        chart_frame.pack(fill="both", expand=True)

        self.fig = Figure(facecolor=BG2(), tight_layout=True)
        self.fig.set_tight_layout({"pad": 1.5, "rect": [0, 0, 1, 0.97]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(parent, bg=BG2())
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.config(bg=BG2())
        for w in toolbar.winfo_children():
            try: w.config(bg=BG2(), fg=TEXT_DIM(), relief="flat")
            except Exception: pass
        toolbar.update()

        self.insight_lbl = tk.Label(parent, text="",
                                     bg=BG3(), fg=TEXT_MID(), font=F_SMALL,
                                     wraplength=900, justify="left",
                                     padx=16, pady=10, anchor="w")
        self.insight_lbl.pack(fill="x")
        self._tooltip = None

    # ── Valuta ────────────────────────────────────────────────────────────────
    def _toggle_currency(self):
        if self.currency.get() == "DKK":
            self.currency_btn.config(text="⏳ ...", state="disabled")
            def _fetch():
                rate = fetch_dkk_eur_rate()
                self.after(0, lambda: self._apply_currency("EUR", rate))
            threading.Thread(target=_fetch, daemon=True).start()
        else:
            self._apply_currency("DKK", self.dkk_eur_rate)

    def _apply_currency(self, new_currency: str, rate: float):
        self.dkk_eur_rate = rate
        self.currency.set(new_currency)
        is_eur = new_currency == "EUR"
        self.currency_btn.config(
            text="💱  EUR → DKK" if is_eur else "💱  DKK → EUR",
            bg=self._darken(ORANGE) if is_eur else self._darken(GREEN),
            fg=ORANGE if is_eur else GREEN,
            state="normal")
        self.rate_lbl.config(
            text=(f"1 DKK = {rate:.4f} EUR  ✅" if is_eur
                  else f"1 DKK = {rate:.4f} EUR"))
        self._refresh()

    # ── Export ────────────────────────────────────────────────────────────────
    def _export_dialog(self):
        if not self._last_data:
            messagebox.showwarning("Export", "Esegui prima una simulazione.")
            return
        win = tk.Toplevel(self)
        win.title("📥  Export")
        win.configure(bg=BG())
        win.geometry("360x220")
        win.resizable(False, False)
        tk.Label(win, text="📥  Esporta simulazione",
                 bg=BG(), fg=TEXT(), font=F_HEADER, padx=20, pady=16).pack(anchor="w")
        self._btn(win, "📊  Esporta Excel (.xlsx)",
                  GREEN, self._export_excel, fill="x", pady=4)
        tk.Frame(win, bg=BG(), height=4).pack()
        self._btn(win, "📄  Esporta PDF (grafico corrente)",
                  BLUE, lambda: self._export_pdf(win), fill="x", pady=4)
        tk.Frame(win, bg=BG(), height=4).pack()
        self._btn(win, "🖼️  Esporta PNG (grafico corrente)",
                  PURPLE, lambda: self._export_png(win), fill="x", pady=4)

    def _export_excel(self):
        if not PD_AVAILABLE:
            messagebox.showerror("Export Excel",
                "pandas non installato.\nEsegui: pip install pandas openpyxl")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile="piano_accumulo.xlsx")
        if not path: return
        try:
            rows = []
            for d in self._last_data:
                row = {
                    "Mese":           d["month"],
                    "Anno":           round(d["year_frac"], 2),
                    "ASK Saldo":      round(d["ask_balance"], 0),
                    "Opspar Saldo":   round(d["ops_balance"], 0),
                    "Totale":         round(d["total"], 0),
                    "Versato":        round(d["deposited"], 0),
                    "Guadagno Netto": round(d["net_gain"], 0),
                    "ASK Versato":    round(d["ask_deposited"], 0),
                    "Totale Reale":   round(d.get("real_total", d["total"]), 0),
                    "FIRE Raggiunto": d.get("fire_reached", False),
                    "Netto Mensile":  round(d["net_monthly"], 0),
                    "Risparmio Mese": round(d["sav_monthly"], 0),
                }
                # ETF singoli
                for t, v in d.get("etf_balances", {}).items():
                    row[f"ETF {t}"] = round(v, 0)
                rows.append(row)

            df = pd.DataFrame(rows)
            params = self._last_params
            meta = pd.DataFrame([{
                "Parametro": k, "Valore": v
            } for k, v in {
                "Stipendio lordo":   params["grossSalary"],
                "Aliquota":          f"{params['incomeTax']*100:.1f}%",
                "Spese mensili":     params["monthlyExpenses"],
                "Rendimento risp.":  f"{params['savingsReturn']*100:.1f}%",
                "Inflazione":        f"{params.get('inflation',0)*100:.1f}%",
                "Anni":              params["years"],
                "PAL annuale":       params.get("palAnnual", True),
                "ETF":               ", ".join(e["ticker"] for e in self.portfolio),
                "Esportato il":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }.items()])

            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer,   sheet_name="Simulazione", index=False)
                meta.to_excel(writer, sheet_name="Parametri",   index=False)
            messagebox.showinfo("Export Excel", f"✅ Salvato:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Excel", f"Errore:\n{e}")

    def _export_pdf(self, parent=None):
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile="piano_accumulo.pdf")
        if not path: return
        try:
            self.fig.savefig(path, facecolor=self.fig.get_facecolor(),
                             dpi=150, bbox_inches="tight")
            messagebox.showinfo("Export PDF", f"✅ Salvato:\n{path}")
        except Exception as e:
            messagebox.showerror("Export PDF", f"Errore:\n{e}")

    def _export_png(self, parent=None):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile="piano_accumulo.png")
        if not path: return
        try:
            self.fig.savefig(path, facecolor=self.fig.get_facecolor(),
                             dpi=200, bbox_inches="tight")
            messagebox.showinfo("Export PNG", f"✅ Salvato:\n{path}")
        except Exception as e:
            messagebox.showerror("Export PNG", f"Errore:\n{e}")

    # ── Core ──────────────────────────────────────────────────────────────────
    def _set_tab(self, t):
        self.tab.set(t)
        self._highlight_tab(t)
        self._draw_chart()

    def _highlight_tab(self, t):
        for name, btn in self.tab_btns.items():
            btn.config(bg=BG3() if name == t else BG2(),
                       fg=BLUE  if name == t else TEXT_DIM())

    def _refresh(self):
        try:
            params = self._params()
            res    = simulate_monthly(params, self.events)
            data   = res["data"]
            sav_m  = res["sav_monthly"]

            self.kpi_labels["net"].config(text=self._fmt(res["net_monthly"]))
            self.kpi_labels["sav"].config(
                text=self._fmt(sav_m) if sav_m > 0 else "⚠️ negativo",
                fg=ORANGE if sav_m > 0 else RED)
            self.kpi_labels["ask_dep"].config(
                text=self._fmt(res["ask_deposited"]),
                fg=GREEN if res["ask_full"] else PURPLE)
            self.kpi_labels["tot"].config(text=self._fmt(res["final_total"]))
            self.kpi_labels["gain"].config(text=self._fmt(res["final_net_gain"]))
            self.kpi_labels["real_tot"].config(text=self._fmt(res["final_real"]))

            # Mesi → ASK pieno
            total_ask_pct = sum(e["pct"] for e in self.portfolio)
            if sav_m > 0 and total_ask_pct > 0:
                monthly_ask = sav_m * total_ask_pct
                mtf = ASK_DEPOSIT_LIMIT / monthly_ask
                self.kpi_labels["ask_months"].config(
                    text=f"{mtf:.0f} mesi  ({mtf/12:.1f} anni)")
            elif total_ask_pct == 0:
                self.kpi_labels["ask_months"].config(text="∞  (nessun ETF)")
            else:
                self.kpi_labels["ask_months"].config(text="—")

            # Insight — rimosso fire_str
            ask_fill = min(100, res["ask_deposited"] / ASK_DEPOSIT_LIMIT * 100)
            ask_fill_month = next(
                (d["month"] for d in data if d["ask_deposited"] >= ASK_DEPOSIT_LIMIT), None)
            portfolio_str = (", ".join(f"{e['ticker']} {e['pct']*100:.0f}%"
                                       for e in self.portfolio)
                             if self.portfolio else "nessun ETF")
            infl_str = f"  Inflazione: {params['inflation']*100:.1f}%."
            if sav_m <= 0:
                insight = "⚡  ATTENZIONE: spese > reddito netto. Impossibile risparmiare."
            else:
                fill_txt = (f"ASK pieno al mese {ask_fill_month} ({ask_fill_month/12:.1f} anni)."
                            if ask_fill_month else
                            f"ASK: {ask_fill:.1f}% riempito in {params['years']} anni.")
                insight = (f"⚡  Portafoglio: {portfolio_str}.  {fill_txt}"
                           f"  Patrimonio: {self._fmt(res['final_total'])}"
                           f"  (reale: {self._fmt(res['final_real'])})"
                           f"{infl_str}")
            self.insight_lbl.config(text=insight)

            self._last_data   = data
            self._last_res    = res
            self._last_params = params

            self._update_strategy_panel(params)
            self._draw_chart()
        except Exception:
            import traceback; traceback.print_exc()

    # ── Stile assi ────────────────────────────────────────────────────────────
    def _style_ax(self, ax, monthly=False, cur="DKK"):
        ax.set_facecolor(BG())
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER())
        ax.tick_params(colors=TEXT_DIM(), labelsize=8, which="both")
        ax.xaxis.label.set_color(TEXT_DIM())
        ax.yaxis.label.set_color(TEXT_DIM())
        ax.grid(True, color=BORDER(), linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        rate   = self.dkk_eur_rate if cur == "EUR" else 1.0
        prefix = "€" if cur == "EUR" else ""
        suffix = "" if cur == "EUR" else " kr"
        def _fmt_y(v, _):
            v2 = v * rate
            if abs(v2) >= 1_000_000:
                return f"{prefix}{v2/1_000_000:.1f}M{suffix}"
            return f"{prefix}{int(v2/1000)}k{suffix}"
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_y))
        ax.set_xlabel("Mese" if monthly else "Anno",
                      color=TEXT_DIM(), fontsize=8, labelpad=6)

    def _add_event_vlines(self, ax, ymax, monthly=False):
        evs = (sorted(set(ev["month"] for ev in self.events)) if monthly
               else sorted(set(ev["month"] // 12 for ev in self.events)))
        for x in evs:
            ax.axvline(x=x, color=YELLOW, lw=1, linestyle=":", alpha=0.7)
            ax.text(x + 0.3, ymax * 0.91, "Δ", color=YELLOW,
                    fontsize=7, fontfamily=F_MONO)

    def _monthly_to_yearly(self, data):
        yearly = [data[0]]
        for m in range(12, len(data), 12):
            yearly.append(data[m])
        if data[-1] not in yearly:
            yearly.append(data[-1])
        return yearly

    def _legend(self, ax):
        leg = ax.legend(fontsize=8, facecolor=BG2(), edgecolor=BORDER(),
                        labelcolor=TEXT(), loc="upper left",
                        framealpha=0.9, borderpad=0.8)
        for line in leg.get_lines():
            line.set_linewidth(2.5)

    # ── Draw ──────────────────────────────────────────────────────────────────
    def _draw_chart(self):
        if not hasattr(self, "_last_data") or self._last_data is None:
            return

        # Disconnetti tooltip precedente
        if self._tooltip:
            self._tooltip.disconnect()
            self._tooltip = None

        data   = self._last_data
        params = self._last_params
        tab    = self.tab.get()
        months = params["years"] * 12
        xs_m   = [d["month"] for d in data]
        cur    = self.currency.get()

        self.fig.clf()
        self.fig.set_facecolor(BG2())
        ax = self.fig.add_subplot(111)
        self._current_ax = ax
        self._style_ax(ax, monthly=(tab in ("ask_fill", "montecarlo")), cur=cur)

        etf_colors = [PURPLE, BLUE, CYAN, GREEN, ORANGE, PINK, YELLOW, TEAL]

        if tab == "proiezione":
            ax.set_title("PROIEZIONE PATRIMONIO — annuale",
                         color=TEXT_MID(), fontsize=10, loc="left",
                         fontfamily=F_MONO, pad=10)
            yd    = self._monthly_to_yearly(data)
            xs    = [d["month"] / 12 for d in yd]
            ask_v = [d["ask_balance"]  for d in yd]
            ops_v = [d["ops_balance"]  for d in yd]
            tot_v = [d["total"]        for d in yd]
            dep_v = [d["deposited"]    for d in yd]
            real_v = [d.get("real_total", d["total"]) for d in yd]
            ax.fill_between(xs, ask_v, alpha=0.12, color=PURPLE)
            ax.fill_between(xs, ops_v, alpha=0.10, color=BLUE)
            ax.plot(xs, ask_v,  color=PURPLE, lw=2.5, label="ASK (ETF)",
                    marker="o", markersize=4, markevery=max(1, len(xs)//8))
            ax.plot(xs, ops_v,  color=BLUE,   lw=2.5, label="Opsparingskonto",
                    marker="s", markersize=4, markevery=max(1, len(xs)//8))
            ax.plot(xs, tot_v,  color=GREEN,  lw=3,   linestyle="--", label="Totale",
                    marker="D", markersize=4, markevery=max(1, len(xs)//8))
            ax.plot(xs, real_v, color=CYAN,   lw=2,   linestyle="-.",
                    label="Totale reale (inflaz.)")
            ax.plot(xs, dep_v,  color=ORANGE, lw=1.5, linestyle=":", label="Versato")
            ax.axhline(y=ASK_DEPOSIT_LIMIT, color=YELLOW, lw=1.2,
                       linestyle="--", alpha=0.7, label="Limite ASK")
            ymax = max(tot_v) if tot_v else 1
            self._add_event_vlines(ax, ymax)
            if len(xs) > 1:
                ax.annotate(f"  {self._fmt(tot_v[-1])}",
                            xy=(xs[-1], tot_v[-1]),
                            color=GREEN, fontsize=8, fontfamily=F_MONO, va="bottom")
            # Tooltip solo per proiezione
            self._tooltip = ChartTooltip(
                self.fig, self.canvas,
                lambda: self._current_ax, lambda: self._last_data, self._fmt)

        elif tab == "portafoglio":
            ax.set_title("COMPOSIZIONE PORTAFOGLIO ETF NEL TEMPO",
                         color=TEXT_MID(), fontsize=10, loc="left",
                         fontfamily=F_MONO, pad=10)
            if not self.portfolio:
                ax.text(0.5, 0.5, "Aggiungi ETF al portafoglio →",
                        transform=ax.transAxes, color=TEXT_DIM(),
                        ha="center", va="center", fontsize=13, fontfamily=F_MONO)
            else:
                yd     = self._monthly_to_yearly(data)
                xs     = [d["month"] / 12 for d in yd]
                bottom = np.zeros(len(xs))
                for i, etf in enumerate(self.portfolio):
                    col  = etf_colors[i % len(etf_colors)]
                    vals = np.array([d["etf_balances"].get(etf["ticker"], 0) for d in yd])
                    ax.bar(xs, vals, bottom=bottom, color=col,
                           width=0.65, alpha=0.85, label=etf["ticker"])
                    bottom += vals
                ops_v = np.array([d["ops_balance"] for d in yd])
                ax.bar(xs, ops_v, bottom=bottom, color=TEXT_DIM(),
                       width=0.65, alpha=0.45, label="Opspar")

        elif tab == "strategie":
            ax.set_title("CONFRONTO STRATEGIE — Patrimonio vs Scala Allocazione ASK",
                         color=TEXT_MID(), fontsize=10, loc="left",
                         fontfamily=F_MONO, pad=10)
            scales = list(range(0, 101, 5))
            totals = []
            for sc in scales:
                if self.portfolio:
                    total_pct = sum(e["pct"] for e in self.portfolio)
                    factor    = (sc / 100) / total_pct if total_pct > 0 else 0
                    scaled    = [{**e, "pct": min(1.0, e["pct"] * factor)}
                                 for e in self.portfolio]
                else:
                    scaled = []
                r = simulate_monthly({**params, "portfolio": scaled}, self.events)
                totals.append(r["final_total"])
            ax.fill_between(scales, totals, alpha=0.15, color=GREEN)
            ax.plot(scales, totals, color=GREEN, lw=2.5, marker="o",
                    markersize=5, label="Patrimonio finale")
            best_i = int(np.argmax(totals))
            ax.axvline(x=scales[best_i], color=YELLOW, lw=1.5,
                       linestyle="--", alpha=0.8)
            ax.annotate(f"  Ottimo: {scales[best_i]}%\n  {self._fmt(totals[best_i])}",
                        xy=(scales[best_i], totals[best_i]),
                        color=YELLOW, fontsize=8, fontfamily=F_MONO)
            ax.set_xlabel("Scala allocazione ASK (%)",
                          color=TEXT_DIM(), fontsize=8, labelpad=6)

        elif tab == "montecarlo":
            ax.set_title("ANALISI DI RISCHIO — Monte Carlo  300 simulazioni",
                         color=TEXT_MID(), fontsize=10, loc="left",
                         fontfamily=F_MONO, pad=10)
            mc     = simulate_monte_carlo_monthly(params, self.events, 300)
            xs_arr = np.arange(months + 1)
            ax.fill_between(xs_arr, mc["p5"],  mc["p95"], color=BLUE, alpha=0.08, label="P5–P95")
            ax.fill_between(xs_arr, mc["p25"], mc["p75"], color=BLUE, alpha=0.18, label="P25–P75")
            ax.plot(xs_arr, mc["p50"], color=GREEN,  lw=2.5, label="Mediana (P50)")
            ax.plot(xs_arr, mc["p5"],  color=RED,    lw=1.2, linestyle="--", label="P5 (pessimista)")
            ax.plot(xs_arr, mc["p95"], color=PURPLE, lw=1.2, linestyle="--", label="P95 (ottimista)")
            ax.plot(xs_m, [d["total"] for d in data], color=YELLOW,
                    lw=1.8, linestyle=":", label="Scenario base")
            ymax = float(mc["p95"].max()) if len(mc["p95"]) > 0 else 1
            self._add_event_vlines(ax, ymax, monthly=True)

        elif tab == "composizione":
            ax.set_title("VERSATO  vs  GUADAGNO NETTO — per anno",
                         color=TEXT_MID(), fontsize=10, loc="left",
                         fontfamily=F_MONO, pad=10)
            yd   = self._monthly_to_yearly(data)
            xs   = [d["month"] / 12 for d in yd]
            dep  = [d["deposited"]        for d in yd]
            gain = [max(0, d["net_gain"]) for d in yd]
            ax.bar(xs, dep,  color=BORDER(), label="Versato", width=0.65, zorder=2)
            ax.bar(xs, gain, bottom=dep, color=GREEN,
                   label="Guadagno netto", width=0.65, alpha=0.85, zorder=2)
            # Etichette %
            for x, d_val, g_val in zip(xs, dep, gain):
                tot = d_val + g_val
                if tot > 0 and g_val > 0:
                    ax.text(x, tot + tot * 0.02, f"{g_val/tot*100:.0f}%",
                            ha="center", va="bottom",
                            color=GREEN, fontsize=7, fontfamily=F_MONO)

        elif tab == "ask_fill":
            ax.set_title("RIEMPIMENTO ASK — Capitale versato  (mensile)",
                         color=TEXT_MID(), fontsize=10, loc="left",
                         fontfamily=F_MONO, pad=10)
            ask_dep_v = [d["ask_deposited"] for d in data]
            ask_bal_v = [d["ask_balance"]   for d in data]
            ops_bal_v = [d["ops_balance"]   for d in data]
            # Limite ASK dinamico (se crescita attiva)
            ask_lim_v = [d.get("ask_limit", ASK_DEPOSIT_LIMIT) for d in data]
            ax.fill_between(xs_m, ask_dep_v, alpha=0.18, color=PURPLE)
            ax.plot(xs_m, ask_dep_v, color=PURPLE, lw=2.5, label="ASK versato")
            ax.plot(xs_m, ask_bal_v, color=GREEN,  lw=2,   linestyle="--",
                    label="ASK saldo (netto tasse)")
            ax.plot(xs_m, ops_bal_v, color=BLUE,   lw=2,   linestyle="-.",
                    label="Opspar saldo")
            ax.plot(xs_m, ask_lim_v, color=YELLOW, lw=1.5, linestyle="--",
                    alpha=0.8, label="Limite ASK (dinamico)")
            ymax = max(max(ask_bal_v), max(ops_bal_v), ASK_DEPOSIT_LIMIT)
            self._add_event_vlines(ax, ymax, monthly=True)

        self._legend(ax)
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = App()
    app.mainloop()