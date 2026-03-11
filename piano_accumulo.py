import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import datetime
import io
import os

# Librerie opzionali (con fallback graceful)
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import pandas as pd
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

try:
    import requests
    REQ_AVAILABLE = True
except ImportError:
    REQ_AVAILABLE = False

# ── Costanti fiscali danesi ────────────────────────────────────────────────────
ASK_DEPOSIT_LIMIT = 174200
ASK_TAX           = 0.17
KAPITAL_TAX       = 0.40
DEFAULT_YEARS     = 4

# ── Colori tema scuro ──────────────────────────────────────────────────────────
BG       = "#0d1117"
BG2      = "#161b22"
BORDER   = "#21262d"
TEXT     = "#c9d1d9"
TEXT_DIM = "#8b949e"
BLUE     = "#58a6ff"
ORANGE   = "#f0883e"
GREEN    = "#3fb950"
PURPLE   = "#a371f7"
RED      = "#ff7b72"
YELLOW   = "#e3b341"
CYAN     = "#39d353"

# ── ETF predefiniti dalla lista ABIS/SKAT con ticker Yahoo Finance ─────────────
# Fonte: https://skat.dk (ABIS-listen 2020-2025)
# Solo ETF Acc (accumulazione) — lagerbeskatning 17% in ASK
DEFAULT_ETF_LIST = [
    # ticker Yahoo,  ISIN,                nome breve,              tipo
    ("CSPX.L",  "IE00B5BMR087", "iShares Core S&P 500 Acc",        "S&P500"),
    ("VUSA.AS", "IE00B3XXRP09", "Vanguard S&P 500 Acc",            "S&P500"),
    ("SWDA.L",  "IE00B4L5Y983", "iShares Core MSCI World Acc",     "World"),
    ("EUNL.DE", "IE00B4L5Y983", "iShares Core MSCI World Acc (DE)","World"),
    ("VWCE.DE", "IE00BK5BQT80", "Vanguard FTSE All-World Acc",     "All-World"),
    ("IUSQ.DE", "IE00B4L5Y983", "iShares MSCI World Acc",          "World"),
    ("SPPW.DE", "IE00BYML9W36", "SPDR MSCI World Acc",             "World"),
    ("MEUD.PA", "LU1681044480", "Amundi MSCI Europe Acc",          "Europe"),
    ("QDVE.DE", "IE00B3XXRP09", "iShares S&P 500 IT Acc",          "Sector-IT"),
    ("IQQQ.DE", "IE00B3RBWM25", "iShares Nasdaq 100 Acc",          "Nasdaq"),
    ("VFEM.AS", "IE00B3VVMM84", "Vanguard FTSE EM Acc",            "EM"),
    ("AGGH.L",  "IE00BDBRDM35", "iShares Global Agg Bond Acc",     "Bond"),
]

# Cache locale dei dati ETF (evita download ripetuti)
_etf_cache = {}   # {ticker: {"annual_return": float, "volatility": float, ...}}


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH DATI ETF
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_etf_data(ticker: str, years_back: int = 5) -> dict:
    """
    Scarica storico prezzi da Yahoo Finance e calcola:
    - Rendimento annualizzato (CAGR)
    - Volatilità annualizzata
    - Sharpe ratio (risk-free ~3%)
    - Max drawdown
    - Rendimento YTD
    """
    if ticker in _etf_cache:
        return _etf_cache[ticker]

    if not YF_AVAILABLE:
        return _fallback_etf(ticker)

    try:
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=years_back * 365)

        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(start=start.isoformat(),
                                  end=end.isoformat(),
                                  interval="1mo")   # mensile

        if hist.empty or len(hist) < 6:
            return _fallback_etf(ticker)

        prices = hist["Close"].dropna().values

        # CAGR
        n_years = len(prices) / 12
        cagr    = (prices[-1] / prices[0]) ** (1 / n_years) - 1

        # Rendimenti mensili
        monthly_returns = np.diff(prices) / prices[:-1]

        # Volatilità annuale
        vol_annual = np.std(monthly_returns) * np.sqrt(12)

        # Sharpe ratio (risk-free 3%)
        rf_monthly = 0.03 / 12
        sharpe = ((np.mean(monthly_returns) - rf_monthly)
                  / np.std(monthly_returns) * np.sqrt(12)
                  if np.std(monthly_returns) > 0 else 0)

        # Max drawdown
        cumulative = np.cumprod(1 + monthly_returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = float(np.min(drawdown))

        # YTD
        ytd = 0.0
        try:
            hist_ytd = ticker_obj.history(
                start=f"{end.year}-01-01", end=end.isoformat())
            if not hist_ytd.empty:
                p = hist_ytd["Close"].dropna().values
                ytd = (p[-1] / p[0]) - 1 if len(p) > 1 else 0.0
        except Exception:
            pass

        # Prezzo corrente e nome
        info  = {}
        try:
            info = ticker_obj.fast_info
        except Exception:
            pass

        result = {
            "ticker":        ticker,
            "annual_return": round(cagr, 4),
            "volatility":    round(vol_annual, 4),
            "sharpe":        round(sharpe, 3),
            "max_drawdown":  round(max_dd, 4),
            "ytd":           round(ytd, 4),
            "prices":        prices.tolist(),
            "monthly_ret":   monthly_returns.tolist(),
            "last_price":    float(prices[-1]),
            "n_months":      len(prices),
            "status":        "ok",
        }
        _etf_cache[ticker] = result
        return result

    except Exception as e:
        return _fallback_etf(ticker, str(e))


def _fallback_etf(ticker: str, error: str = "no data") -> dict:
    """Dati fallback se Yahoo Finance non disponibile."""
    defaults = {
        "CSPX.L":  {"annual_return": 0.112, "volatility": 0.175, "sharpe": 0.65},
        "VUSA.AS": {"annual_return": 0.112, "volatility": 0.175, "sharpe": 0.65},
        "SWDA.L":  {"annual_return": 0.095, "volatility": 0.155, "sharpe": 0.55},
        "VWCE.DE": {"annual_return": 0.090, "volatility": 0.150, "sharpe": 0.52},
        "IQQQ.DE": {"annual_return": 0.150, "volatility": 0.240, "sharpe": 0.60},
        "AGGH.L":  {"annual_return": 0.020, "volatility": 0.060, "sharpe": 0.10},
    }
    base = defaults.get(ticker, {"annual_return": 0.08, "volatility": 0.18, "sharpe": 0.40})
    return {
        **base,
        "ticker":       ticker,
        "max_drawdown": -0.35,
        "ytd":          0.0,
        "prices":       [],
        "monthly_ret":  [],
        "last_price":   0.0,
        "n_months":     0,
        "status":       f"fallback ({error})",
    }


def score_etf(data: dict) -> float:
    """
    Score composito per ranking ETF:
    - 40% Sharpe ratio (rischio/rendimento)
    - 30% CAGR annualizzato
    - 20% Max drawdown (penalizzato)
    - 10% Volatilità (penalizzata)
    """
    sharpe  = min(max(data.get("sharpe", 0), -1), 3)
    cagr    = min(max(data.get("annual_return", 0), -0.5), 0.5)
    mdd     = max(data.get("max_drawdown", -1), -1)   # negativo
    vol     = min(data.get("volatility", 1), 1)

    return (0.40 * sharpe / 3
            + 0.30 * cagr / 0.5
            + 0.20 * (1 + mdd)       # mdd in [-1,0] → (1+mdd) in [0,1]
            + 0.10 * (1 - vol))


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULAZIONE MENSILE (multi-ETF)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_monthly(params, events=None):
    """
    Simulazione MENSILE con versamenti dinamici e portafoglio multi-ETF.

    params["portfolio"]: [{"ticker": str, "pct": float, "annual_return": float, "volatility": float}, ...]
    La percentuale (pct) è relativa al risparmio mensile disponibile.
    La somma delle pct degli ETF deve essere ≤ 1.0
    Il restante va in Opsparingskonto.
    """
    months        = params["years"] * 12
    cur_gross     = params["grossSalary"]
    cur_itax      = params["incomeTax"]
    cur_expenses  = params["monthlyExpenses"]
    cur_sav_r     = params["savingsReturn"]
    portfolio     = params.get("portfolio", [])  # lista ETF
    ask_pct_total = sum(e["pct"] for e in portfolio)  # % totale → ASK/ETF

    monthly_overrides = {}
    if events:
        for ev in events:
            m = ev["month"]
            if m not in monthly_overrides:
                monthly_overrides[m] = {}
            monthly_overrides[m][ev["field"]] = ev["value"]

    # Stato per ogni ETF nel portafoglio
    etf_states = []
    for etf in portfolio:
        etf_states.append({
            "ticker":    etf["ticker"],
            "pct":       etf["pct"],
            "r_annual":  etf["annual_return"],
            "deposited": 0.0,
            "balance":   0.0,
            "ask_dep":   0.0,   # capitale versato in ASK per questo ETF
        })

    ops_balance     = 0.0
    total_deposited = 0.0
    ask_deposited   = 0.0   # totale globale versato in ASK

    data = []
    net0 = cur_gross * (1 - cur_itax)
    data.append({
        "month": 0, "year_frac": 0.0,
        "etf_balances":  {e["ticker"]: 0.0 for e in portfolio},
        "ask_balance":   0.0,
        "ops_balance":   0.0,
        "total":         0.0,
        "deposited":     0.0,
        "net_gain":      0.0,
        "net_monthly":   net0,
        "sav_monthly":   net0 - cur_expenses,
        "ask_deposited": 0.0,
        "ask_full":      False,
    })

    for m in range(1, months + 1):
        if m in monthly_overrides:
            ov = monthly_overrides[m]
            cur_gross    = ov.get("grossSalary",     cur_gross)
            cur_itax     = ov.get("incomeTax",       cur_itax)
            cur_expenses = ov.get("monthlyExpenses", cur_expenses)
            cur_sav_r    = ov.get("savingsReturn",   cur_sav_r)
            # Aggiornamento rendimento ETF dinamico
            for es in etf_states:
                key = f"etfReturn_{es['ticker']}"
                if key in ov:
                    es["r_annual"] = ov[key]

        net_monthly = cur_gross * (1 - cur_itax)
        sav_monthly = net_monthly - cur_expenses
        ask_full    = ask_deposited >= ASK_DEPOSIT_LIMIT

        # ── Versamento mensile agli ETF ───────────────────────────────────
        if sav_monthly > 0:
            remaining_sav = sav_monthly

            for es in etf_states:
                if ask_full:
                    # ASK pieno: tutto va in Opspar (gestito sotto)
                    dep = 0.0
                else:
                    desired   = sav_monthly * es["pct"]
                    ask_room  = max(0.0, ASK_DEPOSIT_LIMIT - ask_deposited)
                    dep       = min(desired, ask_room)
                    ask_deposited  += dep
                    es["ask_dep"]  += dep

                es["deposited"] += dep
                es["balance"]   += dep
                remaining_sav   -= dep

            # Il resto va in Opspar (incluso overflow ASK)
            actual_ops = max(0.0, remaining_sav)
            ops_balance     += actual_ops
            total_deposited += sav_monthly

        # ── Interessi mensili per ogni ETF (lagerbeskatning 17%) ──────────
        ask_balance_total = 0.0
        etf_balances_snap = {}
        for es in etf_states:
            r_monthly  = es["r_annual"] / 12
            gain       = es["balance"] * r_monthly
            es["balance"] += gain * (1 - ASK_TAX)
            etf_balances_snap[es["ticker"]] = es["balance"]
            ask_balance_total += es["balance"]

        # ── Interessi Opspar (kapitalindkomst 40%) ────────────────────────
        ops_gain    = ops_balance * (cur_sav_r / 12)
        ops_balance += ops_gain * (1 - KAPITAL_TAX)

        total    = ask_balance_total + ops_balance
        net_gain = total - total_deposited

        data.append({
            "month":         m,
            "year_frac":     m / 12,
            "etf_balances":  etf_balances_snap,
            "ask_balance":   ask_balance_total,
            "ops_balance":   ops_balance,
            "total":         total,
            "deposited":     total_deposited,
            "net_gain":      net_gain,
            "net_monthly":   net_monthly,
            "sav_monthly":   sav_monthly,
            "ask_deposited": ask_deposited,
            "ask_full":      ask_full,
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
        "final_deposited": data[-1]["deposited"],
        "final_net_gain":  data[-1]["net_gain"],
        "etf_states":      etf_states,
    }


def find_optimal_monthly(params, events=None):
    """
    Con portafoglio multi-ETF: cerca la % ottimale di allocazione
    tra ASK (ETF) e Opspar variando la proporzione globale.
    """
    if not params.get("portfolio"):
        # Fallback: nessun ETF → tutto Opspar
        return {"pct": 0.0, "total": simulate_monthly(params, events)["final_total"]}

    best = {"pct": 0.0, "total": 0.0}
    base_portfolio = params["portfolio"]
    total_etf_pct  = sum(e["pct"] for e in base_portfolio)

    if total_etf_pct == 0:
        return {"pct": 0.0, "total": simulate_monthly(params, events)["final_total"]}

    # Scala il portafoglio da 0% a 100% mantenendo proporzioni interne
    for scale in range(0, 101, 5):
        factor = scale / 100 / total_etf_pct if total_etf_pct > 0 else 0
        scaled = [{**e, "pct": e["pct"] * factor} for e in base_portfolio]
        test_params = {**params, "portfolio": scaled}
        res = simulate_monthly(test_params, events)
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
            # Rendimento stocastico per ogni ETF
            for etf in portfolio:
                mu    = etf["annual_return"]
                sigma = etf.get("volatility", 0.18)
                r     = float(np.random.lognormal(
                    mean=np.log(1 + mu) - 0.5 * sigma**2,
                    sigma=sigma
                )) - 1
                for mo in range(y * 12 + 1, (y + 1) * 12 + 1):
                    mc_events.append({
                        "month": mo,
                        "field": f"etfReturn_{etf['ticker']}",
                        "value": r
                    })

            # Rendimento stocastico risparmio
            r_sav = max(0.0, float(np.random.normal(sav_mean, sav_sigma)))
            for mo in range(y * 12 + 1, (y + 1) * 12 + 1):
                mc_events.append({"month": mo, "field": "savingsReturn", "value": r_sav})

        res = simulate_monthly(params, mc_events)
        for mo in range(months + 1):
            all_totals[i, mo] = res["data"][mo]["total"]

    return {
        "p5":  np.percentile(all_totals, 5,  axis=0),
        "p10": np.percentile(all_totals, 10, axis=0),
        "p25": np.percentile(all_totals, 25, axis=0),
        "p50": np.percentile(all_totals, 50, axis=0),
        "p75": np.percentile(all_totals, 75, axis=0),
        "p90": np.percentile(all_totals, 90, axis=0),
        "p95": np.percentile(all_totals, 95, axis=0),
    }


def fmt(n):
    return f"{int(round(n)):,} kr".replace(",", ".")

def fmt_pct(v):
    return f"{v*100:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
# APPLICAZIONE
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("📈 Piano di Accumulo")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(1300, 800)

        # ── Parametri base (StringVar per input testuale) ───────────────────
        self.gross_salary = tk.StringVar(value="36000")
        self.income_tax   = tk.StringVar(value="37")
        self.monthly_exp  = tk.StringVar(value="12550")
        self.sav_return   = tk.StringVar(value="3")
        self.years        = tk.StringVar(value=str(DEFAULT_YEARS))
        self.tab          = tk.StringVar(value="proiezione")

        # ── Portafoglio ETF ─────────────────────────────────────────────────
        # [{"ticker": str, "name": str, "pct": float,
        #   "annual_return": float, "volatility": float,
        #   "sharpe": float, "status": str}]
        self.portfolio = []

        # ── Stato ETF selector ──────────────────────────────────────────────
        self.sel_ticker_var = tk.StringVar(value="CSPX.L")
        self.sel_pct_var    = tk.StringVar(value="60")
        self.etf_data_cache = {}   # ticker → dati fetchati
        self.loading        = False

        # ── Eventi mensili ──────────────────────────────────────────────────
        self.events       = []
        self.ev_month_var = tk.StringVar(value="13")
        self.ev_field_var = tk.StringVar(value="grossSalary")
        self.ev_value_var = tk.StringVar(value="40000")

        # Debounce
        self._refresh_job = None

        for v in (self.gross_salary, self.income_tax, self.monthly_exp,
                  self.sav_return, self.years):
            v.trace_add("write", lambda *_: self._schedule_refresh())

        self._build_ui()
        self._refresh()

    # ── Debounce ──────────────────────────────────────────────────────────────
    def _schedule_refresh(self):
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
        self._refresh_job = self.after(400, self._refresh)

    # ── Parse sicuro ──────────────────────────────────────────────────────────
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
            "askTax":          ASK_TAX,
            "kapitalTax":      KAPITAL_TAX,
            "years":           max(1, int(self._parse(self.years, DEFAULT_YEARS, 1, 30))),
            "portfolio":       self.portfolio,
        }

    # ── UI Builder ────────────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(18, 6))
        tk.Label(hdr, text="📈 Piano di Accumulo — Aktiesparekonto (ASK) · Opsparingskonto (Opspar)",
                 bg=BG, fg="#f0f6fc", font=("Courier", 15, "bold")).pack(anchor="w")
        # self.hdr_sub = tk.Label(hdr,
        #          text="S&P500 ETF Acc · Danimarca · Limite ASK: 174.200 kr  |  17% ASK · 40% Opspar",
        #          bg=BG, fg=TEXT_DIM, font=("Courier", 9))
        self.hdr_sub.pack(anchor="w")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 18))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Sinistra scrollable
        left_outer = tk.Frame(body, bg=BG, width=355)
        left_outer.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        left_outer.pack_propagate(False)

        self._scroll_cv = tk.Canvas(left_outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(left_outer, orient="vertical",
                           command=self._scroll_cv.yview)
        self.left_frame = tk.Frame(self._scroll_cv, bg=BG)
        self.left_frame.bind("<Configure>",
            lambda e: self._scroll_cv.configure(
                scrollregion=self._scroll_cv.bbox("all")))
        self._scroll_cv.create_window((0, 0), window=self.left_frame, anchor="nw")
        self._scroll_cv.configure(yscrollcommand=sb.set)
        self._scroll_cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._scroll_cv.bind("<MouseWheel>",
            lambda e: self._scroll_cv.yview_scroll(
                int(-1*(e.delta/120)), "units"))

        self._build_controls(self.left_frame)
        self._build_portfolio_panel(self.left_frame)
        self._build_events_panel(self.left_frame)
        self._build_strategy_panel(self.left_frame)

        # Destra grafici
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_charts(right)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _card(self, parent, title=None, color=BLUE):
        outer = tk.Frame(parent, bg=BORDER)
        outer.pack(fill="x", pady=(0, 10))
        inner = tk.Frame(outer, bg=BG2, padx=14, pady=12)
        inner.pack(fill="both", padx=1, pady=1)
        if title:
            tk.Label(inner, text=title.upper(), bg=BG2, fg=color,
                     font=("Courier", 9, "bold")).pack(anchor="w", pady=(0, 10))
        return inner

    def _input_row(self, parent, label, var, unit="", color=BLUE, tip=""):
        row = tk.Frame(parent, bg=BG2)
        row.pack(fill="x", pady=(0, 7))
        top = tk.Frame(row, bg=BG2)
        top.pack(fill="x")
        tk.Label(top, text=label, bg=BG2, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(side="left")
        if tip:
            tk.Label(top, text=f"  {tip}", bg=BG2, fg="#444d56",
                     font=("Courier", 7)).pack(side="left")
        ef = tk.Frame(row, bg=BG2)
        ef.pack(fill="x")
        e = tk.Entry(ef, textvariable=var,
                     bg="#0d1117", fg=color,
                     insertbackground=color,
                     font=("Courier", 11, "bold"),
                     relief="flat", bd=0,
                     highlightthickness=1,
                     highlightbackground=BORDER,
                     highlightcolor=color,
                     width=16)
        e.pack(side="left", ipady=5, padx=(0, 4))
        if unit:
            tk.Label(ef, text=unit, bg=BG2, fg=TEXT_DIM,
                     font=("Courier", 9)).pack(side="left")
        return e

    # ── Pannello parametri base ───────────────────────────────────────────────
    def _build_controls(self, parent):
        card = self._card(parent, "📊 Parametri Base", BLUE)
        fields = [
            ("Stipendio lordo / mese",  self.gross_salary, "DKK",   BLUE,   "es. 36000"),
            ("Tassa sul reddito",       self.income_tax,   "%",     ORANGE, "es. 37"),
            ("Spese mensili totali",    self.monthly_exp,  "DKK",   GREEN,  "affitto+vitto"),
            ("Rendimento conto risp.", self.sav_return,   "% ann", BLUE,   "es. 3"),
            ("Orizzonte temporale",    self.years,        "anni",  ORANGE, "1–30"),
        ]
        for lbl, var, unit, col, tip in fields:
            self._input_row(card, lbl, var, unit, col, tip)

        # KPI grid
        kpi_outer = tk.Frame(parent, bg=BG)
        kpi_outer.pack(fill="x")
        kpi_outer.columnconfigure(0, weight=1)
        kpi_outer.columnconfigure(1, weight=1)
        self.kpi_labels = {}
        kpis = [
            ("net",        "Netto / mese",        GREEN,  0, 0),
            ("sav",        "Risparmio / mese",     ORANGE, 0, 1),
            ("ask_dep",    "ASK versato",          PURPLE, 1, 0),
            ("tot",        "Patrimonio finale",    BLUE,   1, 1),
            ("gain",       "Guadagno netto",       GREEN,  2, 0),
            ("ask_months", "Mesi → ASK pieno",     YELLOW, 2, 1),
        ]
        for key, lbl, col, r, c in kpis:
            cell = tk.Frame(kpi_outer, bg=BORDER)
            cell.grid(row=r, column=c, sticky="ew",
                      padx=(0, 3 if c == 0 else 0), pady=(0, 3))
            inn = tk.Frame(cell, bg=BG2, padx=10, pady=8)
            inn.pack(fill="both", padx=1, pady=1)
            tk.Label(inn, text=lbl, bg=BG2, fg=TEXT_DIM,
                     font=("Courier", 8)).pack(anchor="w")
            v = tk.Label(inn, text="—", bg=BG2, fg=col,
                         font=("Courier", 10, "bold"))
            v.pack(anchor="w")
            self.kpi_labels[key] = v

    # ── Pannello portafoglio ETF ──────────────────────────────────────────────
    def _build_portfolio_panel(self, parent):
        card = self._card(parent, "📦 Portafoglio ETF (ASK)", PURPLE)

        # Info
        tk.Label(card,
                 text="ETF dalla lista ABIS/SKAT · tassazione lagerbeskatning 17%\n"
                      "La % indica la quota del risparmio mensile allocata in ASK.",
                 bg=BG2, fg=TEXT_DIM, font=("Courier", 8),
                 wraplength=310, justify="left").pack(anchor="w", pady=(0, 8))

        # ── Selettore ETF ──────────────────────────────────────────────────
        sel_frame = tk.Frame(card, bg=BG2)
        sel_frame.pack(fill="x", pady=(0, 4))

        tk.Label(sel_frame, text="ETF:", bg=BG2, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(side="left")

        ticker_names = [f"{t}  —  {n}" for t, _, n, _ in DEFAULT_ETF_LIST]
        self._etf_combo = ttk.Combobox(sel_frame,
                                       textvariable=self.sel_ticker_var,
                                       values=ticker_names,
                                       state="readonly", width=26,
                                       font=("Courier", 8))
        self._etf_combo.pack(side="left", padx=(4, 0))
        self._etf_combo.bind("<<ComboboxSelected>>", self._on_etf_selected)

        # Percentuale + pulsante
        pct_frame = tk.Frame(card, bg=BG2)
        pct_frame.pack(fill="x", pady=(6, 0))
        tk.Label(pct_frame, text="% risparmio:", bg=BG2, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(side="left")
        tk.Entry(pct_frame, textvariable=self.sel_pct_var,
                 width=6, bg="#0d1117", fg=PURPLE,
                 insertbackground=PURPLE,
                 font=("Courier", 11, "bold"),
                 relief="flat", highlightthickness=1,
                 highlightbackground=BORDER,
                 highlightcolor=PURPLE).pack(side="left", padx=(4, 8), ipady=4)
        tk.Label(pct_frame, text="%  →  ASK", bg=BG2, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(side="left")

        # Preview dati ETF
        self.etf_preview_lbl = tk.Label(card, text="",
                                         bg=BG2, fg=TEXT_DIM,
                                         font=("Courier", 8),
                                         wraplength=310, justify="left")
        self.etf_preview_lbl.pack(anchor="w", pady=(4, 0))

        # Pulsanti
        btn_row = tk.Frame(card, bg=BG2)
        btn_row.pack(fill="x", pady=(6, 0))

        tk.Button(btn_row, text="➕ Aggiungi ETF",
                  bg="#1f2d3d", fg=BLUE, relief="flat",
                  font=("Courier", 9, "bold"), cursor="hand2", pady=4,
                  command=self._add_etf).pack(side="left", fill="x",
                                              expand=True, padx=(0, 4))

        tk.Button(btn_row, text="🔄 Aggiorna dati",
                  bg="#1a3a2a", fg=GREEN, relief="flat",
                  font=("Courier", 9), cursor="hand2", pady=4,
                  command=self._fetch_all_async).pack(side="left", fill="x",
                                                      expand=True)

        # Status caricamento
        self.load_status_lbl = tk.Label(card, text="", bg=BG2, fg=CYAN,
                                         font=("Courier", 8))
        self.load_status_lbl.pack(anchor="w", pady=(4, 0))

        # Lista portafoglio
        self.portfolio_list_frame = tk.Frame(card, bg=BG2)
        self.portfolio_list_frame.pack(fill="x", pady=(8, 0))

        # Totale allocazione
        self.alloc_lbl = tk.Label(card, text="Allocazione: 0% ASK + 100% Opspar",
                                   bg=BG2, fg=ORANGE, font=("Courier", 9, "bold"))
        self.alloc_lbl.pack(anchor="w", pady=(6, 0))

        # Pulsante raccomandazioni
        tk.Button(card, text="⭐ Raccomandazioni SKAT (top ETF ora)",
                  bg="#2d1f3d", fg=PURPLE, relief="flat",
                  font=("Courier", 9, "bold"), cursor="hand2", pady=5,
                  command=self._show_recommendations).pack(fill="x", pady=(8, 0))

        self._refresh_portfolio_list()

    def _on_etf_selected(self, event=None):
        """Mostra preview dati ETF selezionato."""
        raw    = self.sel_ticker_var.get()
        ticker = raw.split("  —  ")[0].strip() if "  —  " in raw else raw.strip()

        if ticker in self.etf_data_cache:
            self._update_etf_preview(ticker, self.etf_data_cache[ticker])
        else:
            self.etf_preview_lbl.config(
                text="⏳ Clicca 'Aggiorna dati' per scaricare le statistiche reali.")

    def _update_etf_preview(self, ticker, d):
        status_icon = "✅" if d.get("status") == "ok" else "⚠️"
        self.etf_preview_lbl.config(
            text=f"{status_icon}  CAGR {d['annual_return']*100:.1f}%  |  "
                 f"Vol {d['volatility']*100:.1f}%  |  "
                 f"Sharpe {d['sharpe']:.2f}  |  "
                 f"MaxDD {d['max_drawdown']*100:.1f}%  |  "
                 f"YTD {d['ytd']*100:.1f}%",
            fg=GREEN if d.get("status") == "ok" else YELLOW)

    def _add_etf(self):
        raw    = self.sel_ticker_var.get()
        ticker = raw.split("  —  ")[0].strip() if "  —  " in raw else raw.strip()

        try:
            pct = float(self.sel_pct_var.get().replace(",", "."))
        except ValueError:
            pct = 60.0

        pct = max(1.0, min(100.0, pct))

        # Controlla che totale non superi 100%
        current_total = sum(e["pct"] for e in self.portfolio)
        if current_total + pct > 100.0:
            pct = max(0.0, 100.0 - current_total)
            if pct < 1.0:
                self.load_status_lbl.config(
                    text="⚠️ Allocazione totale già al 100%!", fg=RED)
                return

        # Trova nome ETF
        name = ticker
        for t, _, n, typ in DEFAULT_ETF_LIST:
            if t == ticker:
                name = f"{n} ({typ})"
                break

        # Dati ETF (cache o fallback)
        d = self.etf_data_cache.get(ticker, _fallback_etf(ticker))

        self.portfolio.append({
            "ticker":       ticker,
            "name":         name,
            "pct":          pct / 100,
            "annual_return": d["annual_return"],
            "volatility":   d.get("volatility", 0.18),
            "sharpe":       d.get("sharpe", 0.4),
            "status":       d.get("status", "fallback"),
        })

        self._refresh_portfolio_list()
        self.load_status_lbl.config(
            text=f"✅ {ticker} aggiunto ({pct:.0f}%)", fg=GREEN)
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
                     text="Nessun ETF aggiunto. Il 100% va in Opspar.",
                     bg=BG2, fg=TEXT_DIM, font=("Courier", 8)).pack(anchor="w")
            self.alloc_lbl.config(text="Allocazione: 0% ASK + 100% Opspar")
            return

        # Header
        hdr = tk.Frame(self.portfolio_list_frame, bg=BG2)
        hdr.pack(fill="x", pady=(8, 2))
        for col, w in zip(["Ticker", "%", "CAGR", "Vol", "Sharpe", ""], [9, 5, 7, 6, 7, 3]):
            tk.Label(hdr, text=col, bg=BORDER, fg=TEXT_DIM,
                     font=("Courier", 7, "bold"),
                     width=w, anchor="w", padx=2).pack(side="left", padx=1, pady=2)

        total_ask_pct = 0.0
        for i, etf in enumerate(self.portfolio):
            total_ask_pct += etf["pct"]
            d   = self.etf_data_cache.get(etf["ticker"], etf)
            row = tk.Frame(self.portfolio_list_frame, bg=BG2)
            row.pack(fill="x", pady=1)

            status_color = GREEN if etf.get("status") == "ok" else YELLOW
            vals = [
                (etf["ticker"],                           BLUE,         9),
                (f"{etf['pct']*100:.0f}%",                PURPLE,       5),
                (f"{d['annual_return']*100:.1f}%",         status_color, 7),
                (f"{d.get('volatility',0)*100:.0f}%",      TEXT_DIM,     6),
                (f"{d.get('sharpe',0):.2f}",               CYAN,         7),
            ]
            for val, col, w in vals:
                tk.Label(row, text=val, bg=BG2, fg=col,
                         font=("Courier", 8), width=w,
                         anchor="w", padx=2).pack(side="left")
            tk.Button(row, text="✕", bg=BG2, fg=RED, relief="flat",
                      font=("Courier", 7), cursor="hand2", width=2,
                      command=lambda i=i: self._remove_etf(i)).pack(side="left")

        ops_pct = max(0.0, 1.0 - total_ask_pct)
        self.alloc_lbl.config(
            text=f"ASK: {total_ask_pct*100:.0f}%  +  Opspar: {ops_pct*100:.0f}%  "
                 f"{'✅' if abs(total_ask_pct + ops_pct - 1) < 0.01 else '⚠️'}",
            fg=GREEN if total_ask_pct > 0 else ORANGE)

    def _fetch_all_async(self):
        """Scarica dati ETF in background (thread separato)."""
        if self.loading:
            return
        self.loading = True
        self.load_status_lbl.config(text="⏳ Download dati Yahoo Finance...", fg=CYAN)

        def _worker():
            tickers = [t for t, _, _, _ in DEFAULT_ETF_LIST]
            for i, ticker in enumerate(tickers):
                self.after(0, lambda t=ticker, i=i:
                    self.load_status_lbl.config(
                        text=f"⏳ [{i+1}/{len(tickers)}] {t}...", fg=CYAN))
                d = fetch_etf_data(ticker)
                self.etf_data_cache[ticker] = d
                # Aggiorna portfolio se già presente
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
        ok_count = sum(1 for d in self.etf_data_cache.values()
                       if d.get("status") == "ok")
        self.load_status_lbl.config(
            text=f"✅ Dati aggiornati: {ok_count}/{len(DEFAULT_ETF_LIST)} ETF OK",
            fg=GREEN)
        self._refresh_portfolio_list()
        self._refresh()
        # Aggiorna preview se ETF selezionato
        raw    = self.sel_ticker_var.get()
        ticker = raw.split("  —  ")[0].strip() if "  —  " in raw else raw.strip()
        if ticker in self.etf_data_cache:
            self._update_etf_preview(ticker, self.etf_data_cache[ticker])

    def _show_recommendations(self):
        """Finestra popup con ranking ETF basato su score composito."""
        if not self.etf_data_cache:
            self.load_status_lbl.config(
                text="⚠️ Prima clicca 'Aggiorna dati'!", fg=YELLOW)
            return

        win = tk.Toplevel(self)
        win.title("⭐ Raccomandazioni ETF — Lista ABIS/SKAT")
        win.configure(bg=BG)
        win.geometry("680x420")

        tk.Label(win, text="⭐ RANKING ETF — Lista ABIS/SKAT",
                 bg=BG, fg=BLUE, font=("Courier", 12, "bold"),
                 padx=20, pady=12).pack(anchor="w")
        tk.Label(win,
                 text="Score = 40% Sharpe + 30% CAGR + 20% (1+MaxDD) + 10% (1-Vol)\n"
                      "Solo ETF con lagerbeskatning 17% (idonei per ASK)",
                 bg=BG, fg=TEXT_DIM, font=("Courier", 8),
                 padx=20).pack(anchor="w")

        # Calcola ranking
        ranked = []
        for ticker, d in self.etf_data_cache.items():
            name = ticker
            for t, _, n, typ in DEFAULT_ETF_LIST:
                if t == ticker:
                    name = f"{n} ({typ})"
                    break
            ranked.append((score_etf(d), ticker, name, d))
        ranked.sort(reverse=True)

        # Tabella
        frame = tk.Frame(win, bg=BG, padx=20)
        frame.pack(fill="both", expand=True)

        cols = ["#", "Ticker", "Nome", "CAGR", "Vol", "Sharpe", "MaxDD", "YTD", "Score"]
        widths = [3, 10, 28, 7, 6, 8, 8, 7, 7]
        hdr = tk.Frame(frame, bg=BG2)
        hdr.pack(fill="x", pady=(8, 2))
        for col, w in zip(cols, widths):
            tk.Label(hdr, text=col, bg=BORDER, fg=TEXT_DIM,
                     font=("Courier", 8, "bold"),
                     width=w, anchor="w", padx=3).pack(side="left", padx=1, pady=2)

        for rank, (sc, ticker, name, d) in enumerate(ranked, 1):
            color = [GREEN, CYAN, BLUE, TEXT, TEXT_DIM][min(rank-1, 4)]
            row   = tk.Frame(frame, bg=BG2 if rank % 2 == 0 else BG)
            row.pack(fill="x", pady=1)
            vals = [
                (f"#{rank}",                          color),
                (ticker,                              BLUE),
                (name[:26],                           TEXT),
                (f"{d['annual_return']*100:.1f}%",    GREEN),
                (f"{d.get('volatility',0)*100:.0f}%", TEXT_DIM),
                (f"{d.get('sharpe',0):.2f}",          CYAN),
                (f"{d.get('max_drawdown',0)*100:.1f}%", RED),
                (f"{d.get('ytd',0)*100:.1f}%",        YELLOW),
                (f"{sc:.3f}",                         color),
            ]
            for (val, col2), w in zip(vals, widths):
                tk.Label(row, text=val, bg=row.cget("bg"), fg=col2,
                         font=("Courier", 8), width=w,
                         anchor="w", padx=3).pack(side="left")

            # Pulsante aggiungi direttamente
            tk.Button(row, text="➕", bg=row.cget("bg"), fg=GREEN,
                      relief="flat", font=("Courier", 8), cursor="hand2",
                      command=lambda t=ticker: (
                          self.sel_ticker_var.set(t),
                          self._add_etf()
                      )).pack(side="right", padx=4)

    # ── Events Panel ──────────────────────────────────────────────────────────
    def _build_events_panel(self, parent):
        card = self._card(parent, "📅 Variazioni nel Tempo", YELLOW)
        tk.Label(card,
                 text="Cambia stipendio, spese o tasse in un mese specifico.",
                 bg=BG2, fg=TEXT_DIM, font=("Courier", 8),
                 wraplength=300).pack(anchor="w", pady=(0, 8))

        field_labels = {
            "grossSalary":     "💼 Stipendio lordo (DKK/mese)",
            "incomeTax":       "🏛️ Tassa reddito (es. 37 = 37%)",
            "monthlyExpenses": "🛒 Spese mensili (DKK)",
            "savingsReturn":   "🏦 Rendimento risparmio (es. 3 = 3%)",
        }

        r1 = tk.Frame(card, bg=BG2)
        r1.pack(fill="x", pady=(0, 4))
        tk.Label(r1, text="Campo:", bg=BG2, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(side="left")
        ttk.Combobox(r1, textvariable=self.ev_field_var,
                     values=list(field_labels.keys()),
                     state="readonly", width=22,
                     font=("Courier", 9)).pack(side="right")

        self.ev_field_lbl = tk.Label(card, text="", bg=BG2, fg=YELLOW,
                                     font=("Courier", 8))
        self.ev_field_lbl.pack(anchor="w")
        def on_fc(*_):
            self.ev_field_lbl.config(
                text=field_labels.get(self.ev_field_var.get(), ""))
        self.ev_field_var.trace_add("write", on_fc); on_fc()

        r2 = tk.Frame(card, bg=BG2)
        r2.pack(fill="x", pady=(6, 0))
        tk.Label(r2, text="Mese:", bg=BG2, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(side="left")
        tk.Entry(r2, textvariable=self.ev_month_var,
                 width=5, bg="#0d1117", fg=YELLOW,
                 insertbackground=YELLOW, font=("Courier", 10),
                 relief="flat", highlightthickness=1,
                 highlightbackground=BORDER,
                 highlightcolor=YELLOW).pack(side="left", padx=(4, 12), ipady=3)
        tk.Label(r2, text="Valore:", bg=BG2, fg=TEXT_DIM,
                 font=("Courier", 9)).pack(side="left")
        tk.Entry(r2, textvariable=self.ev_value_var,
                 width=10, bg="#0d1117", fg=YELLOW,
                 insertbackground=YELLOW, font=("Courier", 10),
                 relief="flat", highlightthickness=1,
                 highlightbackground=BORDER,
                 highlightcolor=YELLOW).pack(side="left", padx=(4, 0), ipady=3)

        tk.Button(card, text="➕ Aggiungi Evento",
                  bg="#1f3d2a", fg=GREEN, relief="flat",
                  font=("Courier", 9, "bold"), cursor="hand2", pady=4,
                  command=self._add_event).pack(fill="x", pady=(8, 4))

        self.events_list_frame = tk.Frame(card, bg=BG2)
        self.events_list_frame.pack(fill="x", pady=(0, 8))
        self._refresh_events_list()

    def _add_event(self):
        try:
            field = self.ev_field_var.get()
            month = int(self.ev_month_var.get())
            value = float(self.ev_value_var.get().replace(",", "."))
            max_m = max(1, int(self._parse(self.years, DEFAULT_YEARS, 1, 30))) * 12
            if month < 1 or month > max_m:
                return
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
            tk.Label(self.events_list_frame, text="Nessun evento.",
                     bg=BG2, fg=TEXT_DIM, font=("Courier", 8)).pack(anchor="w")
            return
        for i, ev in enumerate(self.events):
            row = tk.Frame(self.events_list_frame, bg=BG2)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=ev["label"], bg=BG2, fg=YELLOW,
                     font=("Courier", 8)).pack(side="left")
            tk.Button(row, text="✕", bg=BG2, fg=RED, relief="flat",
                      font=("Courier", 8), cursor="hand2",
                      command=lambda i=i: self._remove_event(i)).pack(side="right")

    # ── Strategy Panel ────────────────────────────────────────────────────────
    def _build_strategy_panel(self, parent):
        card = self._card(parent, "🎯 Confronto Strategie", CYAN)
        tk.Label(card, text="Confronto automatico strategie principali:",
                 bg=BG2, fg=TEXT_DIM, font=("Courier", 8)).pack(anchor="w", pady=(0, 6))
        self.strategy_frame = tk.Frame(card, bg=BG2)
        self.strategy_frame.pack(fill="x")

    def _update_strategy_panel(self, params):
        for w in self.strategy_frame.winfo_children():
            w.destroy()

        strategies = []
        if self.portfolio:
            strategies.append(("Portafoglio corrente", params["portfolio"]))

        # Strategia: tutto S&P500
        cspx = self.etf_data_cache.get("CSPX.L", _fallback_etf("CSPX.L"))
        strategies.append(("100% S&P500 (CSPX.L)", [
            {"ticker": "CSPX.L", "name": "S&P500", "pct": 1.0,
             "annual_return": cspx["annual_return"],
             "volatility": cspx.get("volatility", 0.175)}
        ]))
        # Strategia: tutto Opspar
        strategies.append(("100% Opspar", []))

        widths = [20, 14, 13]
        hdr = tk.Frame(self.strategy_frame, bg=BG2)
        hdr.pack(fill="x")
        for col, w in zip(["Strategia", "Patrimonio", "Guadagno"], widths):
            tk.Label(hdr, text=col, bg=BORDER, fg=TEXT_DIM,
                     font=("Courier", 8, "bold"),
                     width=w, anchor="w", padx=3).pack(side="left", padx=1, pady=1)

        for i, (name, portfolio) in enumerate(strategies):
            test_params = {**params, "portfolio": portfolio}
            res   = simulate_monthly(test_params, self.events)
            color = GREEN if i == 0 and self.portfolio else TEXT
            row   = tk.Frame(self.strategy_frame, bg=BG2)
            row.pack(fill="x", pady=1)
            for val, w in zip([name, fmt(res["final_total"]),
                                fmt(res["final_net_gain"])], widths):
                tk.Label(row, text=val, bg=BG2, fg=color,
                         font=("Courier", 8),
                         width=w, anchor="w", padx=3).pack(side="left")

    # ── Charts ────────────────────────────────────────────────────────────────
    def _build_charts(self, parent):
        tab_bar = tk.Frame(parent, bg=BG)
        tab_bar.pack(fill="x", pady=(0, 10))
        self.tab_btns = {}
        tabs = [
            ("proiezione",   "PROIEZIONE"),
            ("portafoglio",  "PORTAFOGLIO"),
            ("strategie",    "STRATEGIE"),
            ("montecarlo",   "RISCHIO MC"),
            ("composizione", "COMPOSIZIONE"),
            ("ask_fill",     "ASK FILL"),
        ]
        for t, lbl in tabs:
            b = tk.Button(tab_bar, text=lbl, bg=BG, fg=TEXT_DIM,
                          relief="flat", font=("Courier", 9, "bold"),
                          cursor="hand2", padx=10, pady=6,
                          command=lambda x=t: self._set_tab(x))
            b.pack(side="left", padx=(0, 3))
            self.tab_btns[t] = b
        self._highlight_tab(self.tab.get())

        self.fig    = Figure(figsize=(8, 4.8), facecolor=BG2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.insight_lbl = tk.Label(parent, text="", bg=BG2, fg=TEXT,
                                    font=("Courier", 9), wraplength=760,
                                    justify="left", padx=12, pady=8)
        self.insight_lbl.pack(fill="x", pady=(6, 0))

    # ── Core logic ────────────────────────────────────────────────────────────
    def _apply_optimal(self):
        opt = find_optimal_monthly(self._params(), self.events)
        # Scala tutte le % del portafoglio
        if self.portfolio and opt["pct"] > 0:
            total = sum(e["pct"] for e in self.portfolio)
            if total > 0:
                factor = opt["pct"] / total
                for e in self.portfolio:
                    e["pct"] = min(1.0, e["pct"] * factor)
        self._refresh_portfolio_list()
        self._refresh()

    def _set_tab(self, t):
        self.tab.set(t)
        self._highlight_tab(t)
        self._draw_chart()

    def _highlight_tab(self, t):
        for name, btn in self.tab_btns.items():
            btn.config(bg=("#21262d" if name == t else BG),
                       fg=(BLUE if name == t else TEXT_DIM))

    def _refresh(self):
        try:
            params = self._params()
            res    = simulate_monthly(params, self.events)
            data   = res["data"]
            sav_m  = res["sav_monthly"]

            self.kpi_labels["net"].config(text=fmt(res["net_monthly"]))
            self.kpi_labels["sav"].config(
                text=fmt(sav_m) if sav_m > 0 else "⚠️ negativo",
                fg=ORANGE if sav_m > 0 else RED)
            self.kpi_labels["ask_dep"].config(
                text=fmt(res["ask_deposited"]),
                fg=GREEN if res["ask_full"] else PURPLE)
            self.kpi_labels["tot"].config(text=fmt(res["final_total"]))
            self.kpi_labels["gain"].config(text=fmt(res["final_net_gain"]))

            # Mesi a riempire ASK
            total_ask_pct = sum(e["pct"] for e in self.portfolio)
            if sav_m > 0 and total_ask_pct > 0:
                monthly_ask = sav_m * total_ask_pct
                mtf = ASK_DEPOSIT_LIMIT / monthly_ask
                self.kpi_labels["ask_months"].config(
                    text=f"{mtf:.0f} mesi ({mtf/12:.1f} anni)")
            elif total_ask_pct == 0:
                self.kpi_labels["ask_months"].config(text="∞ (no ETF)")
            else:
                self.kpi_labels["ask_months"].config(text="—")

            ask_fill = min(100, res["ask_deposited"] / ASK_DEPOSIT_LIMIT * 100)

            # Insight
            ask_fill_month = next(
                (d["month"] for d in data if d["ask_deposited"] >= ASK_DEPOSIT_LIMIT),
                None)
            n_etf = len(self.portfolio)
            portfolio_str = (", ".join(f"{e['ticker']} {e['pct']*100:.0f}%"
                                       for e in self.portfolio)
                             if self.portfolio else "nessun ETF")

            if sav_m <= 0:
                insight = "⚡ ATTENZIONE: spese > reddito netto. Impossibile risparmiare."
            else:
                fill_txt = (f"ASK pieno al mese {ask_fill_month} ({ask_fill_month/12:.1f} anni)."
                            if ask_fill_month else
                            f"ASK: {ask_fill:.1f}% riempito in {params['years']} anni.")
                insight = (
                    f"⚡ Portafoglio: {portfolio_str}.  "
                    f"{fill_txt}  "
                    f"Dopo ASK pieno → tutto Opspar (40%).  "
                    f"Patrimonio finale: {fmt(res['final_total'])}  |  "
                    f"Guadagno: {fmt(res['final_net_gain'])}"
                )
            self.insight_lbl.config(text=insight)

            self._last_data   = data
            self._last_res    = res
            self._last_params = params

            self._update_strategy_panel(params)
            self._draw_chart()

        except Exception:
            import traceback; traceback.print_exc()

    # ── Stile ax ──────────────────────────────────────────────────────────────
    def _style_ax(self, ax, monthly=False):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.grid(True, color=BORDER, linestyle="--", linewidth=0.4, alpha=0.7)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
        ax.set_xlabel("Mese" if monthly else "Anno", color=TEXT_DIM, fontsize=8)

    def _add_event_vlines(self, ax, ymax, monthly=False):
        evs = (sorted(set(ev["month"] for ev in self.events)) if monthly
               else sorted(set(ev["month"] // 12 for ev in self.events)))
        for x in evs:
            ax.axvline(x=x, color=YELLOW, lw=0.8, linestyle=":", alpha=0.8)
            ax.text(x + 0.2, ymax * 0.93, "Δ", color=YELLOW, fontsize=6)

    def _monthly_to_yearly(self, data):
        yearly = [data[0]]
        for m in range(12, len(data), 12):
            yearly.append(data[m])
        if data[-1] not in yearly:
            yearly.append(data[-1])
        return yearly

    # ── Draw ──────────────────────────────────────────────────────────────────
    def _draw_chart(self):
        if not hasattr(self, "_last_data"):
            return

        data   = self._last_data
        params = self._last_params
        tab    = self.tab.get()
        months = params["years"] * 12
        xs_m   = [d["month"] for d in data]

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        self._style_ax(ax, monthly=(tab in ("ask_fill",)))

        etf_colors = [PURPLE, BLUE, CYAN, GREEN, ORANGE, RED, YELLOW]

        if tab == "proiezione":
            ax.set_title("PROIEZIONE PATRIMONIO (annuale)",
                         color=TEXT_DIM, fontsize=9, loc="left", pad=8)
            yd    = self._monthly_to_yearly(data)
            xs    = [d["month"] / 12 for d in yd]
            ask_v = [d["ask_balance"] for d in yd]
            ops_v = [d["ops_balance"] for d in yd]
            tot_v = [d["total"]       for d in yd]
            dep_v = [d["deposited"]   for d in yd]
            ax.fill_between(xs, ask_v, alpha=0.15, color=PURPLE)
            ax.fill_between(xs, ops_v, alpha=0.15, color=BLUE)
            ax.plot(xs, ask_v, color=PURPLE, lw=2,   label="ASK (ETF)")
            ax.plot(xs, ops_v, color=BLUE,   lw=2,   label="Opsparingskonto")
            ax.plot(xs, tot_v, color=GREEN,  lw=2.5, linestyle="--", label="Totale")
            ax.plot(xs, dep_v, color=ORANGE, lw=1.5, linestyle=":",  label="Versato")
            ax.axhline(y=ASK_DEPOSIT_LIMIT, color=YELLOW, lw=1,
                       linestyle="--", alpha=0.6, label=f"Limite ASK")
            ymax = max(tot_v) if tot_v else 1
            self._add_event_vlines(ax, ymax)
            ax.set_xlabel("Anno", color=TEXT_DIM, fontsize=8)

        elif tab == "portafoglio":
            ax.set_title("COMPOSIZIONE PORTAFOGLIO ETF NEL TEMPO",
                         color=TEXT_DIM, fontsize=9, loc="left", pad=8)
            if not self.portfolio:
                ax.text(0.5, 0.5, "Aggiungi ETF al portafoglio →",
                        transform=ax.transAxes, color=TEXT_DIM,
                        ha="center", va="center", fontsize=11)
            else:
                yd  = self._monthly_to_yearly(data)
                xs  = [d["month"] / 12 for d in yd]
                bottom = np.zeros(len(xs))
                for i, etf in enumerate(self.portfolio):
                    col   = etf_colors[i % len(etf_colors)]
                    vals  = [d["etf_balances"].get(etf["ticker"], 0) for d in yd]
                    ax.bar(xs, vals, bottom=bottom, color=col, width=0.7,
                           alpha=0.8, label=etf["ticker"])
                    bottom += np.array(vals)
                # Opspar
                ops_v = [d["ops_balance"] for d in yd]
                ax.bar(xs, ops_v, bottom=bottom, color=TEXT_DIM,
                       width=0.7, alpha=0.5, label="Opspar")
            ax.set_xlabel("Anno", color=TEXT_DIM, fontsize=8)

        elif tab == "strategie":
            ax.set_title("CONFRONTO STRATEGIE",
                         color=TEXT_DIM, fontsize=9, loc="left", pad=8)
            # ETF correnti vs variazioni scala
            scales = list(range(0, 101, 10))
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
            ax.plot(scales, totals, color=GREEN, lw=2.5,
                    label="Patrimonio vs scala allocazione ETF")
            ax.set_xlabel("Scala allocazione ASK (%)", color=TEXT_DIM, fontsize=8)

        elif tab == "montecarlo":
            ax.set_title("RISCHIO — MONTE CARLO 300 sim",
                         color=TEXT_DIM, fontsize=9, loc="left", pad=8)
            mc     = simulate_monte_carlo_monthly(params, self.events, 300)
            xs_arr = np.arange(months + 1)
            ax.fill_between(xs_arr, mc["p5"],  mc["p95"], color=BLUE, alpha=0.10,
                            label="P5–P95")
            ax.fill_between(xs_arr, mc["p10"], mc["p90"], color=BLUE, alpha=0.15,
                            label="P10–P90")
            ax.fill_between(xs_arr, mc["p25"], mc["p75"], color=BLUE, alpha=0.22,
                            label="P25–P75")
            ax.plot(xs_arr, mc["p50"], color=GREEN,  lw=2.5, label="Mediana")
            ax.plot(xs_arr, mc["p10"], color=RED,    lw=1.2, linestyle="--",
                    label="P10")
            ax.plot(xs_arr, mc["p90"], color=PURPLE, lw=1.2, linestyle="--",
                    label="P90")
            ax.plot(xs_m, [d["total"] for d in data], color=YELLOW, lw=2,
                    linestyle=":", label="Base")
            ymax = float(mc["p90"].max()) if len(mc["p90"]) > 0 else 1
            self._add_event_vlines(ax, ymax, monthly=True)
            ax.set_xlabel("Mese", color=TEXT_DIM, fontsize=8)

        elif tab == "composizione":
            ax.set_title("VERSATO vs GUADAGNO NETTO",
                         color=TEXT_DIM, fontsize=9, loc="left", pad=8)
            yd   = self._monthly_to_yearly(data)
            xs   = [d["month"] / 12 for d in yd]
            dep  = [d["deposited"]       for d in yd]
            gain = [max(0, d["net_gain"]) for d in yd]
            ax.bar(xs, dep,  color="#30363d", label="Versato",       width=0.7)
            ax.bar(xs, gain, bottom=dep, color=GREEN,
                   label="Guadagno netto", width=0.7)
            ax.set_xlabel("Anno", color=TEXT_DIM, fontsize=8)

        elif tab == "ask_fill":
            ax.set_title("RIEMPIMENTO ASK — Capitale versato",
                         color=TEXT_DIM, fontsize=9, loc="left", pad=8)
            ask_dep_v = [d["ask_deposited"] for d in data]
            ask_bal_v = [d["ask_balance"]   for d in data]
            ops_bal_v = [d["ops_balance"]   for d in data]
            ax.fill_between(xs_m, ask_dep_v, alpha=0.25, color=PURPLE)
            ax.plot(xs_m, ask_dep_v, color=PURPLE, lw=2,
                    label="ASK versato")
            ax.plot(xs_m, ask_bal_v, color=GREEN,  lw=2, linestyle="--",
                    label="ASK saldo")
            ax.plot(xs_m, ops_bal_v, color=BLUE,   lw=2, linestyle="-.",
                    label="Opspar saldo")
            ax.axhline(y=ASK_DEPOSIT_LIMIT, color=YELLOW, lw=1.5,
                       linestyle="--", alpha=0.8, label="Limite ASK")
            ymax = max(max(ask_bal_v), max(ops_bal_v), ASK_DEPOSIT_LIMIT)
            self._add_event_vlines(ax, ymax, monthly=True)

        ax.legend(fontsize=8, facecolor=BG2, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper left")
        self.canvas.draw()


if __name__ == "__main__":
    app = App()
    app.mainloop()