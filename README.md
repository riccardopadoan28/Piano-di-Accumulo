# 📈 Piano di Accumulo — Aktiesparekonto & Opsparingskonto

> Applicazione desktop in Python per simulare e ottimizzare un piano di
> accumulo finanziario secondo la fiscalità danese.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🎯 Cosa fa

Simula **mese per mese** la crescita di un portafoglio composto da:

- **Aktiesparekonto (ASK)** — conto titoli danese con tassazione
  *lagerbeskatning* **17%** sui guadagni annuali.  
  Limite di versamento: **174.200 DKK** (2025), con crescita automatica
  del **+2.5%/anno** opzionale.
- **Opsparingskonto (Opspar)** — conto di risparmio ordinario con
  tassazione *kapitalindkomst* **40%** sugli interessi.

Il capitale che supera il limite ASK viene automaticamente dirottato
in Opspar.

---

## ⚙️ Funzionalità principali

| Funzione | Descrizione |
|---|---|
| **Parametri base** | Stipendio lordo, aliquota, spese mensili, rendimento risparmio, orizzonte, inflazione |
| **Portafoglio ETF** | Aggiunta di ETF dalla lista ABIS/SKAT con allocazione % personalizzata |
| **Dati reali** | Download automatico storico prezzi da Yahoo Finance (CAGR, volatilità, Sharpe, MaxDD, YTD) |
| **Fallback offline** | Dati storici predefiniti se Yahoo Finance non è disponibile |
| **PAL-skat** | Tassazione annuale realistica applicata a dicembre (opzionale) |
| **Crescita limite ASK** | Aggiornamento automatico del limite +2.5%/anno (opzionale) |
| **Eventi temporali** | Variazioni future di stipendio, tasse o spese a partire da un mese specifico |
| **Ottimizzazione** | Calcolo automatico della ripartizione ASK/Opspar ottimale |
| **Monte Carlo** | **1000** simulazioni stocastiche con bande di confidenza P5–P95 |
| **Confronto piani** | Confronto visivo di fino a **6 scenari** con parametri diversi |
| **Ranking ETF** | Score composito (Sharpe 40% + CAGR 30% + MaxDD 20% + Vol 10%) |
| **Export Excel** | Esportazione dati simulazione in `.xlsx` con `pandas` + `openpyxl` |
| **Tema chiaro/scuro** | Toggle dark/light mode in tempo reale |

---

## 📊 Grafici disponibili

| Tab | Icona | Descrizione |
|---|---|---|
| **PROIEZIONE** | 📈 | Crescita annuale di ASK, Opspar e totale vs versato + curva reale (inflazione) |
| **PORTAFOGLIO** | 📊 | Composizione a barre degli ETF nel tempo |
| **STRATEGIE** | 🎯 | Confronto patrimonio al variare della scala di allocazione ASK |
| **RISCHIO MC** | 🎲 | Bande Monte Carlo (P5/P25/P50/P75/P95) — 1000 simulazioni |
| **COMPOSIZIONE** | 🍰 | Versato vs guadagno netto |
| **ASK FILL** | 🟣 | Progressione mensile del riempimento ASK |
| **CONFRONTO PIANI** | ⚖️ | Confronto tra scenari diversi con tabella riepilogativa |

---

## 🇩🇰 ETF supportati (lista ABIS/SKAT)

Solo ETF ad accumulazione (*Acc*) idonei per ASK con tassazione lagerbeskatning:

| Ticker | Nome | Categoria |
|---|---|---|
| CSPX.L | iShares Core S&P 500 Acc | S&P 500 |
| VUSA.AS | Vanguard S&P 500 Acc | S&P 500 |
| SWDA.L | iShares Core MSCI World Acc | World |
| VWCE.DE | Vanguard FTSE All-World Acc | All-World |
| IQQQ.DE | iShares Nasdaq 100 Acc | Nasdaq |
| AGGH.L | iShares Global Agg Bond Acc | Bond |
| IUSQ.DE | iShares MSCI World Acc | World |
| EUNL.DE | iShares Core MSCI World Acc (EUR) | World |
| IS3N.DE | iShares Core MSCI EM IMI Acc | Emerging Markets |
| VTWO.L  | Vanguard U.S. Small Cap Acc | Small Cap |
| ZPRV.DE | SPDR MSCI USA Small Cap Acc | Small Cap |
| IQQH.DE | iShares Global Clean Energy Acc | Settoriale |

---

## 🚀 Installazione e avvio

### 1. Prerequisiti

- Python **3.10+**
- `tkinter` incluso nella distribuzione standard di Python su Windows

### 2. Clona o scarica il progetto

```bash
git clone https://github.com/tuo-utente/piano-accumulo.git
cd piano-accumulo
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

> **Librerie opzionali:** `yfinance`, `pandas`, `openpyxl` e `requests`.  
> Se non installate, l'app funziona offline con dati storici predefiniti  
> (disabilitando download Yahoo Finance ed export Excel).

### 4. Avvia l'applicazione

```bash
python piano_accumulo.py
```

---

## 🖥️ Interfaccia

```
┌──────────────────────────────────────────────────────────────────────┐
│  📈 Piano di Accumulo — ASK · Opspar                    [☀️ / 🌙]   │
├─────────────────────────┬────────────────────────────────────────────┤
│  📊 PARAMETRI BASE      │                                            │
│    Stipendio · Tasse    │                                            │
│    Spese · Rendimento   │   📈 PROIEZIONE   [aggiornato in realtime] │
│                         │                                            │
│  📦 PORTAFOGLIO ETF     │   Tab: PROIEZIONE | PORTAFOGLIO |          │
│    + Cerca / Aggiungi   │        STRATEGIE  | MC  |                  │
│    Ranking automatico   │        COMPOSIZIONE | ASK FILL |           │
│                         │        CONFRONTO PIANI                     │
│  📅 VARIAZIONI FUTURE   │                                            │
│    Stipendio / Tasse    ├────────────────────────────────────────────┤
│    Spese mensili        │  ⚡ Insight automatici sul piano           │
│                         │                                            │
│  🎯 STRATEGIE           │                                            │
│                         │                                            │
│  ⚖️  CONFRONTO PIANI    │                                            │
│    Fino a 6 scenari     │                                            │
└─────────────────────────┴────────────────────────────────────────────┘
│  KPI: Netto/mese · Risparmio · ASK versato · Patrimonio · Guadagno  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Formule utilizzate

### CAGR (Compound Annual Growth Rate)
```
CAGR = (P_finale / P_iniziale)^(1/n_anni) - 1
```

### Sharpe Ratio (risk-free 3%)
```
Sharpe = (R_mensile_medio - Rf_mensile) / σ_mensile × √12
```

### Score ETF composito
```
Score = 0.40 × (Sharpe/3) + 0.30 × (CAGR/0.5)
      + 0.20 × (1 + MaxDD) + 0.10 × (1 - Vol)
```

### Tassazione ASK — lagerbeskatning (PAL-skat)
```
Guadagno_netto = Guadagno_lordo × (1 - 0.17)
```
> Applicata a dicembre su guadagno annuale (saldo 31/12 - saldo 1/1)

### Tassazione Opspar — kapitalindkomst
```
Interesse_netto = Interesse_lordo × (1 - 0.40)
```

### Monte Carlo (log-normale)
```
r_annuale ~ LogNormal( log(1+μ) - σ²/2 , σ )
```
> μ = rendimento atteso ETF, σ = volatilità storica (default 18%)

### Patrimonio reale (corretto per inflazione)
```
Patrimonio_reale = Patrimonio_nominale / (1 + inflazione)^(mesi/12)
```

---

## 📁 Struttura file

```
Piano di Accumulo/
├── piano_accumulo.py      # Applicazione principale (~1800 righe)
├── requirements.txt       # Dipendenze Python
├── README.md              # Questa documentazione
└── docs/
    ├── screenshot_proiezione.png
    ├── screenshot_portafoglio.png
    ├── screenshot_montecarlo.png
    └── screenshot_confronto.png
```

---

## ⚠️ Disclaimer

Questo strumento è a **scopo educativo e di pianificazione personale**.
Non costituisce consulenza finanziaria. Le proiezioni si basano su
rendimenti storici che **non garantiscono risultati futuri**.  
Verificare sempre le normative fiscali aggiornate su [skat.dk](https://skat.dk).
