# 📈 Piano di Accumulo — Aktiesparekonto & Opsparingskonto

Applicazione desktop in Python per simulare e ottimizzare un piano di
accumulo finanziario secondo la fiscalità danese.

---

## 🎯 Cosa fa

Simula mese per mese la crescita di un portafoglio composto da:

- **Aktiesparekonto (ASK)** — conto titoli danese con tassazione
  *lagerbeskatning* **17%** sui guadagni annuali.  
  Limite di versamento: **174.200 DKK** (2025).
- **Opsparingskonto (Opspar)** — conto di risparmio ordinario con
  tassazione *kapitalindkomst* **40%** sugli interessi.

Il capitale che supera il limite ASK viene automaticamente dirottato
in Opspar.

---

## ⚙️ Funzionalità principali

| Funzione | Descrizione |
|---|---|
| **Parametri base** | Stipendio lordo, aliquota IRPEF, spese mensili, rendimento risparmio, orizzonte |
| **Portafoglio ETF** | Aggiunta di ETF dalla lista ABIS/SKAT con allocazione percentuale personalizzata |
| **Dati reali** | Download automatico storico prezzi da Yahoo Finance (CAGR, volatilità, Sharpe, MaxDD, YTD) |
| **Fallback** | Dati storici predefiniti se Yahoo Finance non è disponibile |
| **Eventi temporali** | Variazioni future di stipendio, tasse o spese a partire da un mese specifico |
| **Ottimizzazione** | Calcolo automatico della ripartizione ASK/Opspar ottimale |
| **Monte Carlo** | 300 simulazioni stocastiche con bande di confidenza P5–P95 |
| **Ranking ETF** | Score composito (Sharpe 40% + CAGR 30% + MaxDD 20% + Vol 10%) |

---

## 📊 Grafici disponibili

- **PROIEZIONE** — Crescita annuale di ASK, Opspar e totale vs versato
- **PORTAFOGLIO** — Composizione a barre degli ETF nel tempo
- **STRATEGIE** — Confronto patrimonio al variare della scala di allocazione ASK
- **RISCHIO MC** — Bande Monte Carlo (P5/P25/P50/P75/P95)
- **COMPOSIZIONE** — Versato vs guadagno netto
- **ASK FILL** — Progressione mensile del riempimento ASK

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
| … | (altri 6 ETF inclusi) | |

---

## 🚀 Installazione e avvio

### 1. Prerequisiti

- Python **3.10+**
- `tkinter` incluso nella distribuzione standard di Python su Windows

### 2. Installa le dipendenze

```bash
pip install -r requirements.txt
```

> `yfinance`, `pandas` e `requests` sono **opzionali**: se non installati,
> l'app usa dati storici predefiniti senza connessione a internet.

### 3. Avvia l'applicazione

```bash
python piano_accumulo.py
```

---

## 🖥️ Interfaccia

```
┌─────────────────────────────────────────────────────────────┐
│ 📈 Piano di Accumulo — ASK · Opspar                         │
├──────────────────┬──────────────────────────────────────────┤
│ 📊 Parametri     │                                          │
│ 📦 Portafoglio   │   Grafici interattivi (6 tab)            │
│ 📅 Variazioni    │                                          │
│ 🎯 Strategie     │                                          │
└──────────────────┴──────────────────────────────────────────┘
```

- **Pannello sinistro** — Input scrollable con parametri, ETF, eventi
- **Pannello destro** — Grafici aggiornati in tempo reale
- **KPI** — Netto mensile, risparmio, ASK versato, patrimonio, guadagno

---

## 📐 Formule utilizzate

### CAGR (Compound Annual Growth Rate)
```
CAGR = (P_finale / P_iniziale)^(1/n_anni) - 1
```

### Sharpe Ratio (risk-free 3%)
```
Sharpe = (R_mensile_medio - Rf_mensile) / σ_mensile * √12
```

### Score ETF composito
```
Score = 0.40 × (Sharpe/3) + 0.30 × (CAGR/0.5)
      + 0.20 × (1 + MaxDD) + 0.10 × (1 - Vol)
```

### Tassazione ASK (lagerbeskatning)
```
Guadagno_netto = Guadagno_lordo × (1 - 0.17)
```

### Tassazione Opspar (kapitalindkomst)
```
Interesse_netto = Interesse_lordo × (1 - 0.40)
```

---

## ⚠️ Disclaimer

Questo strumento è a **scopo educativo e di pianificazione personale**.
Non costituisce consulenza finanziaria. Le proiezioni si basano su
rendimenti storici che non garantiscono risultati futuri.  
Verificare sempre le normative fiscali aggiornate su [skat.dk](https://skat.dk).

---

## 📁 Struttura file

```
Piano di Accumulo/
├── piano_accumulo.py   # Applicazione principale
├── requirements.txt    # Dipendenze Python
└── README.md           # Questa documentazione
```