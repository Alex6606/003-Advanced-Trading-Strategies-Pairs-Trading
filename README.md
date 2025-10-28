# Pairs Trading — Quick Start

This repository implements a statistical arbitrage **pairs trading** strategy (data via yfinance, Kalman filter, backtesting, and visualizations).

## Requirements

* **Python 3.11** (recommended)
* **Git** (optional, for cloning)

---
## 1) Get the Code

```
git clone https://github.com/<your-user>/003-Advanced-Trading-Strategies-Pairs-Trading.git
cd 003-Advanced-Trading-Strategies-Pairs-Trading
```

(Or download the ZIP and open this folder in your terminal.)

---
## 2) Create & Activate a Virtual Environment

### Windows (PowerShell)

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of your prompt.

---

## 3) Install Dependencies

With the virtual environment **activated**:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4) Run the Project

```bash
python main.py #Here you can run everything and see results
```

---

## 5) Freeze Exact Versions (Optional)

Capture the exact working set for reproducibility:

```bash
pip freeze > requirements.lock.txt
```

Recreate later from the lock:

```bash
pip install -r requirements.lock.txt
```

---

## 6) Update/Add Dependencies

```bash
pip install <package-name>
pip freeze > requirements.lock.txt
```

---

## 7) Clean Reset (Delete & Recreate venv)

### Windows (PowerShell)

```powershell
deactivate
Remove-Item -Recurse -Force .\venv
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
