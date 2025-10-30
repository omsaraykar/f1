# F1 Model — Group 18 ML Lab V B

Comprehensive project for working with Formula 1 telemetry and session data using FastF1, plus modeling/analysis utilities. This repository contains code and cached session data (FastF1 cache) used by the project's scripts and models.

Repository: https://github.com/omsaraykar/f1

## Table of contents
- Project overview
- Quick start
- Environment & dependencies
- Data (FastF1 cache)
- Usage examples
- Project structure
- Contract (inputs / outputs / errors)
- Edge cases & notes
- Development & contribution
- Next steps and suggestions

---

## Project overview

This project collects and analyzes Formula 1 session telemetry and event data using the FastF1 library. It is structured to:

- Load cached F1 session data (already stored under `fastf1_cache/`).
- Provide scripts (entrypoint `main.py`) and modules to process telemetry, extract features, and run models or analysis.
- Be reproducible: environment is provided via `requirements.txt` and an optional `myenv` virtual environment in the repository (if present).

The repository was prepared for a university ML lab (Group 18). Use this README to get started, run experiments, and extend the project.

## Quick start

1. Clone the repository (if not already present):

```bash
# already in this workspace; if cloning elsewhere:
git clone github.com/omsaraykar/f1
cd f1-model
```

2. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Note: This repo contains a `myenv/` virtualenv in the workspace — you can reuse it, but creating a fresh env is recommended for reproducibility.

3. Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the main script (example):

```bash
python main.py
```

`main.py` is the project's primary entrypoint — see the "Usage examples" section below for common command examples and options (if implemented in the repo's CLI).

## Environment & dependencies

- Python: recommended 3.10+ (this repo includes `myenv` with Python 3.13). Use the Python version that matches your environment and installed packages.
- Dependencies are listed in `requirements.txt`. Typical required packages include `fastf1`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and others used for telemetry and modeling.

If you need to replicate the exact environment from `myenv/`, either activate it or recreate the environment using the Python version inside `myenv/bin/python`.

## Data (FastF1 cache)

This repository includes a FastF1 cache under `fastf1_cache/` with multiple seasons (2023, 2024). The cache contains downloaded session data (practice/qualifying/race) and is used to avoid repeated downloads from Ergast/fastf1 servers.

How to use the cache in code (FastF1 example):

```python
import fastf1
fastf1.Cache.enable_cache('fastf1_cache')
session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load()  # loads data from cache if available
```

Be mindful of cache size and disk usage; the cache can be large depending on how many sessions are stored.

## Usage examples

Below are typical usage snippets and examples — adapt to the code in `main.py` or other scripts present in the repository.

- Simple session analysis (Python REPL or script):

```python
import fastf1
fastf1.Cache.enable_cache('fastf1_cache')
session = fastf1.get_session(2023, 'Monaco', 'Q')
session.load()
laps = session.laps
print(laps[['Driver', 'LapTime']].head())
```

- Running the project's entrypoint:

```bash
python main.py --season 2023 --event "Monaco" --session Q
```

If `main.py` has no CLI flags, open the file to see how it is configured and adapt the invocation accordingly.

## Project structure

Top-level files and directories (explain purpose):

- `main.py` — primary script/entrypoint (load sessions, run experiments, or produce outputs).
- `requirements.txt` — pinned Python dependencies for the project.
- `fastf1_cache/` — local FastF1 session cache (many sessions across seasons).
- `myenv/` — included Python virtual environment (optional; typically not committed in public repos).

If the repository contains additional modules (e.g., `src/` or `notebooks/`), list them here and their responsibilities.

## Contract (short)

Inputs:

- Season, event name, session type (e.g., practice/qualifying/race) or path to cached session files.
- Optional: telemetry filters, driver selections, model parameters.

Outputs:

- Dataframes with lap/session/telemetry data.
- Visualizations (plots), model predictions, metrics, and saved model artifacts if implemented.

Error modes:

- Missing cache/session: fastf1 will raise an error if session data is not available — ensure cache exists or allow fastf1 to download.
- Dependency mismatch: import errors or runtime exceptions if required packages are missing or versions differ.

Success criteria:

- Scripts run without unhandled exceptions for a supported session and produce expected dataframes or model outputs.

## Edge cases & notes

- Empty sessions or sessions missing telemetry for some drivers — code should handle empty data frames and log useful warnings.
- Large memory/disk usage when loading many sessions — use streaming/partial loads when possible.
- Timezones and timestamps — FastF1 handles many details but ensure consistency when combining data from multiple sources.

---

Authors: Group 18 ML Lab V B

Repository: https://github.com/omsaraykar/f1

Created: 2025-10-30
