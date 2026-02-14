Bitcoin price prediction research

This repository contains exploratory notebooks in `notebooks/` and supporting
project code in `src/`.

Goal
----
I will develop algorithms to predict Bitcoin price (a specific target) using
orderbook and trades data. The README will be expanded later with details.

Quickstart
----------
1. Create and activate a Python virtual environment (example using python3.9):

```bash
python3.9 -m venv myenv
source myenv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook `notebooks/1.ipynb` with Jupyter or open it in VS Code.

Notes
-----
- Project source code is under `src/`. Notebooks should be able to import
  project modules by adding `sys.path.append(os.path.join(os.getcwd(), 'src'))`
  at the top of the notebook (see first cell of `notebooks/1.ipynb`).
- If you rename the folder `data collector` to `data_collector`, the compatibility
  shim in `src/data_collector/__init__.py` can be simplified/removed.
