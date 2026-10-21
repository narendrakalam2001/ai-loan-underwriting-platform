"""Print the exact versions of the packages that matter for unpickling the saved models.

Run it in the SAME environment you trained in (conda env `capstone_env`):
    python scripts/print_pins.py
Paste the output into requirements_api.txt so Render loads the models with the same
library versions they were saved with.
"""
from importlib import metadata

PKGS = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost",
        "lightgbm", "catboost", "joblib"]

for name in PKGS:
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"# {name}: not installed")
