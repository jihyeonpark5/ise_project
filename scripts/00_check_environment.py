from __future__ import annotations

import os
from pathlib import Path
import importlib
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"

# Avoid Windows permission issues when matplotlib tries to write cache files.
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_DIRECTORIES = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "raw" / "boundary",
    PROJECT_ROOT / "data" / "raw" / "facilities",
    PROJECT_ROOT / "data" / "raw" / "population",
    PROJECT_ROOT / "data" / "raw" / "traffic",
    PROJECT_ROOT / "data" / "raw" / "slope",
    PROJECT_ROOT / "data" / "raw" / "accident",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "output",
    PROJECT_ROOT / "notebooks",
    PROJECT_ROOT / "scripts",
]

REQUIRED_PACKAGES = {
    "osmnx": "osmnx",
    "geopandas": "geopandas",
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "folium": "folium",
    "shapely": "shapely",
    "pyproj": "pyproj",
    "statsmodels": "statsmodels",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "ortools": "ortools",
}


def check_directories() -> bool:
    print("[1/2] Checking project directories...")
    missing_directories: list[Path] = []

    for directory in REQUIRED_DIRECTORIES:
        if directory.exists() and directory.is_dir():
            print(f"  [OK] {directory.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  [MISSING] {directory.relative_to(PROJECT_ROOT)}")
            missing_directories.append(directory)

    if missing_directories:
        print("\nDirectory check failed.")
        return False

    print("Directory check passed.\n")
    return True


def check_imports() -> bool:
    print("[2/2] Checking package imports...")
    failed_imports: list[str] = []

    for package_name, module_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            print(f"  [OK] {package_name}")
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(f"  [FAILED] {package_name}: {exc}")
            failed_imports.append(package_name)

    if failed_imports:
        print("\nImport check failed.")
        print("Missing or broken packages:", ", ".join(failed_imports))
        return False

    print("Import check passed.\n")
    return True


def main() -> int:
    print("=== pm_safe_route Step 1 Environment Check ===\n")

    directories_ok = check_directories()
    imports_ok = check_imports()

    if directories_ok and imports_ok:
        print("All checks passed. Your Step 1 environment is ready.")
        return 0

    print("Some checks failed. Please review the messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
