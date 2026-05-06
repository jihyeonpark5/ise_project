from __future__ import annotations

from pathlib import Path
import os
import sys

import geopandas as gpd

# Step 2에서 생성한 경계 및 격자 파일을 시각적으로 확인하기 위한 입력 경로를 정의함.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# 저장 전용 스크립트이므로 화면 출력이 아닌 Agg 백엔드를 사용하여 권한 문제를 줄임.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Step 2에서 생성한 경계 및 격자 파일을 시각적으로 확인하기 위한 입력 경로를 정의함.
BOUNDARY_PATH = PROJECT_ROOT / "data" / "processed" / "01_gangnam_boundary.parquet"
GRID_PATH = PROJECT_ROOT / "data" / "processed" / "02_gangnam_grid_50m.parquet"

# 미리보기 이미지를 저장할 출력 경로를 정의함.
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_IMAGE_PATH = OUTPUT_DIR / "gangnam_grid_preview.png"


def log(message: str) -> None:
    # 실행 과정을 콘솔에 간단히 출력하기 위함.
    print(message)


def ensure_input_files_exist() -> None:
    # 시각화에 필요한 전처리 결과 파일이 모두 존재하는지 먼저 확인함.
    missing_files: list[str] = []

    if not BOUNDARY_PATH.exists():
        missing_files.append(str(BOUNDARY_PATH))
    if not GRID_PATH.exists():
        missing_files.append(str(GRID_PATH))

    if missing_files:
        raise FileNotFoundError(
            "Preview input files are missing. "
            "Please run scripts/01_build_gangnam_boundary_and_grid.py first. "
            f"Missing: {', '.join(missing_files)}"
        )


def load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # 경계와 격자 데이터를 불러와 시각화에 사용할 GeoDataFrame으로 준비함.
    ensure_input_files_exist()
    boundary = gpd.read_parquet(BOUNDARY_PATH)
    grid = gpd.read_parquet(GRID_PATH)
    return boundary, grid


def save_preview(boundary: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> Path:
    # 격자는 회색 선으로, 강남구 경계는 빨간 선으로 표시하여 겹침 상태를 쉽게 확인함.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    grid.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.2)
    boundary.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1.5)

    ax.set_title("Gangnam 50m Grid Preview")
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_IMAGE_PATH


def main() -> int:
    # 전처리 결과 파일을 읽고 격자 미리보기 이미지를 생성함.
    try:
        boundary, grid = load_data()
        image_path = save_preview(boundary, grid)
        log(f"Boundary rows: {len(boundary)}")
        log(f"Grid rows: {len(grid)}")
        log(f"Saved preview image: {image_path}")
        return 0
    except Exception as exc:
        log(f"Preview generation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
