from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

INPUT_PATH = PROCESSED_DIR / "10_dynamic_variables.parquet"
SUMMARY_PATH = OUTPUT_DIR / "pedestrian_distribution_summary.csv"
TOP_GRIDS_PATH = OUTPUT_DIR / "pedestrian_top_grids.csv"
ANOMALY_PATH = OUTPUT_DIR / "pedestrian_order_anomalies.csv"

PEDESTRIAN_COLUMNS = ["pedestrian_10h", "pedestrian_18h", "pedestrian_22h"]

## 보행량 분포를 검증하는 스크립트
def log(message: str) -> None:
    print(message)


def ensure_input_file_exists() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "보행량 검증 입력 파일이 없습니다. "
            f"누락 파일: {INPUT_PATH}"
        )


def ensure_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dynamic_data() -> gpd.GeoDataFrame:
    ensure_input_file_exists()
    gdf = gpd.read_parquet(INPUT_PATH)

    required_columns = {"grid_id", *PEDESTRIAN_COLUMNS}
    missing_columns = required_columns - set(gdf.columns)
    if missing_columns:
        raise ValueError(f"보행량 검증에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    if gdf.empty:
        raise ValueError("동적 변수 파일이 비어 있습니다.")

    return gdf


def build_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for column in PEDESTRIAN_COLUMNS:
        series = pd.to_numeric(gdf[column], errors="coerce")
        q1 = float(series.quantile(0.25))
        q2 = float(series.quantile(0.50))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr

        rows.append(
            {
                "column": column,
                "row_count": int(series.shape[0]),
                "na_count": int(series.isna().sum()),
                "zero_count": int((series == 0).sum()),
                "one_count": int((series == 1).sum()),
                "min": float(series.min()),
                "p25": q1,
                "median": q2,
                "p75": q3,
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "iqr": iqr,
                "outlier_threshold": outlier_threshold,
                "outlier_count": int((series > outlier_threshold).sum()),
            }
        )

    return pd.DataFrame(rows)


def build_top_grids(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    top_frames: list[pd.DataFrame] = []

    for column in PEDESTRIAN_COLUMNS:
        top = (
            gdf[["grid_id", *PEDESTRIAN_COLUMNS]]
            .sort_values(column, ascending=False)
            .head(20)
            .copy()
        )
        top.insert(0, "rank_source", column)
        top_frames.append(top)

    return pd.concat(top_frames, ignore_index=True)


def build_order_anomalies(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    # 일반적인 기대 패턴은 18h >= 10h >= 22h 이므로, 이 순서가 깨지는 격자를 따로 본다.
    check = gdf[["grid_id", *PEDESTRIAN_COLUMNS]].copy()
    anomaly_mask = ~(
        (check["pedestrian_18h"] >= check["pedestrian_10h"]) &
        (check["pedestrian_10h"] >= check["pedestrian_22h"])
    )
    anomalies = check.loc[anomaly_mask].copy()
    anomalies["pattern"] = anomalies.apply(
        lambda row: f"10h={row['pedestrian_10h']:.4f}, 18h={row['pedestrian_18h']:.4f}, 22h={row['pedestrian_22h']:.4f}",
        axis=1,
    )
    return anomalies.sort_values(["pedestrian_18h", "pedestrian_10h", "pedestrian_22h"], ascending=False)


def save_outputs(summary_df: pd.DataFrame, top_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> None:
    ensure_output_directory()
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    top_df.to_csv(TOP_GRIDS_PATH, index=False, encoding="utf-8-sig")
    anomaly_df.to_csv(ANOMALY_PATH, index=False, encoding="utf-8-sig")


def print_summary(summary_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> None:
    log("=== pedestrian 분포 검증 ===")

    for _, row in summary_df.iterrows():
        log(
            f"{row['column']}: "
            f"min={row['min']:.4f}, p25={row['p25']:.4f}, median={row['median']:.4f}, "
            f"p75={row['p75']:.4f}, max={row['max']:.4f}, mean={row['mean']:.4f}"
        )
        log(
            f"  zero={int(row['zero_count'])}, one={int(row['one_count'])}, na={int(row['na_count'])}, "
            f"outlier_threshold={row['outlier_threshold']:.4f}, outlier_count={int(row['outlier_count'])}"
        )

    log(f"시간대 순서 이상 격자 수(18h >= 10h >= 22h 위반): {len(anomaly_df)}")
    log(f"요약 CSV 저장: {SUMMARY_PATH}")
    log(f"상위 격자 CSV 저장: {TOP_GRIDS_PATH}")
    log(f"이상 패턴 CSV 저장: {ANOMALY_PATH}")


def main() -> int:
    try:
        gdf = load_dynamic_data()
        summary_df = build_summary(gdf)
        top_df = build_top_grids(gdf)
        anomaly_df = build_order_anomalies(gdf)
        save_outputs(summary_df, top_df, anomaly_df)
        print_summary(summary_df, anomaly_df)
        return 0
    except Exception as exc:
        log(f"pedestrian 분포 검증 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
