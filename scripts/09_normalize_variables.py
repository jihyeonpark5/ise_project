from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


# 프로젝트 경로와 주요 입출력 파일 경로를 정의한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "merged_grid_variables.parquet"
FULL_OUTPUT_PATH = PROCESSED_DIR / "final_grid_variables_normalized.parquet"
MODEL_OUTPUT_PATH = PROCESSED_DIR / "model_input_variables.parquet"

# 모든 공간 데이터는 EPSG:5179를 유지한다.
TARGET_CRS = "EPSG:5179"
TARGET_EPSG = 5179

# Min-Max 정규화 대상 연속형 변수 목록을 고정한다.
CONTINUOUS_COLUMNS = [
    "intersection_count",
    "road_type_score",
    "pedestrian_10h",
    "pedestrian_18h",
    "pedestrian_22h",
    "vehicle_10h",
    "vehicle_18h",
    "vehicle_22h",
]

# 0/1 상태를 그대로 유지할 이진 변수 목록이다.
BINARY_COLUMNS = [
    "is_school_zone",
    "is_hospital_zone",
    "is_elderly_zone",
    "pm_accident",
]

# 모델 학습 전용 파일에는 아래 컬럼만 남겨 입력 혼선을 줄인다.
MODEL_OUTPUT_COLUMNS = [
    "grid_id",
    "is_school_zone",
    "is_hospital_zone",
    "is_elderly_zone",
    "pm_accident",
    "intersection_count_norm",
    "road_type_score_norm",
    "pedestrian_10h_norm",
    "pedestrian_18h_norm",
    "pedestrian_22h_norm",
    "vehicle_10h_norm",
    "vehicle_18h_norm",
    "vehicle_22h_norm",
]

# 모든 정규화 컬럼은 동일한 자릿수로 반올림해 저장한다.
NORM_ROUND_DIGITS = 6


def log(message: str) -> None:
    # 실행 과정을 콘솔에서 쉽게 따라갈 수 있도록 단순 로그 함수를 둔다.
    print(message)


def ensure_input_file_exists() -> None:
    # 정규화 입력 파일이 없으면 먼저 Step 7 병합을 수행하도록 안내한다.
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "정규화 입력 파일이 없습니다. "
            "먼저 scripts/07_merge_static_dynamic.py를 실행해 주세요. "
            f"누락 파일: {INPUT_PATH}"
        )


def ensure_output_directory() -> None:
    # 출력 폴더가 없더라도 저장할 수 있도록 미리 생성한다.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def ensure_target_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # 좌표계가 다르면 후속 공간 분석에서 혼선이 생기므로 EPSG:5179로 맞춘다.
    if gdf.crs is None:
        raise ValueError("입력 파일에 CRS 정보가 없습니다.")

    if gdf.crs.to_epsg() != TARGET_EPSG:
        log(f"입력 CRS를 {gdf.crs}에서 {TARGET_CRS}로 변환합니다.")
        return gdf.to_crs(TARGET_CRS)

    return gdf


def load_input_data() -> gpd.GeoDataFrame:
    # 입력 파일을 읽고 필요한 컬럼과 기본 구조를 검증한다.
    ensure_input_file_exists()
    gdf = gpd.read_parquet(INPUT_PATH)
    gdf = ensure_target_crs(gdf)

    required_columns = {"grid_id", "geometry", *CONTINUOUS_COLUMNS, *BINARY_COLUMNS}
    missing_columns = required_columns - set(gdf.columns)
    if missing_columns:
        raise ValueError(f"입력 파일에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    if gdf.empty:
        raise ValueError("입력 파일이 비어 있습니다.")

    if gdf["grid_id"].duplicated().any():
        duplicate_count = int(gdf["grid_id"].duplicated().sum())
        raise ValueError(f"grid_id 중복이 있습니다: {duplicate_count}개")

    return gdf


def validate_binary_columns(gdf: gpd.GeoDataFrame) -> dict[str, list[int]]:
    # 이진 변수는 반드시 0/1만 가져야 하므로 실행 초기에 바로 검증한다.
    binary_summary: dict[str, list[int]] = {}

    for column in BINARY_COLUMNS:
        unique_values = sorted(pd.Series(gdf[column]).dropna().unique().tolist())
        binary_summary[column] = unique_values

        if not set(unique_values).issubset({0, 1}):
            raise ValueError(f"{column} 컬럼은 0/1 이진 변수여야 합니다. 현재 값: {unique_values}")

    return binary_summary


def build_normalized_columns(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, dict[str, float | int | bool]]]:
    # 원본 컬럼은 유지하고, 연속형 변수별 _norm 컬럼만 추가한다.
    normalized = gdf.copy()
    normalization_summary: dict[str, dict[str, float | int | bool]] = {}

    for column in CONTINUOUS_COLUMNS:
        norm_column = f"{column}_norm"
        series = pd.to_numeric(normalized[column], errors="coerce")

        min_value = series.min()
        max_value = series.max()
        is_constant = pd.notna(min_value) and pd.notna(max_value) and min_value == max_value

        if pd.isna(min_value) or pd.isna(max_value):
            # 값이 전부 결측이면 정규화 결과도 전부 결측으로 유지한다.
            normalized[norm_column] = pd.Series(pd.NA, index=normalized.index, dtype="float64")
        elif is_constant:
            # 분모가 0이 되는 경우 계획대로 전부 0.0으로 처리한다.
            normalized[norm_column] = series.where(series.isna(), 0.0)
        else:
            normalized[norm_column] = (series - min_value) / (max_value - min_value)

        normalization_summary[column] = {
            "min": float(min_value) if pd.notna(min_value) else float("nan"),
            "max": float(max_value) if pd.notna(max_value) else float("nan"),
            "na_count": int(series.isna().sum()),
            "is_constant": bool(is_constant),
        }

    # 부동소수점 계산 잔차를 줄이기 위해 모든 _norm 컬럼을 같은 자릿수로 반올림한다.
    norm_columns = [f"{column}_norm" for column in CONTINUOUS_COLUMNS]
    normalized[norm_columns] = normalized[norm_columns].round(NORM_ROUND_DIGITS)

    return normalized, normalization_summary


def build_model_input(full_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    # Risk Score 모델 학습에는 road_type_score가 있는 격자만 사용하고,
    # 원본 연속형 변수와 geometry는 제거한 최소 입력 테이블만 남긴다.
    model_input = full_gdf.loc[full_gdf["road_type_score"].notna()].copy()
    return pd.DataFrame(model_input[MODEL_OUTPUT_COLUMNS]).copy()


def validate_geometry(gdf: gpd.GeoDataFrame, label: str) -> None:
    # geometry는 50m 격자 polygon이어야 하므로 타입과 CRS를 함께 검사한다.
    if gdf.crs is None or gdf.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"{label} CRS가 {TARGET_CRS}가 아닙니다.")

    if not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"{label} geometry 타입은 Polygon 또는 MultiPolygon이어야 합니다.")


def validate_model_input_columns(model_df: pd.DataFrame) -> None:
    # 모델 입력 파일은 최소 컬럼 스키마만 가져야 한다.
    missing_columns = [column for column in MODEL_OUTPUT_COLUMNS if column not in model_df.columns]
    if missing_columns:
        raise ValueError(f"모델 입력 파일에 필요한 컬럼이 없습니다: {missing_columns}")

    extra_columns = [column for column in model_df.columns if column not in MODEL_OUTPUT_COLUMNS]
    if extra_columns:
        raise ValueError(f"모델 입력 파일에 불필요한 컬럼이 남아 있습니다: {extra_columns}")


def validate_norm_ranges(gdf: gpd.GeoDataFrame) -> dict[str, dict[str, float | int]]:
    # 각 _norm 컬럼이 0~1 범위에 있는지 확인하고 요약 통계를 모은다.
    range_summary: dict[str, dict[str, float | int]] = {}

    for column in CONTINUOUS_COLUMNS:
        norm_column = f"{column}_norm"
        series = pd.to_numeric(gdf[norm_column], errors="coerce")
        non_null = series.dropna()

        if non_null.empty:
            min_value = float("nan")
            max_value = float("nan")
            out_of_range_count = 0
        else:
            min_value = float(non_null.min())
            max_value = float(non_null.max())
            out_of_range_count = int(((non_null < 0) | (non_null > 1)).sum())

        range_summary[norm_column] = {
            "min": min_value,
            "max": max_value,
            "out_of_range_count": out_of_range_count,
            "na_count": int(series.isna().sum()),
        }

        if out_of_range_count > 0:
            raise ValueError(f"{norm_column} 컬럼에 0~1 범위를 벗어나는 값이 있습니다: {out_of_range_count}개")

    return range_summary


def save_outputs(full_gdf: gpd.GeoDataFrame, model_df: pd.DataFrame) -> None:
    # 전체 격자용 파일과 모델 학습용 파일을 각각 저장한다.
    ensure_output_directory()
    full_gdf.to_parquet(FULL_OUTPUT_PATH, index=False)
    model_df.to_parquet(MODEL_OUTPUT_PATH, index=False)

    log(f"전체 정규화 파일 저장 완료: {FULL_OUTPUT_PATH}")
    log(f"모델 입력 파일 저장 완료: {MODEL_OUTPUT_PATH}")


def print_summary(
    input_gdf: gpd.GeoDataFrame,
    full_gdf: gpd.GeoDataFrame,
    model_df: pd.DataFrame,
    normalization_summary: dict[str, dict[str, float | int | bool]],
    norm_range_summary: dict[str, dict[str, float | int]],
    binary_summary: dict[str, list[int]],
) -> None:
    # 계획에서 요구한 검증 출력을 한 번에 보기 좋게 정리한다.
    log(f"전체 입력 행 수: {len(input_gdf)}")
    log(f"전체 정규화 결과 행 수: {len(full_gdf)}")
    log(f"모델 학습용 데이터 행 수: {len(model_df)}")
    log(f"road_type_score NaN 행 수: {int(full_gdf['road_type_score'].isna().sum())}")

    log("연속형 변수 정규화 전 min/max:")
    for column in CONTINUOUS_COLUMNS:
        info = normalization_summary[column]
        log(
            f"  - {column}: min={info['min']:.6f}, max={info['max']:.6f}, "
            f"na={info['na_count']}, constant={info['is_constant']}"
        )

    log("_norm 컬럼 정규화 후 min/max:")
    for column in CONTINUOUS_COLUMNS:
        norm_column = f"{column}_norm"
        info = norm_range_summary[norm_column]
        min_text = f"{info['min']:.6f}" if pd.notna(info["min"]) else "NaN"
        max_text = f"{info['max']:.6f}" if pd.notna(info["max"]) else "NaN"
        log(
            f"  - {norm_column}: min={min_text}, max={max_text}, "
            f"out_of_range={info['out_of_range_count']}, na={info['na_count']}"
        )

    log("주요 결측 개수:")
    for column in ["road_type_score", "road_type_score_norm", *[f"{col}_norm" for col in CONTINUOUS_COLUMNS if col != "road_type_score"]]:
        if column in full_gdf.columns:
            log(f"  - {column}: {int(full_gdf[column].isna().sum())}")

    log("이진 변수 고유값 검증:")
    for column, values in binary_summary.items():
        log(f"  - {column}: {values}")

    log(f"CRS 확인: {full_gdf.crs.to_epsg()}")
    geom_types = full_gdf.geom_type.value_counts().to_dict()
    log(f"geometry 타입 확인: {geom_types}")
    log(f"모델 입력 컬럼: {list(model_df.columns)}")


def main() -> int:
    # Step 7 정규화 전체 흐름을 순차적으로 실행한다.
    log("=== Step 7: 연속형 변수 정규화 및 모델 입력 파일 생성 ===")

    try:
        input_gdf = load_input_data()
        binary_summary = validate_binary_columns(input_gdf)
        full_gdf, normalization_summary = build_normalized_columns(input_gdf)
        full_gdf = gpd.GeoDataFrame(full_gdf, geometry="geometry", crs=input_gdf.crs)
        model_df = build_model_input(full_gdf)

        validate_geometry(full_gdf, "전체 정규화 결과")
        validate_model_input_columns(model_df)

        norm_range_summary = validate_norm_ranges(full_gdf)

        if int(model_df["road_type_score_norm"].isna().sum()) != 0:
            raise ValueError("모델 입력 파일에는 road_type_score_norm 결측이 남아 있으면 안 됩니다.")

        print_summary(
            input_gdf=input_gdf,
            full_gdf=full_gdf,
            model_df=model_df,
            normalization_summary=normalization_summary,
            norm_range_summary=norm_range_summary,
            binary_summary=binary_summary,
        )
        save_outputs(full_gdf, model_df)
        log("Step 7 정규화가 정상 완료되었습니다.")
        return 0
    except Exception as exc:
        log(f"Step 7 정규화 실행 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
