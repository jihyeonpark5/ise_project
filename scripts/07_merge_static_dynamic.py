from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd


# 프로젝트 기본 경로와 입출력 파일 경로를 정의한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STATIC_PATH = PROCESSED_DIR / "static_grid.parquet"
DYNAMIC_PATH = PROCESSED_DIR / "10_dynamic_variables.parquet"
OUTPUT_PATH = PROCESSED_DIR / "merged_grid_variables.parquet"
DYNAMIC_DATA_LABEL = DYNAMIC_PATH.stem

# 모든 공간 데이터는 EPSG:5179로 맞춰 사용한다.
TARGET_CRS = "EPSG:5179"
TARGET_EPSG = 5179

# 최종 결과에 포함할 컬럼 순서를 명시적으로 고정한다.
OUTPUT_COLUMNS = [
    "grid_id",
    "intersection_count",
    "road_type_score",
    "is_school_zone",
    "is_hospital_zone",
    "is_elderly_zone",
    "pedestrian_10h",
    "pedestrian_18h",
    "pedestrian_22h",
    "vehicle_10h",
    "vehicle_18h",
    "vehicle_22h",
    "pm_accident",
    "geometry",
]


def log(message: str) -> None:
    # 실행 상태와 검증 결과를 콘솔에 일관되게 출력한다.
    print(message)


def ensure_output_directory() -> None:
    # 출력 폴더가 없을 때 저장 오류가 나지 않도록 미리 생성한다.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def ensure_input_files_exist() -> None:
    # 병합에 필요한 두 입력 파일이 모두 있는지 먼저 확인한다.
    missing_files: list[str] = []
    for path in [STATIC_PATH, DYNAMIC_PATH]:
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            "Step 7 입력 파일이 없습니다. "
            f"누락 파일: {', '.join(missing_files)}"
        )


def ensure_target_crs(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    # 좌표계가 다르면 병합 후 공간 검증이 꼬일 수 있어 EPSG:5179로 통일한다.
    if gdf.crs is None:
        raise ValueError(f"{label} 데이터에 CRS 정보가 없습니다.")

    if gdf.crs.to_epsg() != TARGET_EPSG:
        log(f"{label} CRS를 {gdf.crs}에서 {TARGET_CRS}로 변환합니다.")
        return gdf.to_crs(TARGET_CRS)

    return gdf


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # 정적 변수와 동적 변수 파일을 읽고 기본 구조를 검증한다.
    ensure_input_files_exist()
    static_gdf = gpd.read_parquet(STATIC_PATH)
    dynamic_gdf = gpd.read_parquet(DYNAMIC_PATH)

    static_gdf = ensure_target_crs(static_gdf, "정적 변수")
    dynamic_gdf = ensure_target_crs(dynamic_gdf, "동적 변수")

    static_required = {"grid_id", "intersection_count", "road_type_score", "geometry"}
    dynamic_required = {
        "grid_id",
        "is_school_zone",
        "is_hospital_zone",
        "is_elderly_zone",
        "geometry",
        "pedestrian_10h",
        "pedestrian_18h",
        "pedestrian_22h",
        "vehicle_10h",
        "vehicle_18h",
        "vehicle_22h",
        "pm_accident",
    }

    missing_static = static_required - set(static_gdf.columns)
    missing_dynamic = dynamic_required - set(dynamic_gdf.columns)
    if missing_static:
        raise ValueError(f"정적 변수 파일에 필요한 컬럼이 없습니다: {sorted(missing_static)}")
    if missing_dynamic:
        raise ValueError(f"동적 변수 파일에 필요한 컬럼이 없습니다: {sorted(missing_dynamic)}")

    if static_gdf.empty:
        raise ValueError("정적 변수 파일이 비어 있습니다.")
    if dynamic_gdf.empty:
        raise ValueError("동적 변수 파일이 비어 있습니다.")

    return static_gdf, dynamic_gdf


def validate_unique_grid_ids(static_gdf: gpd.GeoDataFrame, dynamic_gdf: gpd.GeoDataFrame) -> None:
    # grid_id 기준 1:1 병합을 보장하기 위해 입력 단계에서 중복을 막는다.
    static_duplicates = int(static_gdf["grid_id"].duplicated().sum())
    dynamic_duplicates = int(dynamic_gdf["grid_id"].duplicated().sum())

    if static_duplicates:
        raise ValueError(f"정적 변수 파일에 중복 grid_id가 있습니다: {static_duplicates}개")
    if dynamic_duplicates:
        raise ValueError(f"동적 변수 파일에 중복 grid_id가 있습니다: {dynamic_duplicates}개")


def build_merged_grid(static_gdf: gpd.GeoDataFrame, dynamic_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # 최종 geometry는 정적 변수 파일의 격자 polygon만 사용하고,
    # 동적 변수 파일의 geometry는 병합 전에 제거한다.
    dynamic_features = dynamic_gdf.drop(columns=["geometry"], errors="ignore").copy()
    merged = static_gdf.merge(dynamic_features, on="grid_id", how="left", validate="1:1")

    return gpd.GeoDataFrame(
        merged[OUTPUT_COLUMNS],
        geometry="geometry",
        crs=static_gdf.crs,
    )


def validate_result(static_gdf: gpd.GeoDataFrame, dynamic_gdf: gpd.GeoDataFrame, result: gpd.GeoDataFrame) -> dict[str, int]:
    # 병합 전후 행 수, grid_id 차이, geometry 조건을 함께 검증한다.
    static_grid_ids = set(static_gdf["grid_id"])
    dynamic_grid_ids = set(dynamic_gdf["grid_id"])

    only_in_static = static_grid_ids - dynamic_grid_ids
    only_in_dynamic = dynamic_grid_ids - static_grid_ids

    if len(result) != len(static_gdf):
        raise ValueError(f"결과 행 수가 정적 변수 행 수와 다릅니다. 정적: {len(static_gdf)}, 결과: {len(result)}")

    if result.crs is None or result.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"결과 CRS가 {TARGET_CRS}가 아닙니다.")

    if not result.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("결과 geometry는 Polygon 또는 MultiPolygon이어야 합니다.")

    missing_output_columns = [column for column in OUTPUT_COLUMNS if column not in result.columns]
    if missing_output_columns:
        raise ValueError(f"결과 파일에 필요한 컬럼이 없습니다: {missing_output_columns}")

    return {
        "static_rows": len(static_gdf),
        "dynamic_rows": len(dynamic_gdf),
        "result_rows": len(result),
        "static_duplicates": int(static_gdf["grid_id"].duplicated().sum()),
        "dynamic_duplicates": int(dynamic_gdf["grid_id"].duplicated().sum()),
        "only_in_static": len(only_in_static),
        "only_in_dynamic": len(only_in_dynamic),
        "road_type_score_missing": int(result["road_type_score"].isna().sum()),
    }


def print_summary(summary: dict[str, int]) -> None:
    # 사용자가 바로 확인할 수 있도록 핵심 검증 결과를 보기 좋게 출력한다.
    log(f"static_grid 행 수: {summary['static_rows']}")
    log(f"{DYNAMIC_DATA_LABEL} 행 수: {summary['dynamic_rows']}")
    log(f"병합 결과 행 수: {summary['result_rows']}")
    log(f"static_grid 중복 grid_id 수: {summary['static_duplicates']}")
    log(f"{DYNAMIC_DATA_LABEL} 중복 grid_id 수: {summary['dynamic_duplicates']}")
    log(f"static에만 있는 grid_id 수: {summary['only_in_static']}")
    log(f"dynamic에만 있는 grid_id 수: {summary['only_in_dynamic']}")
    log(f"road_type_score 결측 개수: {summary['road_type_score_missing']}")


def save_result(result: gpd.GeoDataFrame) -> None:
    # 최종 병합 결과를 GeoParquet으로 저장해 geometry와 CRS를 함께 보존한다.
    ensure_output_directory()
    result.to_parquet(OUTPUT_PATH, index=False)
    log(f"병합 결과 저장 완료: {OUTPUT_PATH}")


def main() -> int:
    # Step 7 병합 전용 실행 흐름을 순차적으로 제어한다.
    log("=== Step 7: 정적·동적 변수 병합 ===")

    try:
        static_gdf, dynamic_gdf = load_inputs()
        validate_unique_grid_ids(static_gdf, dynamic_gdf)
        result = build_merged_grid(static_gdf, dynamic_gdf)
        summary = validate_result(static_gdf, dynamic_gdf, result)
        print_summary(summary)
        save_result(result)
        log("Step 7이 정상 완료되었습니다.")
        return 0
    except Exception as exc:
        log(f"Step 7 실행 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
