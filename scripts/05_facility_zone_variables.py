from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd


# 프로젝트 루트 경로를 기준으로 입력/출력 파일 위치를 일관되게 관리하기 위함.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FACILITIES_DIR = PROJECT_ROOT / "data" / "raw" / "facilities"

# Step 5 입력 파일과 출력 파일 경로를 정의함.
GRID_PATH = PROCESSED_DIR / "02_gangnam_grid_50m.parquet"
SCHOOL_BUFFER_PATH = FACILITIES_DIR / "school_buffer.gpkg"
HOSPITAL_BUFFER_PATH = FACILITIES_DIR / "hospital_buffer.gpkg"
ELDERLY_BUFFER_PATH = FACILITIES_DIR / "elderly_buffer.gpkg"
OUTPUT_PATH = PROCESSED_DIR / "09_facility_zone_grid.parquet"

# 모든 공간 연산은 미터 단위 좌표계인 EPSG:5179에서 수행함.
TARGET_CRS = "EPSG:5179"
TARGET_EPSG = 5179


def log(message: str) -> None:
    # 실행 과정을 콘솔에 일관된 형식으로 출력하기 위함.
    print(message)


def ensure_input_files_exist() -> None:
    # 보호구역 변수 생성에 필요한 격자와 buffer 파일이 모두 있는지 먼저 확인함.
    missing_files: list[str] = []
    for path in [GRID_PATH, SCHOOL_BUFFER_PATH, HOSPITAL_BUFFER_PATH, ELDERLY_BUFFER_PATH]:
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            "Step 5 입력 파일이 없습니다. "
            "격자 파일과 시설 buffer 파일을 확인해주세요. "
            f"누락 파일: {', '.join(missing_files)}"
        )


def ensure_output_directory() -> None:
    # 전처리 결과 저장 폴더가 없더라도 자동으로 생성하여 저장 오류를 방지함.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def ensure_target_crs(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    # GeoDataFrame의 CRS를 확인하고, EPSG:5179가 아니면 변환함.
    # 보호구역 포함 여부는 거리/면적 기반 공간 연산이므로 좌표계 통일이 중요함.
    if gdf.crs is None:
        raise ValueError(f"{label} 데이터에 CRS 정보가 없습니다.")

    if gdf.crs.to_epsg() != TARGET_EPSG:
        log(f"{label} CRS를 {gdf.crs}에서 {TARGET_CRS}로 변환함.")
        return gdf.to_crs(TARGET_CRS)

    return gdf


def load_grid() -> gpd.GeoDataFrame:
    # 50m 격자를 불러오고 Step 5 결과의 기준 geometry로 사용함.
    grid = gpd.read_parquet(GRID_PATH)
    grid = ensure_target_crs(grid, "격자")

    if "grid_id" not in grid.columns:
        raise ValueError("격자 데이터에 grid_id 컬럼이 없습니다.")

    if grid.empty:
        raise ValueError("격자 데이터가 비어 있습니다.")

    if not grid.geom_type.eq("Polygon").all():
        raise ValueError("격자 geometry는 Polygon이어야 합니다.")

    return grid[["grid_id", "geometry"]].copy()


def load_buffer(path: Path, label: str) -> gpd.GeoDataFrame:
    # 시설 buffer 파일을 읽고 polygon geometry만 공간 연산에 사용함.
    buffer_gdf = gpd.read_file(path)
    buffer_gdf = ensure_target_crs(buffer_gdf, label)

    if buffer_gdf.empty:
        raise ValueError(f"{label} buffer 데이터가 비어 있습니다.")

    if not buffer_gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"{label} buffer geometry는 Polygon 또는 MultiPolygon이어야 합니다.")

    buffer_gdf = buffer_gdf.copy()
    buffer_gdf["geometry"] = buffer_gdf.geometry.make_valid()
    return buffer_gdf[["geometry"]]


def mark_intersection(grid: gpd.GeoDataFrame, buffer_gdf: gpd.GeoDataFrame, column_name: str) -> gpd.GeoDataFrame:
    # 같은 유형의 buffer를 하나의 geometry로 통합해 중복 buffer가 있어도 포함 여부만 안정적으로 계산함.
    # 중심점이 아니라 격자 polygon과 buffer polygon이 실제로 겹치는지를 intersects로 판단함.
    output = grid.copy()
    merged_buffer = buffer_gdf.geometry.union_all()
    output[column_name] = output.geometry.intersects(merged_buffer).astype("int64")
    return output


def build_facility_zone_grid() -> gpd.GeoDataFrame:
    # 격자 기준으로 학교, 병원, 노인시설 buffer 포함 여부 변수를 순서대로 생성함.
    ensure_input_files_exist()
    grid = load_grid()

    school_buffer = load_buffer(SCHOOL_BUFFER_PATH, "학교 보호구역")
    hospital_buffer = load_buffer(HOSPITAL_BUFFER_PATH, "병원 인접 구역")
    elderly_buffer = load_buffer(ELDERLY_BUFFER_PATH, "노인시설 보호구역")

    result = mark_intersection(grid, school_buffer, "is_school_zone")
    result = mark_intersection(result, hospital_buffer, "is_hospital_zone")
    result = mark_intersection(result, elderly_buffer, "is_elderly_zone")
    return result[["grid_id", "is_school_zone", "is_hospital_zone", "is_elderly_zone", "geometry"]]


def validate_result(result: gpd.GeoDataFrame, original_grid_count: int) -> None:
    # 저장 전 결과 행 수, CRS, geometry, 변수값 범위를 확인하여 후속 통합 오류를 줄임.
    if len(result) != original_grid_count:
        raise ValueError(f"결과 격자 수가 원본과 다릅니다. 원본: {original_grid_count}, 결과: {len(result)}")

    if result.crs is None or result.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"결과 CRS가 {TARGET_CRS}가 아닙니다.")

    if not result.geom_type.eq("Polygon").all():
        raise ValueError("결과 geometry는 격자 Polygon이어야 합니다.")

    for column in ["is_school_zone", "is_hospital_zone", "is_elderly_zone"]:
        unique_values = set(result[column].dropna().unique().tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"{column} 컬럼에는 0 또는 1만 들어가야 합니다. 현재 값: {sorted(unique_values)}")

        if result[column].isna().any():
            raise ValueError(f"{column} 컬럼에 결측치가 있습니다.")


def save_result(result: gpd.GeoDataFrame) -> None:
    # 보호구역 포함 여부 변수가 붙은 격자 파일을 GeoParquet 형식으로 저장함.
    ensure_output_directory()
    result.to_parquet(OUTPUT_PATH, index=False)
    log(f"보호구역 변수 parquet 저장 완료: {OUTPUT_PATH}")


def print_summary(result: gpd.GeoDataFrame, original_grid_count: int) -> None:
    # 검증에 필요한 핵심 통계를 콘솔에 출력함.
    log(f"기존 격자 수: {original_grid_count}")
    log(f"결과 격자 수: {len(result)}")
    log(f"is_school_zone = 1 격자 수: {int(result['is_school_zone'].sum())}")
    log(f"is_hospital_zone = 1 격자 수: {int(result['is_hospital_zone'].sum())}")
    log(f"is_elderly_zone = 1 격자 수: {int(result['is_elderly_zone'].sum())}")
    log(f"is_school_zone 결측 개수: {int(result['is_school_zone'].isna().sum())}")
    log(f"is_hospital_zone 결측 개수: {int(result['is_hospital_zone'].isna().sum())}")
    log(f"is_elderly_zone 결측 개수: {int(result['is_elderly_zone'].isna().sum())}")


def main() -> int:
    # Step 5 전체 실행 흐름을 순차적으로 제어함.
    # 격자/시설 buffer 로드, CRS 통일, polygon 중첩 여부 계산, 검증, 저장 순서로 진행함.
    log("=== Step 5: 보호구역 변수 컬럼 생성 ===")

    try:
        original_grid_count = len(load_grid())
        result = build_facility_zone_grid()
        validate_result(result, original_grid_count)
        print_summary(result, original_grid_count)
        save_result(result)
        log("Step 5가 정상 완료되었습니다.")
        return 0
    except Exception as exc:
        log(f"Step 5 실행 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
