from __future__ import annotations

from pathlib import Path
import math
import os
import sys

import geopandas as gpd
from shapely.geometry import box


# 프로젝트 루트 경로를 기준으로 입력/출력 파일 위치를 일관되게 관리하기 위함.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 원천 행정경계 Shapefile 입력 경로를 고정하여 실행 방식을 단순하게 유지함.
RAW_BOUNDARY_PATH = PROJECT_ROOT / "data" / "raw" / "boundary" / "seoul_sig.shp"

# 전처리 결과를 저장할 폴더와 파일 경로를 미리 정의함.
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BOUNDARY_OUTPUT_PATH = PROCESSED_DIR / "01_gangnam_boundary.parquet"
GRID_PARQUET_OUTPUT_PATH = PROCESSED_DIR / "02_gangnam_grid_50m.parquet"
GRID_GEOJSON_OUTPUT_PATH = PROCESSED_DIR / "03_gangnam_grid_50m.geojson"

# 공간 연산은 모두 미터 단위 좌표계에서 수행하기 위해 EPSG:5179를 기준으로 사용함.
TARGET_CRS = "EPSG:5179"
GRID_SIZE_METERS = 50

# Shapefile을 읽기 위해 함께 필요한 기본 구성 파일 목록을 정의함.
EXPECTED_SHAPEFILE_SUFFIXES = [".shp", ".shx", ".dbf", ".prj"]

# 원천 데이터마다 컬럼명이 다를 수 있어 이름/코드 후보 컬럼을 함께 정의함.
NAME_COLUMN_CANDIDATES = ["SIG_KOR_NM", "SIGUNGU_NM"]
CODE_COLUMN_CANDIDATES = ["SIG_CD", "SIGUNGU_CD"]
GANGNAM_NAME = "강남구"
GANGNAM_CODE_CANDIDATES = {"11680", "11230"}

# matplotlib 캐시 경로를 프로젝트 내부로 고정하여 권한 문제를 줄이기 위함.
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"

# geopandas가 내부적으로 matplotlib 관련 기능을 건드릴 때 캐시 권한 오류를 방지하기 위함.
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    # 실행 과정을 콘솔에 일관된 형식으로 출력하기 위함.
    print(message)


def ensure_input_files_exist() -> None:
    # Shapefile 입력에 필요한 기본 구성 파일이 모두 존재하는지 확인함.
    # .shp만 있어도 보이는 경우가 있지만, 실제 읽기 과정에서는 .shx, .dbf, .prj가 필요함.
    # 이 단계를 먼저 수행하여 파일 누락으로 인한 모호한 오류를 사전에 방지함.
    missing_files = []
    for suffix in EXPECTED_SHAPEFILE_SUFFIXES:
        path = RAW_BOUNDARY_PATH.with_suffix(suffix)
        if not path.exists():
            missing_files.append(path.name)

    if missing_files:
        # 누락된 파일명을 함께 안내하여 사용자가 바로 입력 데이터를 보완할 수 있게 함.
        expected = ", ".join(missing_files)
        raise FileNotFoundError(
            "Missing required shapefile sidecar files. "
            f"Expected in {RAW_BOUNDARY_PATH.parent}: {expected}"
        )


def select_name_column(columns: list[str]) -> str | None:
    # 강남구 이름을 담고 있을 가능성이 높은 컬럼 후보를 순서대로 확인함.
    for candidate in NAME_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def select_code_column(columns: list[str]) -> str | None:
    # 강남구 행정코드를 담고 있을 가능성이 높은 컬럼 후보를 순서대로 확인함.
    for candidate in CODE_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def load_boundary_source() -> gpd.GeoDataFrame:
    # 원천 행정경계 파일을 읽고, 이후 공간 연산에 필요한 최소 조건을 먼저 검증함.
    # 특히 CRS 정보가 없으면 거리 기반 분석을 안전하게 진행할 수 없으므로 즉시 중단함.
    ensure_input_files_exist()
    gdf = gpd.read_file(RAW_BOUNDARY_PATH)

    if gdf.crs is None:
        raise ValueError(
            "Input shapefile has no CRS information. "
            "Please verify the source boundary data and its .prj file."
        )

    log(f"Input file: {RAW_BOUNDARY_PATH}")
    log(f"Source CRS: {gdf.crs}")
    log(f"Source rows: {len(gdf)}")
    return gdf


def extract_gangnam_boundary(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # 전체 자치구 경계 데이터에서 강남구 경계만 안정적으로 추출함.
    # 이름 컬럼을 우선 사용하고, 원천 데이터 형식 차이를 대비해 코드 컬럼도 fallback으로 사용함.
    columns = list(gdf.columns)
    name_column = select_name_column(columns)
    code_column = select_code_column(columns)

    selected = None
    selection_reason = ""

    if name_column is not None:
        # 가장 직관적인 방식인 한글 자치구명 기준으로 먼저 강남구를 찾음.
        selected = gdf[gdf[name_column].astype(str).str.strip() == GANGNAM_NAME].copy()
        selection_reason = f"{name_column} == '{GANGNAM_NAME}'"

    # 원천 데이터마다 코드 체계가 다를 수 있어 강남구 대표 코드를 fallback으로 허용함.
    # 이름 기반 추출이 실패한 경우에만 코드 기반 추출을 수행하여 우선순위를 명확히 유지함.
    if (selected is None or selected.empty) and code_column is not None:
        selected = gdf[gdf[code_column].astype(str).str.strip().isin(GANGNAM_CODE_CANDIDATES)].copy()
        selection_reason = f"{code_column} in {sorted(GANGNAM_CODE_CANDIDATES)}"

    if selected is None:
        # 이름/코드 후보 컬럼 자체를 찾지 못한 경우, 지원 가능한 컬럼 목록을 함께 안내함.
        raise ValueError(
            "Could not find a supported district name/code column. "
            f"Checked name columns {NAME_COLUMN_CANDIDATES} and code columns {CODE_COLUMN_CANDIDATES}. "
            f"Available columns: {columns}"
        )

    if selected.empty:
        # 컬럼은 존재하지만 강남구 값이 없는 경우, 어떤 조건으로 찾았는지 로그로 남김.
        raise ValueError(
            "Gangnam-gu was not found in the boundary data. "
            f"Checked selection rule: {selection_reason}"
        )

    log(f"Selection rule: {selection_reason}")
    log(f"Matched rows before dissolve: {len(selected)}")

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one Gangnam-gu row in source data. "
            f"Found {len(selected)} rows using {selection_reason}."
        )

    # 이후 거리, buffer, 격자 생성 연산을 모두 미터 단위로 맞추기 위해 기준 좌표계로 변환함.
    # geometry 손상이 있는 경우를 대비해 make_valid()를 적용하여 후속 공간 연산 오류를 줄임.
    selected = selected.to_crs(TARGET_CRS)
    selected["geometry"] = selected.geometry.make_valid()

    # 여러 조각으로 나뉜 geometry가 있을 수 있으므로 하나의 경계 geometry로 통합함.
    dissolved_geometry = selected.geometry.union_all()

    # 이후 단계에서 사용하기 쉬운 단일 강남구 경계 GeoDataFrame 형식으로 재구성함.
    boundary = gpd.GeoDataFrame(
        {"district_name": [GANGNAM_NAME], "source_rule": [selection_reason]},
        geometry=[dissolved_geometry],
        crs=TARGET_CRS,
    )

    # 통합 이후에도 geometry 유효성을 한 번 더 보정하여 저장 안정성을 높임.
    boundary["geometry"] = boundary.geometry.make_valid()
    return boundary


def build_grid(boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # 강남구 경계의 bounding box를 기준으로 50m 정사각형 격자를 생성함.
    # 먼저 전체 후보 격자를 만들고, 이후 경계와 실제로 닿는 격자만 남기는 방식으로 처리함.
    boundary_geometry = boundary_gdf.geometry.iloc[0]
    min_x, min_y, max_x, max_y = boundary_geometry.bounds

    # 실행할 때마다 같은 결과가 나오도록 격자 시작/끝 좌표를 50m 단위에 맞춰 정렬함.
    # 이렇게 하면 경계가 조금 달라져도 동일한 기준점 위에서 격자를 생성할 수 있음.
    start_x = math.floor(min_x / GRID_SIZE_METERS) * GRID_SIZE_METERS
    start_y = math.floor(min_y / GRID_SIZE_METERS) * GRID_SIZE_METERS
    end_x = math.ceil(max_x / GRID_SIZE_METERS) * GRID_SIZE_METERS
    end_y = math.ceil(max_y / GRID_SIZE_METERS) * GRID_SIZE_METERS

    log(
        "Grid bounds (aligned): "
        f"min_x={start_x}, min_y={start_y}, max_x={end_x}, max_y={end_y}"
    )

    # bounding box 전체를 덮는 50m 사각형 후보 격자를 이중 반복문으로 생성함.
    cells = []
    x = start_x
    while x < end_x:
        y = start_y
        while y < end_y:
            cells.append(box(x, y, x + GRID_SIZE_METERS, y + GRID_SIZE_METERS))
            y += GRID_SIZE_METERS
        x += GRID_SIZE_METERS

    grid = gpd.GeoDataFrame({"geometry": cells}, crs=TARGET_CRS)

    # 강남구 경계와 한 점이라도 닿는 격자만 유지함.
    # 이후 분석에서는 격자 형태를 일관되게 유지해야 하므로 clip하지 않고 원본 사각형을 그대로 사용함.
    intersects_mask = grid.geometry.intersects(boundary_geometry)
    grid = grid.loc[intersects_mask].copy().reset_index(drop=True)

    # 격자별 고유 식별자를 고정 길이 문자열로 부여하여 이후 join과 추적을 쉽게 함.
    grid["grid_id"] = [f"G{idx:06d}" for idx in range(1, len(grid) + 1)]

    # 중심점은 참고용 좌표로만 저장함.
    # 이후 보호구역 포함 여부 등은 중심점이 아니라 polygon 중첩 기준으로 판단하는 것을 전제로 함.
    centroids = grid.geometry.centroid
    grid["centroid_x"] = centroids.x
    grid["centroid_y"] = centroids.y

    # 분석과 저장에 필요한 핵심 컬럼만 남겨 출력 형식을 단순하게 유지함.
    grid = grid[["grid_id", "geometry", "centroid_x", "centroid_y"]]
    return grid


def validate_outputs(boundary_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> None:
    # 저장 직전 결과물이 최소 품질 조건을 만족하는지 확인함.
    # CRS, 빈 geometry, geometry 타입, centroid 결측치 여부를 점검하여 후속 단계 오류를 줄임.
    if str(boundary_gdf.crs) != TARGET_CRS:
        raise ValueError(f"Boundary CRS mismatch. Expected {TARGET_CRS}, got {boundary_gdf.crs}")

    if str(grid_gdf.crs) != TARGET_CRS:
        raise ValueError(f"Grid CRS mismatch. Expected {TARGET_CRS}, got {grid_gdf.crs}")

    # 빈 geometry가 있으면 공간 연산이나 저장 과정에서 오류가 발생할 수 있으므로 차단함.
    if boundary_gdf.geometry.is_empty.any() or grid_gdf.geometry.is_empty.any():
        raise ValueError("Detected empty geometry in output data.")

    # 경계는 polygon 또는 multipolygon이어야 하며, grid는 정사각형 polygon만 허용함.
    if not boundary_gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Boundary output contains non-polygon geometry.")

    if not grid_gdf.geom_type.eq("Polygon").all():
        raise ValueError("Grid output contains non-polygon geometry.")

    # 중심점 좌표가 비어 있으면 이후 속성 결합이나 시각화에서 문제가 생기므로 확인함.
    if grid_gdf[["centroid_x", "centroid_y"]].isna().any().any():
        raise ValueError("Detected missing centroid coordinates in grid output.")


def save_outputs(boundary_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> None:
    # 결과 저장 폴더를 먼저 보장한 뒤, boundary와 grid를 지정된 형식으로 저장함.
    # parquet은 후속 분석용, geojson은 시각 확인이나 외부 도구 연동용으로 함께 저장함.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    boundary_gdf.to_parquet(BOUNDARY_OUTPUT_PATH, index=False)
    grid_gdf.to_parquet(GRID_PARQUET_OUTPUT_PATH, index=False)
    grid_gdf.to_file(GRID_GEOJSON_OUTPUT_PATH, driver="GeoJSON")

    log(f"Saved boundary parquet: {BOUNDARY_OUTPUT_PATH}")
    log(f"Saved grid parquet: {GRID_PARQUET_OUTPUT_PATH}")
    log(f"Saved grid geojson: {GRID_GEOJSON_OUTPUT_PATH}")


def main() -> int:
    # Step 2 전체 실행 흐름을 순차적으로 제어함.
    # 입력 확인, 경계 추출, 격자 생성, 결과 검증, 파일 저장 순서로 진행함.
    log("=== Step 2: Gangnam Boundary and 50m Grid Builder ===")

    try:
        source_gdf = load_boundary_source()
        boundary_gdf = extract_gangnam_boundary(source_gdf)
        grid_gdf = build_grid(boundary_gdf)
        log(f"Generated grid count: {len(grid_gdf)}")
        validate_outputs(boundary_gdf, grid_gdf)
        save_outputs(boundary_gdf, grid_gdf)
        log("Step 2 completed successfully.")
        return 0
    except Exception as exc:
        log(f"Step 2 failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
