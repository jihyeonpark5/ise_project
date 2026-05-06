from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


# 프로젝트 루트 경로를 기준으로 입력/출력 파일 위치를 일관되게 관리하기 위함.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Step 2와 Step 3에서 생성한 파일을 입력으로 사용함.
GRID_PATH = PROCESSED_DIR / "02_gangnam_grid_50m.parquet"
EDGES_PATH = PROCESSED_DIR / "06_osm_edges_filtered.parquet"

# 격자와 도로의 교차 관계를 저장할 최종 출력 파일 경로를 정의함.
OUTPUT_PATH = PROCESSED_DIR / "08_grid_road_map.parquet"

# 모든 공간 연산은 미터 단위 좌표계인 EPSG:5179에서 수행함.
TARGET_CRS = "EPSG:5179"
TARGET_EPSG = 5179


def log(message: str) -> None:
    # 실행 과정을 콘솔에 일관된 형식으로 출력하기 위함.
    print(message)


def ensure_input_files_exist() -> None:
    # Step 4는 Step 2 격자와 Step 3 필터링 도로망이 모두 준비되어 있어야 실행 가능함.
    missing_files: list[str] = []

    if not GRID_PATH.exists():
        missing_files.append(str(GRID_PATH))
    if not EDGES_PATH.exists():
        missing_files.append(str(EDGES_PATH))

    if missing_files:
        raise FileNotFoundError(
            "Step 4 입력 파일이 없습니다. "
            "먼저 Step 2와 Step 3 스크립트를 실행해주세요. "
            f"누락 파일: {', '.join(missing_files)}"
        )


def ensure_output_directory() -> None:
    # 전처리 결과 저장 폴더가 없더라도 자동으로 생성하여 저장 오류를 방지함.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def ensure_target_crs(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    # GeoDataFrame의 CRS를 확인하고, EPSG:5179가 아니면 변환함.
    # parquet에서 읽은 CRS는 문자열 표현이 길 수 있으므로 to_epsg()로 EPSG 코드를 확인함.
    if gdf.crs is None:
        raise ValueError(f"{label} 데이터에 CRS 정보가 없습니다.")

    if gdf.crs.to_epsg() != TARGET_EPSG:
        log(f"{label} CRS를 {gdf.crs}에서 {TARGET_CRS}로 변환함.")
        return gdf.to_crs(TARGET_CRS)

    return gdf


def normalize_osm_value(value: object) -> object:
    # OSM 속성값은 문자열, 리스트를 직렬화한 문자열, 결측치 등 다양한 형태로 들어올 수 있음.
    # 저장 오류를 줄이고 사람이 읽기 쉽게 하기 위해 리스트형 값은 쉼표로 연결한 문자열로 정리함.
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NA

    if isinstance(value, (list, tuple, set)):
        cleaned = [str(item).strip() for item in value if item is not None and str(item).strip()]
        return ", ".join(cleaned) if cleaned else pd.NA

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "nan", "null", "<na>"}:
            return pd.NA

        # Step 3 저장 과정에서 리스트가 JSON 문자열로 바뀐 경우를 다시 읽기 쉬운 문자열로 정리함.
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None

            if isinstance(parsed, (list, tuple, set)):
                return normalize_osm_value(parsed)

        return text

    if pd.isna(value):
        return pd.NA

    return str(value).strip()


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # 격자와 필터링된 도로망을 불러오고, 두 데이터의 CRS를 EPSG:5179로 통일함.
    ensure_input_files_exist()
    grid = gpd.read_parquet(GRID_PATH)
    edges = gpd.read_parquet(EDGES_PATH)

    grid = ensure_target_crs(grid, "격자")
    edges = ensure_target_crs(edges, "도로")

    if "grid_id" not in grid.columns:
        raise ValueError("격자 데이터에 grid_id 컬럼이 없습니다.")

    if grid.empty:
        raise ValueError("격자 데이터가 비어 있습니다.")

    if edges.empty:
        raise ValueError("도로 edge 데이터가 비어 있습니다.")

    return grid, edges


def prepare_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # edge_id가 없으면 현재 edge 행 순서를 기준으로 새 식별자를 생성함.
    # OSM에서 온 u, v, key 컬럼이 있더라도 이후 격자 매핑에서는 단일 edge_id가 있으면 다루기 쉬움.
    edges = edges.copy().reset_index(drop=True)

    if "edge_id" not in edges.columns:
        edges["edge_id"] = [f"E{idx:06d}" for idx in range(1, len(edges) + 1)]
    else:
        edges["edge_id"] = edges["edge_id"].astype("string")

    for column in ["highway", "access", "width"]:
        if column not in edges.columns:
            edges[column] = pd.NA
        edges[column] = edges[column].apply(normalize_osm_value).astype("string")

    return edges


def build_grid_road_map(grid: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # spatial join으로 격자와 도로가 만나는 후보 쌍을 먼저 찾음.
    # 모든 edge와 모든 grid를 직접 비교하는 반복문보다 공간 인덱스를 활용하는 방식이 훨씬 효율적임.
    grid_for_join = grid[["grid_id", "geometry"]].copy().reset_index(drop=True)
    grid_for_join["grid_index"] = grid_for_join.index

    edge_columns = ["edge_id", "highway", "access", "width", "geometry"]
    edges_for_join = edges[edge_columns].copy()

    joined = gpd.sjoin(edges_for_join, grid_for_join, how="inner", predicate="intersects")
    if joined.empty:
        return gpd.GeoDataFrame(
            columns=["grid_id", "edge_id", "highway", "access", "width", "length_in_grid", "geometry"],
            geometry="geometry",
            crs=TARGET_CRS,
        )

    # spatial join 결과는 edge geometry만 유지하므로, grid geometry를 다시 붙여 실제 교차 선분을 계산함.
    joined = joined.drop(columns=["index_right"], errors="ignore").reset_index(drop=True)
    grid_geometry_by_index = grid_for_join.set_index("grid_index")["geometry"]
    joined["grid_geometry"] = joined["grid_index"].map(grid_geometry_by_index)

    edge_geometries = gpd.GeoSeries(joined.geometry, crs=TARGET_CRS)
    grid_geometries = gpd.GeoSeries(joined["grid_geometry"], crs=TARGET_CRS)

    # edge와 grid polygon의 실제 교차 geometry를 계산함.
    # 이 geometry의 길이가 해당 도로가 해당 격자 내부에서 차지하는 길이임.
    intersections = edge_geometries.intersection(grid_geometries, align=False)
    result = gpd.GeoDataFrame(joined.drop(columns=["geometry", "grid_geometry", "grid_index"]), geometry=intersections, crs=TARGET_CRS)
    result["length_in_grid"] = result.geometry.length

    # 경계선에 점으로만 닿은 경우 길이가 0이므로 분석 대상에서 제외함.
    result = result.loc[result["length_in_grid"] > 0].copy()
    result = result[["grid_id", "edge_id", "highway", "access", "width", "length_in_grid", "geometry"]]
    return result.reset_index(drop=True)


def validate_result(result: gpd.GeoDataFrame) -> None:
    # 저장 전 결과 CRS와 길이 컬럼 상태를 확인하여 후속 분석 오류를 줄임.
    if result.crs is None or result.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"결과 CRS가 {TARGET_CRS}가 아닙니다.")

    if result.empty:
        raise ValueError("격자-도로 매핑 결과가 비어 있습니다.")

    if result["length_in_grid"].isna().any():
        raise ValueError("length_in_grid 컬럼에 결측치가 있습니다.")

    if (result["length_in_grid"] <= 0).any():
        raise ValueError("length_in_grid가 0 이하인 행이 포함되어 있습니다.")


def save_result(result: gpd.GeoDataFrame) -> None:
    # 격자-도로 매핑 결과를 GeoParquet 형식으로 저장함.
    ensure_output_directory()
    result.to_parquet(OUTPUT_PATH, index=False)
    log(f"격자-도로 매핑 parquet 저장 완료: {OUTPUT_PATH}")


def print_summary(grid: gpd.GeoDataFrame, result: gpd.GeoDataFrame) -> None:
    # 전체 매핑 행 수와 도로 포함 여부별 격자 수를 출력하여 결과 규모를 빠르게 확인함.
    mapped_grid_count = result["grid_id"].nunique()
    total_grid_count = grid["grid_id"].nunique()
    empty_grid_count = total_grid_count - mapped_grid_count

    log(f"전체 매핑 행 수: {len(result)}")
    log(f"도로가 포함된 격자 수: {mapped_grid_count}")
    log(f"도로가 없는 격자 수: {empty_grid_count}")


def main() -> int:
    # Step 4 전체 실행 흐름을 순차적으로 제어함.
    # 격자/도로 로드, CRS 통일, 교차 선분 계산, 길이 산정, 결과 저장 순서로 진행함.
    log("=== Step 4: 격자-도로 매핑 생성 ===")

    try:
        grid, edges = load_inputs()
        edges = prepare_edges(edges)
        result = build_grid_road_map(grid, edges)
        validate_result(result)
        print_summary(grid, result)
        save_result(result)
        log("Step 4가 정상 완료되었습니다.")
        return 0
    except Exception as exc:
        log(f"Step 4 실행 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
