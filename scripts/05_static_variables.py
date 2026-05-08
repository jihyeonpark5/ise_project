from __future__ import annotations

import ast
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


# 프로젝트 루트와 주요 폴더를 Path 기준으로 정의한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 모든 공간 데이터는 미터 단위 분석에 적합한 EPSG:5179로 통일한다.
TARGET_CRS = "EPSG:5179"
TARGET_EPSG = 5179

# 교차로 정의에 사용할 최소 degree 기준값이다.
INTERSECTION_DEGREE_THRESHOLD = 3

# road_type_score 계산에 사용할 highway 점수표다.
HIGHWAY_SCORE_MAP = {
    "trunk": 110,
    "trunk_link": 100,
    "primary": 100,
    "primary_link": 90,
    "secondary": 80,
    "secondary_link": 75,
    "tertiary": 60,
    "tertiary_link": 55,
    "unclassified": 45,
    "residential": 40,
    "service": 30,
    "living_street": 25,
    "track": 25,
    "cycleway": 20,
    "path": 20,
}

# 사용자 프롬프트 기준 파일명과 현재 저장소 파일명을 모두 지원한다.
GRID_INPUT_CANDIDATES = [
    PROCESSED_DIR / "gangnam_grid_50m.parquet",
    PROCESSED_DIR / "gangnam_grid_50m.geojson",
    PROCESSED_DIR / "02_gangnam_grid_50m.parquet",
    PROCESSED_DIR / "03_gangnam_grid_50m.geojson",
]
GRID_ROAD_MAP_CANDIDATES = [
    PROCESSED_DIR / "grid_road_map.parquet",
    PROCESSED_DIR / "08_grid_road_map.parquet",
]
NODES_INPUT_CANDIDATES = [
    PROCESSED_DIR / "osm_nodes_raw.parquet",
    PROCESSED_DIR / "04_osm_nodes_raw.parquet",
]
EDGES_INPUT_CANDIDATES = [
    PROCESSED_DIR / "osm_edges_filtered.parquet",
    PROCESSED_DIR / "06_osm_edges_filtered.parquet",
]
OUTPUT_PATH = PROCESSED_DIR / "static_grid.parquet"


def log(message: str) -> None:
    # 실행 상태를 콘솔에서 바로 확인할 수 있도록 단순한 로그 함수를 둔다.
    print(message)


def pick_existing_path(candidates: list[Path], label: str) -> Path:
    # 후보 경로 중 실제로 존재하는 첫 번째 파일을 선택한다.
    for path in candidates:
        if path.exists():
            return path

    candidate_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{label} 파일을 찾을 수 없습니다. 확인한 경로: {candidate_text}")


def ensure_output_directory() -> None:
    # 출력 폴더가 없으면 자동으로 만들어 저장 오류를 방지한다.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def ensure_target_crs(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    # 모든 공간 연산을 같은 좌표계에서 수행하기 위해 CRS를 검사하고 필요 시 변환한다.
    if gdf.crs is None:
        raise ValueError(f"{label} 데이터에 CRS 정보가 없습니다.")

    if gdf.crs.to_epsg() != TARGET_EPSG:
        log(f"{label} CRS를 {gdf.crs}에서 {TARGET_CRS}로 변환합니다.")
        return gdf.to_crs(TARGET_CRS)

    return gdf


def load_geodataframe(path: Path) -> gpd.GeoDataFrame:
    # 확장자에 따라 parquet 또는 벡터 파일을 읽는다.
    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def load_grid() -> tuple[gpd.GeoDataFrame, Path]:
    # 기준 격자는 반드시 50m polygon이어야 하므로 가장 먼저 불러와 검증한다.
    grid_path = pick_existing_path(GRID_INPUT_CANDIDATES, "격자")
    grid = load_geodataframe(grid_path)
    grid = ensure_target_crs(grid, "격자")

    if "grid_id" not in grid.columns:
        raise ValueError("격자 데이터에 grid_id 컬럼이 없습니다.")

    if grid.empty:
        raise ValueError("격자 데이터가 비어 있습니다.")

    if not grid.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("격자 geometry는 Polygon 또는 MultiPolygon이어야 합니다.")

    return grid[["grid_id", "geometry"]].copy(), grid_path


def load_nodes() -> tuple[gpd.GeoDataFrame, Path]:
    # 교차로 위치는 노드 데이터에서 가져오므로 node geometry가 반드시 필요하다.
    nodes_path = pick_existing_path(NODES_INPUT_CANDIDATES, "OSM 노드")
    nodes = gpd.read_parquet(nodes_path)
    nodes = ensure_target_crs(nodes, "OSM 노드")

    required_columns = {"osmid", "geometry"}
    missing_columns = required_columns - set(nodes.columns)
    if missing_columns:
        raise ValueError(f"OSM 노드 데이터에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    if nodes.empty:
        raise ValueError("OSM 노드 데이터가 비어 있습니다.")

    return nodes[["osmid", "geometry"]].copy(), nodes_path


def load_edges() -> tuple[pd.DataFrame, Path]:
    # 교차로 정의에 사용할 degree는 필터링된 도로의 u, v 연결 수로 계산한다.
    edges_path = pick_existing_path(EDGES_INPUT_CANDIDATES, "필터링된 OSM 도로")
    edges = gpd.read_parquet(edges_path)
    edges = ensure_target_crs(edges, "필터링된 OSM 도로")

    required_columns = {"u", "v"}
    missing_columns = required_columns - set(edges.columns)
    if missing_columns:
        raise ValueError(f"필터링된 OSM 도로 데이터에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    if edges.empty:
        raise ValueError("필터링된 OSM 도로 데이터가 비어 있습니다.")

    return pd.DataFrame(edges[["u", "v"]]).copy(), edges_path


def load_grid_road_map() -> tuple[gpd.GeoDataFrame, Path]:
    # 도로 유형 점수는 격자-도로 매핑 결과를 사용해 계산한다.
    road_map_path = pick_existing_path(GRID_ROAD_MAP_CANDIDATES, "격자-도로 매핑")
    road_map = gpd.read_parquet(road_map_path)
    road_map = ensure_target_crs(road_map, "격자-도로 매핑")

    required_columns = {"grid_id", "highway", "length_in_grid"}
    missing_columns = required_columns - set(road_map.columns)
    if missing_columns:
        raise ValueError(f"격자-도로 매핑 데이터에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    return road_map[["grid_id", "highway", "length_in_grid"]].copy(), road_map_path


def normalize_highway_values(value: object) -> list[str]:
    # highway 값은 단일 문자열, 결측치, 리스트 문자열 등 여러 형태로 들어올 수 있다.
    if value is None or pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        normalized_values: list[str] = []
        for item in value:
            item_text = str(item).strip()
            if not item_text:
                continue

            # 리스트 안 원소도 "service, residential"처럼 합쳐져 있을 수 있어 다시 분해한다.
            split_values = [part.strip().strip("\"'").lower() for part in item_text.split(",")]
            normalized_values.extend([part for part in split_values if part and part not in {"nan", "none", "null", "<na>"}])
        return normalized_values

    text = str(value).strip()
    if not text:
        return []

    lowered = text.lower()
    if lowered in {"nan", "none", "null", "<na>"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None

        if isinstance(parsed, (list, tuple, set)):
            return normalize_highway_values(list(parsed))

    # 일반 문자열이어도 "service, residential"처럼 쉼표로 여러 값이 붙어 있을 수 있다.
    split_values = [part.strip().strip("\"'").lower() for part in text.split(",")]
    return [part for part in split_values if part and part not in {"nan", "none", "null", "<na>"}]


def get_segment_highway_score(highway_value: object) -> float | None:
    # 한 도로 segment에 highway 유형이 여러 개면 점수 평균을 segment 점수로 사용한다.
    highway_values = normalize_highway_values(highway_value)
    scores = [HIGHWAY_SCORE_MAP[item] for item in highway_values if item in HIGHWAY_SCORE_MAP]

    if not scores:
        return None

    return float(sum(scores) / len(scores))


def build_intersection_count(grid: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    # u, v를 세어 node별 degree를 만들고, degree가 3 이상인 노드만 교차로로 본다.
    node_endpoints = pd.concat([edges["u"], edges["v"]], ignore_index=True)
    degree_table = node_endpoints.value_counts(dropna=True).rename_axis("osmid").reset_index(name="degree")
    intersection_ids = degree_table.loc[degree_table["degree"] >= INTERSECTION_DEGREE_THRESHOLD, "osmid"]

    intersection_nodes = nodes.loc[nodes["osmid"].isin(intersection_ids)].copy()
    if intersection_nodes.empty:
        return pd.DataFrame({"grid_id": grid["grid_id"], "intersection_count": 0})

    joined = gpd.sjoin(
        intersection_nodes[["osmid", "geometry"]],
        grid[["grid_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        return pd.DataFrame({"grid_id": grid["grid_id"], "intersection_count": 0})

    counts = (
        joined[["grid_id", "osmid"]]
        .drop_duplicates()
        .groupby("grid_id", as_index=False)
        .agg(intersection_count=("osmid", "nunique"))
    )

    result = pd.DataFrame({"grid_id": grid["grid_id"]}).merge(counts, on="grid_id", how="left")
    result["intersection_count"] = result["intersection_count"].fillna(0).astype("int64")
    return result


def build_road_type_score(road_map: gpd.GeoDataFrame) -> pd.DataFrame:
    # 길이가 0 이하인 도로는 가중 평균 계산에서 제외한다.
    score_base = road_map.copy()
    score_base["length_in_grid"] = pd.to_numeric(score_base["length_in_grid"], errors="coerce")
    score_base = score_base.loc[score_base["length_in_grid"] > 0].copy()

    if score_base.empty:
        return pd.DataFrame(columns=["grid_id", "road_type_score"])

    score_base["segment_score"] = score_base["highway"].apply(get_segment_highway_score)
    score_base = score_base.dropna(subset=["segment_score"]).copy()

    if score_base.empty:
        return pd.DataFrame(columns=["grid_id", "road_type_score"])

    score_base["weighted_score"] = score_base["segment_score"] * score_base["length_in_grid"]
    grouped = (
        score_base.groupby("grid_id", as_index=False)
        .agg(weighted_score_sum=("weighted_score", "sum"), length_sum=("length_in_grid", "sum"))
    )
    grouped["road_type_score"] = grouped["weighted_score_sum"] / grouped["length_sum"]
    return grouped[["grid_id", "road_type_score"]]


def analyze_road_type_score_missing(
    grid: gpd.GeoDataFrame,
    road_map: gpd.GeoDataFrame,
    result: gpd.GeoDataFrame,
) -> dict[str, object]:
    # road_type_score 결측의 원인을 도로 없음 / 점수 계산 불가로 나눠서 집계한다.
    total_grid_count = len(grid)
    road_type_missing_grid_ids = set(result.loc[result["road_type_score"].isna(), "grid_id"])

    road_map_base = road_map.copy()
    road_map_base["length_in_grid"] = pd.to_numeric(road_map_base["length_in_grid"], errors="coerce")
    road_map_base = road_map_base.loc[road_map_base["length_in_grid"] > 0].copy()

    grid_ids_in_road_map = set(road_map_base["grid_id"].dropna().astype(str))
    no_road_grid_ids = set(grid["grid_id"].astype(str)) - grid_ids_in_road_map
    road_exists_but_nan_grid_ids = road_type_missing_grid_ids & grid_ids_in_road_map

    unknown_highway_counter: dict[str, int] = {}
    for highway_value in road_map_base["highway"]:
        for highway_name in normalize_highway_values(highway_value):
            if highway_name in HIGHWAY_SCORE_MAP:
                continue
            unknown_highway_counter[highway_name] = unknown_highway_counter.get(highway_name, 0) + 1

    unknown_highway_counts = sorted(
        unknown_highway_counter.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return {
        "total_grid_count": total_grid_count,
        "road_type_score_nan_count": len(road_type_missing_grid_ids),
        "no_road_grid_count": len(no_road_grid_ids),
        "road_exists_but_nan_grid_count": len(road_exists_but_nan_grid_ids),
        "unknown_highway_counts": unknown_highway_counts,
    }


def build_static_grid() -> gpd.GeoDataFrame:
    # 기준이 되는 격자 geometry를 중심으로 정적 변수 컬럼을 순서대로 붙인다.
    grid, grid_path = load_grid()
    nodes, nodes_path = load_nodes()
    edges, edges_path = load_edges()
    road_map, road_map_path = load_grid_road_map()

    log(f"사용한 격자 파일: {grid_path}")
    log(f"사용한 OSM 노드 파일: {nodes_path}")
    log(f"사용한 OSM 도로 파일: {edges_path}")
    log(f"사용한 격자-도로 매핑 파일: {road_map_path}")

    intersection_count = build_intersection_count(grid, nodes, edges)
    road_type_score = build_road_type_score(road_map)

    result = grid.merge(intersection_count, on="grid_id", how="left")
    result = result.merge(road_type_score, on="grid_id", how="left")

    result["intersection_count"] = result["intersection_count"].fillna(0).astype("int64")
    result["road_type_score"] = pd.to_numeric(result["road_type_score"], errors="coerce")

    return gpd.GeoDataFrame(
        result[["grid_id", "intersection_count", "road_type_score", "geometry"]],
        geometry="geometry",
        crs=TARGET_CRS,
    )


def validate_result(source_grid: gpd.GeoDataFrame, result: gpd.GeoDataFrame) -> None:
    # 결과 파일의 핵심 조건을 마지막에 다시 확인해 후속 분석 오류를 줄인다.
    if len(result) != len(source_grid):
        raise ValueError(f"결과 행 수가 격자 행 수와 다릅니다. 격자: {len(source_grid)}, 결과: {len(result)}")

    if result.crs is None or result.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"결과 CRS가 {TARGET_CRS}가 아닙니다.")

    if not result.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("최종 geometry는 격자 polygon이어야 합니다.")

    if result["intersection_count"].isna().any():
        raise ValueError("intersection_count에 결측치가 있으면 안 됩니다.")


def print_metric_summary(result: gpd.GeoDataFrame) -> None:
    # 사용자가 바로 확인할 수 있도록 요청한 통계를 실행 직후 출력한다.
    log(f"기존 격자 수: {len(result)}")
    log(f"결과 static_grid 행 수: {len(result)}")
    log(
        "intersection_count min/max/mean: "
        f"{result['intersection_count'].min()} / "
        f"{result['intersection_count'].max()} / "
        f"{result['intersection_count'].mean():.4f}"
    )

    road_type_non_null = result["road_type_score"].dropna()
    if road_type_non_null.empty:
        log("road_type_score min/max/mean: 모두 NaN")
    else:
        log(
            "road_type_score min/max/mean: "
            f"{road_type_non_null.min():.4f} / "
            f"{road_type_non_null.max():.4f} / "
            f"{road_type_non_null.mean():.4f}"
        )

    log(f"road_type_score 결측 개수: {int(result['road_type_score'].isna().sum())}")


def print_road_type_missing_analysis(analysis: dict[str, object]) -> None:
    # road_type_score 결측 원인을 사람이 읽기 쉽게 별도 블록으로 출력한다.
    log("=== road_type_score 결측 원인 분석 ===")
    log(f"전체 격자 수: {analysis['total_grid_count']}")
    log(f"road_type_score가 NaN인 격자 수: {analysis['road_type_score_nan_count']}")
    log(f"grid_road_map에 아예 등장하지 않는 grid_id 수: {analysis['no_road_grid_count']}")
    log(
        "grid_road_map에는 등장하지만 road_type_score가 NaN인 grid_id 수: "
        f"{analysis['road_exists_but_nan_grid_count']}"
    )

    unknown_highway_counts = analysis["unknown_highway_counts"]
    if not unknown_highway_counts:
        log("highway 점수표에 없는 highway 값: 없음")
        return

    log("highway 점수표에 없는 highway 값 목록과 빈도:")
    for highway_name, count in unknown_highway_counts:
        display_name = highway_name if highway_name else "<empty>"
        log(f"  - {display_name}: {count}")


def save_result(result: gpd.GeoDataFrame) -> None:
    # 최종 결과는 GeoParquet으로 저장해 geometry와 CRS를 함께 보존한다.
    ensure_output_directory()
    result.to_parquet(OUTPUT_PATH, index=False)
    log(f"정적 변수 결과 저장 완료: {OUTPUT_PATH}")


def main() -> int:
    # Step 5 정적 변수 생성의 전체 실행 흐름을 순차적으로 제어한다.
    log("=== Step 5: 정적 변수 생성 ===")

    try:
        source_grid, _ = load_grid()
        road_map, _ = load_grid_road_map()
        result = build_static_grid()
        validate_result(source_grid, result)
        print_metric_summary(result)
        missing_analysis = analyze_road_type_score_missing(source_grid, road_map, result)
        print_road_type_missing_analysis(missing_analysis)
        save_result(result)
        log("Step 5가 정상 완료되었습니다.")
        return 0
    except Exception as exc:
        log(f"Step 5 실행 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
