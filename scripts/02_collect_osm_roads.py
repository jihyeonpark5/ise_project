from __future__ import annotations

import json
from pathlib import Path
import os
import sys

# osmnx import 시 matplotlib 캐시 권한 문제를 줄이기 위해 경로를 먼저 고정함.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

import geopandas as gpd
import osmnx as ox


# 프로젝트 루트 경로를 기준으로 입력/출력 파일 위치를 일관되게 관리하기 위함.
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# OSM 도로망 수집 대상 지역명을 고정하여 실행 방식을 단순하게 유지함.
PLACE_NAME = "Gangnam-gu, Seoul, South Korea"
NETWORK_TYPE = "bike"
TARGET_CRS = "EPSG:5179"

# Step 3 원본 도로망 저장 경로를 미리 정의함.
NODES_OUTPUT_PATH = PROCESSED_DIR / "04_osm_nodes_raw.parquet"
EDGES_OUTPUT_PATH = PROCESSED_DIR / "05_osm_edges_raw.parquet"


def log(message: str) -> None:
    # 실행 과정을 콘솔에 일관된 형식으로 출력하기 위함.
    print(message)


def ensure_output_directory() -> None:
    # 전처리 결과 저장 폴더가 없더라도 자동으로 생성하여 저장 오류를 방지함.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def collect_osm_graph() -> ox.MultiDiGraph:
    # OSMnx로 지정된 지역의 bike network를 수집함.
    # 이 단계는 인터넷 연결과 OSM 응답 상태에 영향을 받으므로 예외 메시지를 명확히 출력함.
    try:
        log(f"OSM bike 도로망 수집 시작: {PLACE_NAME}")
        graph = ox.graph_from_place(PLACE_NAME, network_type=NETWORK_TYPE)
        return graph
    except Exception as exc:
        raise RuntimeError(
            "OSMnx 도로망 수집에 실패했습니다. "
            "인터넷 연결, OSM 서비스 상태, 수집 지역명 설정을 확인해주세요. "
            f"원본 오류: {exc}"
        ) from exc


def graph_to_projected_gdfs(graph: ox.MultiDiGraph) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # 수집한 그래프를 nodes와 edges GeoDataFrame으로 분리함.
    # 이후 모든 거리 기반 분석을 위해 반드시 EPSG:5179로 변환함.
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    if nodes.crs is None or edges.crs is None:
        raise ValueError("수집된 OSM 데이터에 CRS 정보가 없습니다.")

    nodes = nodes.to_crs(TARGET_CRS)
    edges = edges.to_crs(TARGET_CRS)

    # MultiIndex를 일반 컬럼으로 풀어 저장성과 후속 join 편의성을 높임.
    nodes = nodes.reset_index()
    edges = edges.reset_index()
    return nodes, edges


def serialize_mixed_object_value(value: object) -> object:
    # OSM 속성은 리스트, 튜플, 집합, 딕셔너리 등 혼합 형태로 들어올 수 있으므로 저장 가능한 값으로 직렬화함.
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    return value


def prepare_for_storage(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # parquet 저장 시 혼합 object 컬럼 때문에 오류가 날 수 있으므로 저장 전에 값을 정리함.
    prepared = gdf.copy()
    for column in prepared.columns:
        if column == prepared.geometry.name:
            continue

        if prepared[column].dtype == "object":
            prepared[column] = prepared[column].apply(serialize_mixed_object_value)
            prepared[column] = prepared[column].astype("string")

    return prepared


def validate_outputs(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> None:
    # 저장 전 최소 품질 조건을 확인하여 후속 분석 오류를 줄임.
    if str(nodes.crs) != TARGET_CRS:
        raise ValueError(f"nodes CRS가 일치하지 않습니다. 예상: {TARGET_CRS}, 실제: {nodes.crs}")

    if str(edges.crs) != TARGET_CRS:
        raise ValueError(f"edges CRS가 일치하지 않습니다. 예상: {TARGET_CRS}, 실제: {edges.crs}")

    if nodes.empty:
        raise ValueError("수집된 OSM nodes가 비어 있습니다.")

    if edges.empty:
        raise ValueError("수집된 OSM edges가 비어 있습니다.")

    if nodes.geometry.is_empty.any() or edges.geometry.is_empty.any():
        raise ValueError("수집된 OSM 데이터에 빈 geometry가 포함되어 있습니다.")


def save_outputs(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> None:
    # 원본 nodes와 edges를 parquet 형식으로 저장하여 후속 필터링 단계의 입력으로 사용함.
    ensure_output_directory()
    prepared_nodes = prepare_for_storage(nodes)
    prepared_edges = prepare_for_storage(edges)
    prepared_nodes.to_parquet(NODES_OUTPUT_PATH, index=False)
    prepared_edges.to_parquet(EDGES_OUTPUT_PATH, index=False)

    log(f"원본 nodes parquet 저장 완료: {NODES_OUTPUT_PATH}")
    log(f"원본 edges parquet 저장 완료: {EDGES_OUTPUT_PATH}")


def main() -> int:
    # Step 3-1 전체 실행 흐름을 순차적으로 제어함.
    # OSM 수집, nodes/edges 분리, 좌표계 변환, 결과 검증, 파일 저장 순서로 진행함.
    log("=== Step 3-1: OSM bike 도로망 수집 ===")

    try:
        graph = collect_osm_graph()
        nodes, edges = graph_to_projected_gdfs(graph)
        log(f"수집된 node 개수: {len(nodes)}")
        log(f"수집된 edge 개수: {len(edges)}")
        validate_outputs(nodes, edges)
        save_outputs(nodes, edges)
        log("Step 3-1이 정상 완료되었습니다.")
        return 0
    except Exception as exc:
        log(f"Step 3-1 실행 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
