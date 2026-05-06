from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import json
import math
import sys

import geopandas as gpd
import pandas as pd


# 프로젝트 루트 경로를 기준으로 입력/출력 파일 위치를 일관되게 관리하기 위함.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Step 3-1 산출물을 입력으로 사용하고, 필터링 결과를 별도 파일로 저장하기 위함.
RAW_EDGES_PATH = PROCESSED_DIR / "05_osm_edges_raw.parquet"
FILTERED_EDGES_PARQUET_PATH = PROCESSED_DIR / "06_osm_edges_filtered.parquet"
FILTERED_EDGES_GEOJSON_PATH = PROCESSED_DIR / "07_osm_edges_filtered.geojson"
TARGET_CRS = "EPSG:5179"

# PM 주행 부적절 도로로 간주할 기준값을 미리 정의함.
EXCLUDED_HIGHWAY_VALUES = {"footway", "steps", "pedestrian"}
EXCLUDED_ACCESS_VALUES = {"private", "no"}


def log(message: str) -> None:
    # 실행 과정을 콘솔에 일관된 형식으로 출력하기 위함.
    print(message)


def ensure_input_file_exists() -> None:
    # 원본 edge 파일이 없으면 필터링을 진행할 수 없으므로 먼저 존재 여부를 확인함.
    if not RAW_EDGES_PATH.exists():
        raise FileNotFoundError(
            "원본 OSM edges 파일이 없습니다. "
            "먼저 scripts/02_collect_osm_roads.py를 실행해주세요. "
            f"누락 파일: {RAW_EDGES_PATH}"
        )


def ensure_output_directory() -> None:
    # 전처리 결과 저장 폴더가 없더라도 자동으로 생성하여 저장 오류를 방지함.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_to_list(value: object) -> list[str]:
    # OSM 속성값은 문자열, 리스트, 튜플, 결측치 등 여러 형태로 들어올 수 있으므로 일관된 리스트로 변환함.
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    if isinstance(value, (list, tuple, set)):
        normalized = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip().lower()
            if text:
                normalized.append(text)
        return normalized

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        lowered = text.lower()
        if lowered in {"none", "nan", "null"}:
            return []

        # 수집 단계에서 리스트를 JSON 문자열로 저장했을 수 있으므로 다시 리스트로 복원함.
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None

            if isinstance(parsed, (list, tuple, set)):
                return normalize_to_list(list(parsed))

        return [lowered]

    if pd.isna(value):
        return []

    text = str(value).strip().lower()
    return [text] if text else []


def load_edges() -> gpd.GeoDataFrame:
    # 원본 edge 데이터를 불러오고 CRS와 geometry 상태를 먼저 확인함.
    ensure_input_file_exists()
    edges = gpd.read_parquet(RAW_EDGES_PATH)

    if edges.crs is None:
        raise ValueError("원본 OSM edges에 CRS 정보가 없습니다.")

    if str(edges.crs) != TARGET_CRS:
        edges = edges.to_crs(TARGET_CRS)

    if edges.empty:
        raise ValueError("원본 OSM edges가 비어 있습니다.")

    if edges.geometry.is_empty.any():
        raise ValueError("원본 OSM edges에 빈 geometry가 포함되어 있습니다.")

    return edges


def build_filter_columns(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # highway와 access 속성을 리스트 형태로 정규화하여 문자열/리스트 혼합 문제를 안정적으로 처리함.
    edges = edges.copy()
    edges["highway_values"] = edges.get("highway", pd.Series(index=edges.index, dtype="object")).apply(normalize_to_list)
    edges["access_values"] = edges.get("access", pd.Series(index=edges.index, dtype="object")).apply(normalize_to_list)

    # 제외 조건에 해당하는 값이 하나라도 포함되면 제거 대상으로 표시함.
    edges["exclude_by_highway"] = edges["highway_values"].apply(
        lambda values: any(value in EXCLUDED_HIGHWAY_VALUES for value in values)
    )
    edges["exclude_by_access"] = edges["access_values"].apply(
        lambda values: any(value in EXCLUDED_ACCESS_VALUES for value in values)
    )
    edges["exclude_candidate"] = edges["exclude_by_highway"] | edges["exclude_by_access"]
    return edges


def count_removed_values(edges: gpd.GeoDataFrame, column_name: str, excluded_values: set[str]) -> dict[str, int]:
    # 제거된 edge들 중 어떤 속성값 때문에 제외되었는지 유형별 개수를 집계함.
    counter: Counter[str] = Counter()
    for values in edges[column_name]:
        for value in values:
            if value in excluded_values:
                counter[value] += 1

    # 제거 건수가 없는 유형도 0으로 채워 항상 같은 형식으로 출력되게 함.
    return {key: counter.get(key, 0) for key in sorted(excluded_values)}


def filter_edges(edges: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, int], dict[str, int]]:
    # PM 주행이 부적절한 후보 구간을 제외하고, 제거 전후 통계를 함께 계산함.
    edges = build_filter_columns(edges)
    removed_edges = edges.loc[edges["exclude_candidate"]].copy()
    filtered_edges = edges.loc[~edges["exclude_candidate"]].copy()

    removed_highway_counts = count_removed_values(removed_edges, "highway_values", EXCLUDED_HIGHWAY_VALUES)
    removed_access_counts = count_removed_values(removed_edges, "access_values", EXCLUDED_ACCESS_VALUES)

    # 분석용 helper 컬럼은 저장 전에 제거하여 결과 파일을 단순하게 유지함.
    helper_columns = [
        "highway_values",
        "access_values",
        "exclude_by_highway",
        "exclude_by_access",
        "exclude_candidate",
    ]
    filtered_edges = filtered_edges.drop(columns=helper_columns, errors="ignore")
    return filtered_edges, removed_highway_counts, removed_access_counts


def validate_filtered_edges(filtered_edges: gpd.GeoDataFrame) -> None:
    # 저장 전 필터링 결과가 최소 품질 조건을 만족하는지 확인함.
    if str(filtered_edges.crs) != TARGET_CRS:
        raise ValueError(f"필터링된 edges CRS가 일치하지 않습니다. 예상: {TARGET_CRS}, 실제: {filtered_edges.crs}")

    if filtered_edges.empty:
        raise ValueError("PM 도로 필터링 후 남은 OSM edges가 없습니다.")

    if filtered_edges.geometry.is_empty.any():
        raise ValueError("필터링된 OSM edges에 빈 geometry가 포함되어 있습니다.")


def save_outputs(filtered_edges: gpd.GeoDataFrame) -> None:
    # 필터링 결과를 parquet과 geojson 형식으로 모두 저장하여 후속 분석과 시각 확인에 활용함.
    ensure_output_directory()
    filtered_edges.to_parquet(FILTERED_EDGES_PARQUET_PATH, index=False)
    filtered_edges.to_file(FILTERED_EDGES_GEOJSON_PATH, driver="GeoJSON")

    log(f"필터링된 edges parquet 저장 완료: {FILTERED_EDGES_PARQUET_PATH}")
    log(f"필터링된 edges geojson 저장 완료: {FILTERED_EDGES_GEOJSON_PATH}")


def print_summary(before_count: int, after_count: int, highway_counts: dict[str, int], access_counts: dict[str, int]) -> None:
    # 필터링 전후 edge 개수와 제거 사유별 개수를 한눈에 볼 수 있도록 출력함.
    log(f"필터링 전 edge 개수: {before_count}")
    log(f"필터링 후 edge 개수: {after_count}")
    log(f"제거된 edge 개수: {before_count - after_count}")

    log("highway 유형별 제거 개수:")
    for key, value in highway_counts.items():
        log(f"  - {key}: {value}")

    log("access 유형별 제거 개수:")
    for key, value in access_counts.items():
        log(f"  - {key}: {value}")


def main() -> int:
    # Step 3-2 전체 실행 흐름을 순차적으로 제어함.
    # 원본 edge 로드, PM 부적절 구간 필터링, 결과 검증, 통계 출력, 파일 저장 순서로 진행함.
    log("=== Step 3-2: PM 주행 부적절 도로 필터링 ===")

    try:
        raw_edges = load_edges()
        filtered_edges, highway_counts, access_counts = filter_edges(raw_edges)
        validate_filtered_edges(filtered_edges)
        print_summary(len(raw_edges), len(filtered_edges), highway_counts, access_counts)
        save_outputs(filtered_edges)
        log("Step 3-2가 정상 완료되었습니다.")
        return 0
    except Exception as exc:
        log(f"Step 3-2 실행 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
