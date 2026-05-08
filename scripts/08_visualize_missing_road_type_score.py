from __future__ import annotations

from pathlib import Path
import sys

import folium
import geopandas as gpd


# 프로젝트 경로와 입출력 파일 경로를 정의한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

INPUT_PATH = PROCESSED_DIR / "merged_grid_variables.parquet"
OUTPUT_GEOJSON_PATH = OUTPUT_DIR / "missing_road_type_score_grids.geojson"
OUTPUT_HTML_PATH = OUTPUT_DIR / "missing_road_type_score_map.html"

# 분석은 EPSG:5179에서 읽고, 웹 지도는 EPSG:4326으로 변환해 표시한다.
TARGET_CRS = "EPSG:5179"
WEB_CRS = "EPSG:4326"


def log(message: str) -> None:
    print(message)


def ensure_input_file_exists() -> None:
    # 시각화 대상 파일이 없으면 먼저 Step 7 병합 결과를 만들도록 안내한다.
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "병합 결과 파일이 없습니다. "
            "먼저 scripts/07_merge_static_dynamic.py를 실행해 주세요. "
            f"누락 파일: {INPUT_PATH}"
        )


def ensure_output_directory() -> None:
    # 출력 폴더가 없으면 자동으로 만든다.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_merged_grid() -> gpd.GeoDataFrame:
    # 병합된 격자 파일을 읽고 필요한 컬럼과 좌표계를 검증한다.
    ensure_input_file_exists()
    merged = gpd.read_parquet(INPUT_PATH)

    if merged.crs is None:
        raise ValueError("병합 결과 파일에 CRS 정보가 없습니다.")

    if merged.crs.to_string() != TARGET_CRS and merged.crs.to_epsg() != 5179:
        merged = merged.to_crs(TARGET_CRS)

    required_columns = {"grid_id", "road_type_score", "geometry"}
    missing_columns = required_columns - set(merged.columns)
    if missing_columns:
        raise ValueError(f"병합 결과 파일에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    if merged.empty:
        raise ValueError("병합 결과 파일이 비어 있습니다.")

    return merged


def extract_missing_grids(merged: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # road_type_score가 결측인 격자만 따로 추출한다.
    missing = merged.loc[merged["road_type_score"].isna(), ["grid_id", "road_type_score", "geometry"]].copy()

    if missing.empty:
        raise ValueError("road_type_score가 결측인 격자가 없습니다.")

    return missing


def save_geojson(missing: gpd.GeoDataFrame) -> None:
    # QGIS 같은 도구에서도 바로 열 수 있도록 GeoJSON을 저장한다.
    missing.to_file(OUTPUT_GEOJSON_PATH, driver="GeoJSON")
    log(f"결측 격자 GeoJSON 저장 완료: {OUTPUT_GEOJSON_PATH}")


def save_html_map(missing: gpd.GeoDataFrame) -> None:
    # 웹 브라우저에서 보기 쉬운 HTML 지도를 생성한다.
    missing_web = missing.to_crs(WEB_CRS)
    center = missing_web.geometry.union_all().centroid

    fmap = folium.Map(location=[center.y, center.x], zoom_start=13, tiles="CartoDB positron")

    # 결측 격자를 반투명 빨간색으로 강조한다.
    folium.GeoJson(
        missing_web,
        name="Missing road_type_score grids",
        style_function=lambda _: {
            "fillColor": "#d73027",
            "color": "#b2182b",
            "weight": 1,
            "fillOpacity": 0.55,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["grid_id"],
            aliases=["grid_id"],
            sticky=False,
        ),
    ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    fmap.save(str(OUTPUT_HTML_PATH))
    log(f"결측 격자 HTML 지도 저장 완료: {OUTPUT_HTML_PATH}")


def print_summary(merged: gpd.GeoDataFrame, missing: gpd.GeoDataFrame) -> None:
    # 생성 결과와 결측 비율을 콘솔에서 바로 확인할 수 있게 출력한다.
    total_count = len(merged)
    missing_count = len(missing)
    missing_ratio = (missing_count / total_count) * 100

    log(f"전체 격자 수: {total_count}")
    log(f"road_type_score 결측 격자 수: {missing_count}")
    log(f"road_type_score 결측 비율: {missing_ratio:.2f}%")


def main() -> int:
    # 병합 결과를 읽고 결측 격자만 추출해 시각화 파일을 저장한다.
    log("=== road_type_score 결측 격자 시각화 ===")

    try:
        ensure_output_directory()
        merged = load_merged_grid()
        missing = extract_missing_grids(merged)
        print_summary(merged, missing)
        save_geojson(missing)
        save_html_map(missing)
        log("시각화 파일 생성이 완료되었습니다.")
        return 0
    except Exception as exc:
        log(f"시각화 생성 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
