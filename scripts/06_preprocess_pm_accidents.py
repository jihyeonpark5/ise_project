from __future__ import annotations

from pathlib import Path
import json
import sys

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

## TAAS PM 사고 데이터를 50m 격자에 매핑해 pm_accident 타깃을 실제 사고값으로 갱신하는 스크립트 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ACCIDENT_PATH = PROJECT_ROOT / "data" / "raw" / "accident" / "taas_pm_accidents.json"
GRID_PATH = PROJECT_ROOT / "data" / "processed" / "02_gangnam_grid_50m.parquet"
DYNAMIC_PATH = PROJECT_ROOT / "data" / "processed" / "10_dynamic_variables.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_GEOJSON_PATH = OUTPUT_DIR / "pm_accident_grids.geojson"
OUTPUT_HTML_PATH = OUTPUT_DIR / "pm_accident_grids_map.html"
OUTPUT_PNG_PATH = OUTPUT_DIR / "pm_accident_grids.png"

# TAAS 사고 좌표와 grid/dynamic 결과물은 모두 EPSG:5179 기준으로 맞춘다.
TARGET_CRS = "EPSG:5179"
TARGET_EPSG = 5179
PM_VEHICLE_LABEL = "개인형이동수단(PM)"
WEB_CRS = "EPSG:4326"


def log(message: str) -> None:
    print(message)


def ensure_input_files_exist() -> None:
    missing_files: list[str] = []
    for path in [RAW_ACCIDENT_PATH, GRID_PATH, DYNAMIC_PATH]:
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            "PM 사고 전처리에 필요한 입력 파일이 없습니다. "
            f"누락 파일: {', '.join(missing_files)}"
        )


def ensure_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_target_crs(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError(f"{label} 데이터에 CRS 정보가 없습니다.")

    if gdf.crs.to_epsg() != TARGET_EPSG:
        log(f"{label} CRS를 {gdf.crs}에서 {TARGET_CRS}로 변환합니다.")
        return gdf.to_crs(TARGET_CRS)

    return gdf


def load_accident_points() -> gpd.GeoDataFrame:
    # TAAS JSON에서 PM 사고만 남기고, 좌표가 있는 사고를 포인트로 변환한다.
    rows = json.loads(RAW_ACCIDENT_PATH.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("TAAS JSON 최상위 구조가 list가 아닙니다.")

    accident_df = pd.DataFrame(rows).copy()
    required_columns = {"acdnt_no", "x_crdnt", "y_crdnt"}
    missing_columns = required_columns - set(accident_df.columns)
    if missing_columns:
        raise ValueError(f"TAAS JSON에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    if "wrngdo_vhcle_asort_dc" in accident_df.columns:
        accident_df = accident_df.loc[
            accident_df["wrngdo_vhcle_asort_dc"].fillna("").eq(PM_VEHICLE_LABEL)
        ].copy()

    accident_df["x_crdnt"] = pd.to_numeric(accident_df["x_crdnt"], errors="coerce")
    accident_df["y_crdnt"] = pd.to_numeric(accident_df["y_crdnt"], errors="coerce")
    accident_df = accident_df.dropna(subset=["acdnt_no", "x_crdnt", "y_crdnt"]).copy()
    # 사고 번호 기준 중복을 제거해 한 사고가 두 번 집계되지 않게 한다.
    accident_df = accident_df.drop_duplicates(subset=["acdnt_no"]).copy()

    if accident_df.empty:
        raise ValueError("좌표가 있는 PM 사고 데이터가 없습니다.")

    return gpd.GeoDataFrame(
        accident_df,
        geometry=gpd.points_from_xy(accident_df["x_crdnt"], accident_df["y_crdnt"]),
        crs=TARGET_CRS,
    )


def load_grid_and_dynamic() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # 사고를 매핑할 기준 grid와, pm_accident를 덮어쓸 동적 변수 파일을 함께 읽는다.
    grid_gdf = ensure_target_crs(gpd.read_parquet(GRID_PATH), "50m grid")
    dynamic_gdf = ensure_target_crs(gpd.read_parquet(DYNAMIC_PATH), "동적 변수")

    required_grid_columns = {"grid_id", "geometry"}
    required_dynamic_columns = {
        "grid_id",
        "geometry",
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
    }

    missing_grid_columns = required_grid_columns - set(grid_gdf.columns)
    missing_dynamic_columns = required_dynamic_columns - set(dynamic_gdf.columns)

    if missing_grid_columns:
        raise ValueError(f"grid 파일에 필요한 컬럼이 없습니다: {sorted(missing_grid_columns)}")
    if missing_dynamic_columns:
        raise ValueError(f"동적 변수 파일에 필요한 컬럼이 없습니다: {sorted(missing_dynamic_columns)}")

    if grid_gdf["grid_id"].duplicated().any():
        raise ValueError(f"grid 파일에 중복 grid_id가 있습니다: {int(grid_gdf['grid_id'].duplicated().sum())}개")
    if dynamic_gdf["grid_id"].duplicated().any():
        raise ValueError(f"동적 변수 파일에 중복 grid_id가 있습니다: {int(dynamic_gdf['grid_id'].duplicated().sum())}개")

    return grid_gdf, dynamic_gdf


def build_pm_accident_by_grid(
    accident_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    # 사고 포인트가 포함되는 50m grid를 찾아 사고 발생 grid 목록을 만든다.
    joined = gpd.sjoin(
        accident_gdf,
        grid_gdf[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )

    matched = joined.loc[joined["grid_id"].notna()].copy()
    grid_flags = (
        matched[["grid_id"]]
        .drop_duplicates()
        .assign(pm_accident=1)
        .reset_index(drop=True)
    )

    summary = {
        "raw_accident_rows": len(accident_gdf),
        "matched_accident_rows": int(matched["grid_id"].notna().sum()),
        "unmatched_accident_rows": int(joined["grid_id"].isna().sum()),
        "unique_grids_with_accident": int(grid_flags["grid_id"].nunique()),
    }
    return grid_flags, summary


def update_dynamic_pm_accident(
    dynamic_gdf: gpd.GeoDataFrame,
    grid_flags: pd.DataFrame,
) -> gpd.GeoDataFrame:
    # 사고가 매핑된 grid만 1로 두고, 나머지 grid는 0으로 채워 이진 타깃을 만든다.
    merged = dynamic_gdf.drop(columns=["pm_accident"]).merge(grid_flags, on="grid_id", how="left", validate="1:1")
    merged["pm_accident"] = merged["pm_accident"].fillna(0).astype("int64")

    unique_values = set(pd.Series(merged["pm_accident"]).dropna().unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"pm_accident 컬럼이 0/1 이진값이 아닙니다: {sorted(unique_values)}")

    return gpd.GeoDataFrame(merged, geometry="geometry", crs=dynamic_gdf.crs)


def save_dynamic_gdf(dynamic_gdf: gpd.GeoDataFrame) -> None:
    dynamic_gdf.to_parquet(DYNAMIC_PATH, index=False)
    log(f"실제 PM 사고값으로 갱신 완료: {DYNAMIC_PATH}")


def extract_pm_accident_grids(dynamic_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # pm_accident가 1인 격자만 따로 뽑아 시각화와 외부 확인용 파일로 사용한다.
    pm_grids = dynamic_gdf.loc[dynamic_gdf["pm_accident"].eq(1), ["grid_id", "pm_accident", "geometry"]].copy()

    if pm_grids.empty:
        raise ValueError("pm_accident가 1인 격자가 없어 시각화할 수 없습니다.")

    return pm_grids


def save_pm_accident_geojson(pm_grids: gpd.GeoDataFrame) -> None:
    pm_grids.to_file(OUTPUT_GEOJSON_PATH, driver="GeoJSON")
    log(f"PM 사고 격자 GeoJSON 저장 완료: {OUTPUT_GEOJSON_PATH}")


def save_pm_accident_html_map(pm_grids: gpd.GeoDataFrame) -> None:
    pm_grids_web = pm_grids.to_crs(WEB_CRS)
    center = pm_grids_web.geometry.union_all().centroid

    fmap = folium.Map(location=[center.y, center.x], zoom_start=13, tiles="CartoDB positron")
    folium.GeoJson(
        pm_grids_web,
        name="PM accident grids",
        style_function=lambda _: {
            "fillColor": "#d73027",
            "color": "#a50026",
            "weight": 1,
            "fillOpacity": 0.65,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["grid_id", "pm_accident"],
            aliases=["grid_id", "pm_accident"],
            sticky=False,
        ),
    ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.save(str(OUTPUT_HTML_PATH))
    log(f"PM 사고 격자 HTML 지도 저장 완료: {OUTPUT_HTML_PATH}")


def save_pm_accident_png(pm_grids: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> None:
    # 전체 grid 위에 사고 격자만 붉게 올려서 분포를 한눈에 확인할 수 있게 한다.
    fig, ax = plt.subplots(figsize=(10, 10))
    grid_gdf.plot(ax=ax, color="#f2f2f2", edgecolor="#d0d0d0", linewidth=0.15)
    pm_grids.plot(ax=ax, color="#d73027", edgecolor="#8b0000", linewidth=0.2, alpha=0.9)
    ax.set_title("PM Accident Grids (pm_accident = 1)", fontsize=14)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"PM 사고 격자 PNG 저장 완료: {OUTPUT_PNG_PATH}")


def print_summary(before_sum: int, after_gdf: gpd.GeoDataFrame, summary: dict[str, int]) -> None:
    log("=== Step 6: TAAS PM 사고 전처리 ===")
    log(f"원본 사고 건수: {summary['raw_accident_rows']}")
    log(f"grid 매핑 성공 건수: {summary['matched_accident_rows']}")
    log(f"grid 매핑 실패 건수: {summary['unmatched_accident_rows']}")
    log(f"사고가 있는 고유 grid 수: {summary['unique_grids_with_accident']}")
    log(f"기존 pm_accident 합계: {before_sum}")
    log(f"변경 후 pm_accident 합계: {int(after_gdf['pm_accident'].sum())}")


def main() -> int:
    try:
        # Step 6 결과물인 10_dynamic_variables.parquet의 pm_accident를 실제 사고값으로 갱신한다.
        ensure_input_files_exist()
        ensure_output_directory()
        accident_gdf = load_accident_points()
        grid_gdf, dynamic_gdf = load_grid_and_dynamic()
        before_sum = int(pd.to_numeric(dynamic_gdf["pm_accident"], errors="coerce").fillna(0).sum())

        grid_flags, summary = build_pm_accident_by_grid(accident_gdf, grid_gdf)
        updated_dynamic_gdf = update_dynamic_pm_accident(dynamic_gdf, grid_flags)
        save_dynamic_gdf(updated_dynamic_gdf)
        pm_grids = extract_pm_accident_grids(updated_dynamic_gdf)
        save_pm_accident_geojson(pm_grids)
        save_pm_accident_html_map(pm_grids)
        save_pm_accident_png(pm_grids, grid_gdf)
        print_summary(before_sum, updated_dynamic_gdf, summary)
        return 0
    except Exception as exc:
        log(f"Step 6 PM 사고 전처리 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
