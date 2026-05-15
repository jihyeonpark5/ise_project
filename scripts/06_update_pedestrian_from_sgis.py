from __future__ import annotations

from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PEDESTRIAN_DIR = PROJECT_ROOT / "data" / "raw" / "pedestrian"
GRID_PATH = PROJECT_ROOT / "data" / "processed" / "02_gangnam_grid_50m.parquet"
DYNAMIC_PATH = PROJECT_ROOT / "data" / "processed" / "10_dynamic_variables.parquet"

TARGET_CRS = "EPSG:5179"
TARGET_EPSG = 5179
BLOCK_CODE_COLUMN = "TOT_REG_CD"
TIME_COLUMN = "시간대구분"
VALUE_COLUMN = "총생활인구수"
BLOCK_XLSX_CODE_COLUMN = "집계구코드"
PEDESTRIAN_COLUMNS_BY_TIME = {
    "10": "pedestrian_10h",
    "18": "pedestrian_18h",
    "22": "pedestrian_22h",
}


def log(message: str) -> None:
    print(message)


def ensure_input_files_exist() -> None:
    if not RAW_PEDESTRIAN_DIR.exists():
        raise FileNotFoundError(f"보행자 원본 폴더가 없습니다: {RAW_PEDESTRIAN_DIR}")

    if not GRID_PATH.exists():
        raise FileNotFoundError(f"grid 파일이 없습니다: {GRID_PATH}")

    if not DYNAMIC_PATH.exists():
        raise FileNotFoundError(f"동적 변수 파일이 없습니다: {DYNAMIC_PATH}")


def find_pedestrian_source_files() -> tuple[Path, Path]:
    xlsx_files = sorted(path for path in RAW_PEDESTRIAN_DIR.iterdir() if path.suffix.lower() == ".xlsx")
    shp_files = sorted(path for path in RAW_PEDESTRIAN_DIR.iterdir() if path.suffix.lower() == ".shp")

    if len(xlsx_files) != 1:
        raise ValueError(f"보행자 엑셀 파일은 1개여야 합니다. 현재 {len(xlsx_files)}개")
    if len(shp_files) != 1:
        raise ValueError(f"집계구 shp 파일은 1개여야 합니다. 현재 {len(shp_files)}개")

    return xlsx_files[0], shp_files[0]


def ensure_target_crs(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError(f"{label} 데이터에 CRS 정보가 없습니다.")

    if gdf.crs.to_epsg() != TARGET_EPSG:
        log(f"{label} CRS를 {gdf.crs}에서 {TARGET_CRS}로 변환합니다.")
        return gdf.to_crs(TARGET_CRS)

    return gdf


def read_xlsx_sheet_as_dataframe(xlsx_path: Path) -> pd.DataFrame:
    # openpyxl 없이 xlsx 내부 XML을 직접 읽어 필요한 표만 DataFrame으로 복원한다.
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
        if "xl/sharedStrings.xml" not in zf.namelist():
            return []

        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        values: list[str] = []
        for item in root.findall("a:si", namespace):
            parts = [node.text or "" for node in item.iterfind(".//a:t", namespace)]
            values.append("".join(parts))
        return values

    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = read_shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = root.find("a:sheetData", namespace)
        if rows is None:
            raise ValueError("엑셀 sheet1에 sheetData가 없습니다.")

        parsed_rows: list[list[str]] = []
        for row in rows.findall("a:row", namespace):
            values: list[str] = []
            for cell in row.findall("a:c", namespace):
                cell_type = cell.attrib.get("t")
                cell_value = cell.find("a:v", namespace)
                if cell_value is None:
                    values.append("")
                    continue

                raw = cell_value.text or ""
                values.append(shared_strings[int(raw)] if cell_type == "s" else raw)
            parsed_rows.append(values)

    if not parsed_rows:
        raise ValueError("엑셀 데이터가 비어 있습니다.")

    headers = parsed_rows[0]
    body = [row for row in parsed_rows[1:] if len(row) == len(headers)]
    return pd.DataFrame(body, columns=headers)


def load_pedestrian_table(xlsx_path: Path) -> pd.DataFrame:
    raw_df = read_xlsx_sheet_as_dataframe(xlsx_path).copy()

    required_columns = {TIME_COLUMN, BLOCK_XLSX_CODE_COLUMN, VALUE_COLUMN}
    missing_columns = required_columns - set(raw_df.columns)
    if missing_columns:
        raise ValueError(f"보행자 엑셀에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    df = raw_df[[TIME_COLUMN, BLOCK_XLSX_CODE_COLUMN, VALUE_COLUMN]].copy()
    df[TIME_COLUMN] = df[TIME_COLUMN].astype(str).str.strip()
    df[BLOCK_XLSX_CODE_COLUMN] = df[BLOCK_XLSX_CODE_COLUMN].astype(str).str.strip()
    df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors="coerce")
    df = df.dropna(subset=[TIME_COLUMN, BLOCK_XLSX_CODE_COLUMN, VALUE_COLUMN]).copy()
    df = df.loc[df[TIME_COLUMN].isin(PEDESTRIAN_COLUMNS_BY_TIME)].copy()

    if df.empty:
        raise ValueError("유효한 보행자 시간대 데이터가 없습니다.")

    wide = (
        df.assign(pedestrian_column=df[TIME_COLUMN].map(PEDESTRIAN_COLUMNS_BY_TIME))
        .pivot_table(
            index=BLOCK_XLSX_CODE_COLUMN,
            columns="pedestrian_column",
            values=VALUE_COLUMN,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    wide.columns.name = None

    for column in PEDESTRIAN_COLUMNS_BY_TIME.values():
        if column not in wide.columns:
            wide[column] = 0.0

    return wide


def load_block_geometries(shp_path: Path) -> gpd.GeoDataFrame:
    block_gdf = gpd.read_file(shp_path)

    required_columns = {BLOCK_CODE_COLUMN, "geometry"}
    missing_columns = required_columns - set(block_gdf.columns)
    if missing_columns:
        raise ValueError(f"집계구 shp에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    if block_gdf.crs is None:
        # SGIS 안내에 따라 집계구 경계는 UTM-K(GRS80), EPSG:5179로 본다.
        block_gdf = block_gdf.set_crs(TARGET_CRS, allow_override=True)
    else:
        block_gdf = ensure_target_crs(block_gdf, "집계구")

    block_gdf[BLOCK_CODE_COLUMN] = block_gdf[BLOCK_CODE_COLUMN].astype(str).str.strip()
    block_gdf = block_gdf[[BLOCK_CODE_COLUMN, "geometry"]].copy()
    block_gdf = block_gdf.loc[block_gdf.geometry.notna() & ~block_gdf.geometry.is_empty].copy()

    if block_gdf[BLOCK_CODE_COLUMN].duplicated().any():
        duplicate_count = int(block_gdf[BLOCK_CODE_COLUMN].duplicated().sum())
        raise ValueError(f"집계구 shp에 중복 코드가 있습니다: {duplicate_count}개")

    return block_gdf


def load_grid_and_dynamic() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    grid_gdf = ensure_target_crs(gpd.read_parquet(GRID_PATH), "50m grid")
    dynamic_gdf = ensure_target_crs(gpd.read_parquet(DYNAMIC_PATH), "동적 변수")

    grid_required_columns = {"grid_id", "geometry"}
    dynamic_required_columns = {"grid_id", "geometry", *PEDESTRIAN_COLUMNS_BY_TIME.values()}

    missing_grid_columns = grid_required_columns - set(grid_gdf.columns)
    missing_dynamic_columns = dynamic_required_columns - set(dynamic_gdf.columns)
    if missing_grid_columns:
        raise ValueError(f"grid 파일에 필요한 컬럼이 없습니다: {sorted(missing_grid_columns)}")
    if missing_dynamic_columns:
        raise ValueError(f"동적 변수 파일에 필요한 컬럼이 없습니다: {sorted(missing_dynamic_columns)}")

    return grid_gdf, dynamic_gdf


def build_block_level_pedestrian_gdf(block_gdf: gpd.GeoDataFrame, pedestrian_wide_df: pd.DataFrame) -> gpd.GeoDataFrame:
    merged = block_gdf.merge(
        pedestrian_wide_df,
        left_on=BLOCK_CODE_COLUMN,
        right_on=BLOCK_XLSX_CODE_COLUMN,
        how="inner",
        validate="1:1",
    )

    if merged.empty:
        raise ValueError("집계구 경계와 엑셀 보행자 데이터가 연결되지 않았습니다.")

    return gpd.GeoDataFrame(merged.drop(columns=[BLOCK_XLSX_CODE_COLUMN]), geometry="geometry", crs=block_gdf.crs)


def allocate_pedestrian_to_grid(block_pedestrian_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    # 집계구와 격자가 겹치는 후보만 찾은 뒤, 교차 면적 비율만큼 보행자 값을 분배한다.
    candidate = gpd.sjoin(
        block_pedestrian_gdf[[BLOCK_CODE_COLUMN, *PEDESTRIAN_COLUMNS_BY_TIME.values(), "geometry"]],
        grid_gdf[["grid_id", "geometry"]],
        how="inner",
        predicate="intersects",
    ).reset_index(drop=True)

    if candidate.empty:
        raise ValueError("집계구와 50m grid가 겹치지 않습니다.")

    grid_lookup = grid_gdf.set_index("grid_id")["geometry"]
    candidate["grid_geometry"] = candidate["grid_id"].map(grid_lookup)
    candidate["block_area"] = candidate.geometry.area
    candidate["intersection_area"] = candidate.apply(
        lambda row: row.geometry.intersection(row["grid_geometry"]).area,
        axis=1,
    )
    candidate = candidate.loc[candidate["intersection_area"] > 0].copy()
    candidate["area_weight"] = candidate["intersection_area"] / candidate["block_area"]

    for column in PEDESTRIAN_COLUMNS_BY_TIME.values():
        candidate[f"{column}_weighted"] = candidate[column] * candidate["area_weight"]

    aggregated = (
        candidate.groupby("grid_id")[[f"{column}_weighted" for column in PEDESTRIAN_COLUMNS_BY_TIME.values()]]
        .sum()
        .rename(columns={f"{column}_weighted": column for column in PEDESTRIAN_COLUMNS_BY_TIME.values()})
        .reset_index()
    )

    summary: dict[str, int | float] = {
        "matched_block_count": int(block_pedestrian_gdf[BLOCK_CODE_COLUMN].nunique()),
        "candidate_overlap_rows": int(len(candidate)),
        "covered_grid_count": int(aggregated["grid_id"].nunique()),
    }
    for column in PEDESTRIAN_COLUMNS_BY_TIME.values():
        summary[f"{column}_source_sum"] = float(block_pedestrian_gdf[column].sum())
        summary[f"{column}_allocated_sum"] = float(aggregated[column].sum())

    return aggregated, summary


def update_dynamic_pedestrian(
    dynamic_gdf: gpd.GeoDataFrame,
    allocated_pedestrian_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    preserved_columns = [column for column in dynamic_gdf.columns if column not in PEDESTRIAN_COLUMNS_BY_TIME.values()]
    merged = dynamic_gdf[preserved_columns].merge(allocated_pedestrian_df, on="grid_id", how="left", validate="1:1")

    for column in PEDESTRIAN_COLUMNS_BY_TIME.values():
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    return gpd.GeoDataFrame(merged, geometry="geometry", crs=dynamic_gdf.crs)


def save_dynamic_gdf(dynamic_gdf: gpd.GeoDataFrame) -> None:
    dynamic_gdf.to_parquet(DYNAMIC_PATH, index=False)
    log(f"보행자 값 갱신 완료: {DYNAMIC_PATH}")


def print_summary(before_gdf: gpd.GeoDataFrame, after_gdf: gpd.GeoDataFrame, summary: dict[str, int | float]) -> None:
    log("=== Step 6: SGIS 보행자 값 교체 ===")
    log(f"매칭된 집계구 수: {summary['matched_block_count']}")
    log(f"집계구-grid 겹침 행 수: {summary['candidate_overlap_rows']}")
    log(f"보행자 값이 할당된 grid 수: {summary['covered_grid_count']}")

    for column in PEDESTRIAN_COLUMNS_BY_TIME.values():
        before_sum = float(pd.to_numeric(before_gdf[column], errors='coerce').fillna(0).sum())
        after_sum = float(pd.to_numeric(after_gdf[column], errors='coerce').fillna(0).sum())
        source_sum = float(summary[f"{column}_source_sum"])
        allocated_sum = float(summary[f"{column}_allocated_sum"])
        log(
            f"{column}: 기존합={before_sum:.4f}, 원본합={source_sum:.4f}, "
            f"분배합={allocated_sum:.4f}, 변경후합={after_sum:.4f}"
        )


def main() -> int:
    try:
        ensure_input_files_exist()
        xlsx_path, shp_path = find_pedestrian_source_files()
        pedestrian_wide_df = load_pedestrian_table(xlsx_path)
        block_gdf = load_block_geometries(shp_path)
        grid_gdf, dynamic_gdf = load_grid_and_dynamic()
        block_pedestrian_gdf = build_block_level_pedestrian_gdf(block_gdf, pedestrian_wide_df)
        allocated_pedestrian_df, summary = allocate_pedestrian_to_grid(block_pedestrian_gdf, grid_gdf)
        updated_dynamic_gdf = update_dynamic_pedestrian(dynamic_gdf, allocated_pedestrian_df)
        save_dynamic_gdf(updated_dynamic_gdf)
        print_summary(dynamic_gdf, updated_dynamic_gdf, summary)
        return 0
    except Exception as exc:
        log(f"SGIS 보행자 값 교체 실패: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
