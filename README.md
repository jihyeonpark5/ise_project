# pm_safe_route

공유 전동킥보드의 시공간 위험도 정량화 및 안전 가중치 기반 최적 경로 추천 시스템

## 폴더 구조

```text
pm_safe_route/
├── data/
│   ├── raw/
│   │   ├── boundary/
│   │   ├── facilities/
│   │   ├── population/
│   │   ├── traffic/
│   │   ├── slope/
│   │   └── accident/
│   ├── processed/
│   └── output/
├── notebooks/
├── scripts/
│   └── 00_check_environment.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 사용 패키지

- `osmnx`
- `geopandas`
- `pandas`
- `numpy`
- `scikit-learn`
- `folium`
- `shapely`
- `pyproj`
- `statsmodels`
- `pyarrow`
- `matplotlib`
- `ortools`

## 패키지 역할 메모

- `osmnx`, `geopandas`, `shapely`, `pyproj`는 공간 데이터 처리와 도로 네트워크 분석에 사용합니다.
- `pandas`, `numpy`, `pyarrow`는 표 형태 데이터 처리 및 파일 입출력에 사용합니다.
- `scikit-learn`, `statsmodels`는 위험도 정량화 과정의 분석 및 모델링 단계에서 활용할 수 있습니다.
- `folium`, `matplotlib`는 시각화에 사용합니다.
- `ortools`는 **현재 전처리 단계용이 아니라 이후 경로 최적화 단계**에서 사용할 예정입니다.

## Windows PowerShell 기준 실행 방법

### 1. 프로젝트 폴더로 이동

```powershell
cd C:\Users\user\Documents\pm_safe_route
```

### 2. 가상환경 생성

```powershell
python -m venv .venv
```

### 3. 가상환경 활성화

```powershell
.\.venv\Scripts\Activate.ps1
```

PowerShell 실행 정책 때문에 활성화가 막히면 아래 명령을 한 번 실행한 뒤 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4. 패키지 설치

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. 환경 점검 스크립트 실행

```powershell
python .\scripts\00_check_environment.py
```

정상 실행되면 주요 패키지 import 결과와 필수 폴더 존재 여부가 출력됩니다.

## processed 산출물 설명

`data/processed` 폴더에는 각 단계에서 생성된 전처리 결과 파일이 저장됩니다. 파일명 앞의 번호는 생성 순서를 의미합니다.

| 파일명 | 생성 단계 | 설명 |
| --- | --- | --- |
| `01_gangnam_boundary.parquet` | Step 2 | 강남구 행정경계 polygon입니다. 모든 공간분석의 기준 경계로 사용합니다. |
| `02_gangnam_grid_50m.parquet` | Step 2 | 강남구 경계와 교차하는 50m x 50m 격자 polygon입니다. `grid_id`, `centroid_x`, `centroid_y`, `geometry`를 포함합니다. |
| `03_gangnam_grid_50m.geojson` | Step 2 | 50m 격자를 QGIS 등에서 확인하기 위한 GeoJSON 파일입니다. 분석보다는 시각 확인용입니다. |
| `04_osm_nodes_raw.parquet` | Step 3 | OSMnx로 수집한 bike network의 원본 node 데이터입니다. 교차로 수 계산 시 node 위치를 가져오는 데 사용합니다. (도로망의 점) |
| `05_osm_edges_raw.parquet` | Step 3 | OSMnx로 수집한 bike network의 원본 edge 데이터입니다. 필터링 전 도로망입니다.(점과 점을 잇는 도로 선) |
| `06_osm_edges_filtered.parquet` | Step 3 | PM 주행이 부적절한 후보 구간을 제거한 edge 데이터입니다. 이후 도로 기반 변수 생성의 기본 도로망으로 사용합니다. |
| `07_osm_edges_filtered.geojson` | Step 3 | 필터링된 도로망을 QGIS 등에서 확인하기 위한 GeoJSON 파일입니다. |
| `08_grid_road_map.parquet` | Step 4 | 50m 격자와 필터링된 도로 edge를 교차시킨 결과입니다. 각 도로가 각 격자 안에서 차지하는 길이 `length_in_grid`를 포함합니다. |

모든 공간 산출물은 거리 계산과 buffer 계산을 위해 `EPSG:5179` 좌표계로 통일합니다.

## processed 주요 변수 설명

아래 설명은 `data/processed`에 생성된 parquet 파일의 주요 컬럼을 기준으로 정리한 것입니다. `03_gangnam_grid_50m.geojson`, `07_osm_edges_filtered.geojson`은 각각 같은 단계의 parquet을 QGIS 등에서 보기 쉽게 저장한 시각 확인용 파일입니다.

### 공통 변수

| 변수명 | 설명 |
| --- | --- |
| `geometry` | 공간 객체입니다. 파일에 따라 polygon, point, line이 들어갑니다. 모든 geometry는 `EPSG:5179` 기준입니다. |

### `01_gangnam_boundary.parquet`

| 변수명 | 설명 |
| --- | --- |
| `district_name` | 추출된 자치구 이름입니다. 현재 값은 `강남구`입니다. |
| `source_rule` | 원천 경계 데이터에서 강남구를 어떤 조건으로 추출했는지 기록한 값입니다. |
| `geometry` | 강남구 행정경계 polygon입니다. |

### `02_gangnam_grid_50m.parquet`

| 변수명 | 설명 |
| --- | --- |
| `grid_id` | 50m 격자의 고유 ID입니다. 예: `G000001` |
| `centroid_x` | 격자 중심점의 x 좌표입니다. 참고용 좌표입니다. |
| `centroid_y` | 격자 중심점의 y 좌표입니다. 참고용 좌표입니다. |
| `geometry` | 50m x 50m 격자 polygon입니다. |

### `04_osm_nodes_raw.parquet`

| 변수명 | 설명 |
| --- | --- |
| `osmid` | OSM node 고유 ID입니다. edge의 `u`, `v`와 연결됩니다. |
| `x` | 원본 OSM node의 경도 좌표입니다. |
| `y` | 원본 OSM node의 위도 좌표입니다. |
| `street_count` | OSMnx가 계산한 node 주변 도로 연결 수입니다. |
| `junction` | OSM의 교차점 관련 태그입니다. 값이 없을 수 있습니다. |
| `highway` | node에 부여된 OSM highway 태그입니다. 값이 없을 수 있습니다. |
| `geometry` | `EPSG:5179`로 변환된 node point입니다. |

### `05_osm_edges_raw.parquet`, `06_osm_edges_filtered.parquet`

| 변수명 | 설명 |
| --- | --- |
| `u` | edge 시작 node의 OSM ID입니다. |
| `v` | edge 끝 node의 OSM ID입니다. |
| `key` | 같은 `u`, `v` 사이에 여러 edge가 있을 때 구분하는 값입니다. |
| `osmid` | OSM way ID입니다. 하나의 edge에 여러 ID가 들어갈 수 있습니다. |
| `highway` | 도로 유형입니다. 예: `residential`, `secondary`, `cycleway` |
| `name` | 도로명입니다. 값이 없을 수 있습니다. |
| `oneway` | 일방통행 여부입니다. |
| `reversed` | OSMnx 그래프 생성 과정에서 방향이 반전되었는지 나타내는 값입니다. |
| `length` | OSMnx가 계산한 edge 길이입니다. 단위는 미터입니다. |
| `width` | OSM에 기록된 도로 폭입니다. 숫자, 문자열, 결측치가 섞여 있을 수 있습니다. |
| `bridge` | 교량 여부 또는 관련 태그입니다. 값이 없을 수 있습니다. |
| `tunnel` | 터널 여부 또는 관련 태그입니다. 값이 없을 수 있습니다. |
| `lanes` | 차로 수입니다. 값이 없거나 문자열일 수 있습니다. |
| `maxspeed` | 제한속도 태그입니다. 값이 없거나 문자열일 수 있습니다. |
| `ref` | 도로 번호 또는 참조 코드입니다. 값이 없을 수 있습니다. |
| `access` | 접근 제한 태그입니다. 예: `no`, `private` |
| `service` | service road의 세부 유형입니다. 값이 없을 수 있습니다. |
| `geometry` | 도로 edge의 line geometry입니다. |

`05_osm_edges_raw.parquet`은 필터링 전 원본 도로망이고, `06_osm_edges_filtered.parquet`은 PM 주행이 어려운 후보 구간을 제거한 도로망입니다.

### `08_grid_road_map.parquet`

| 변수명 | 설명 |
| --- | --- |
| `grid_id` | 도로 segment가 교차하는 50m 격자 ID입니다. |
| `edge_id` | 격자-도로 매핑을 위해 생성한 edge 고유 ID입니다. |
| `highway` | 해당 도로 segment의 OSM 도로 유형입니다. |
| `access` | 해당 도로 segment의 접근 제한 태그입니다. |
| `width` | 해당 도로 segment의 폭 관련 태그입니다. |
| `length_in_grid` | 도로 edge가 해당 격자 내부에서 차지하는 길이입니다. 단위는 미터입니다. |
| `geometry` | 격자 내부에 포함된 도로 segment line geometry입니다. |

