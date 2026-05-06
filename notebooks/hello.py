import geopandas as gpd

gdf = gpd.read_file("data/raw/boundary/seoul_sig.shp")
print(gdf.columns)
print(gdf[gdf["SIGUNGU_NM"] == "강남구"])
