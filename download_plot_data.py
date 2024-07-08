from pathlib import Path
import matplotlib
import osmnx as ox
import openeo
import rasterio
from rasterio.features import rasterize
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from utils import *

from constants import DATASET_DIR, SENTINEL_DATASET_DIR, BUILDING_DATASET_DIR, CITIES, BERLIN_BBOX


def read_all_bands_data(dir_path):
    with rasterio.open(dir_path / "R.tiff") as f:
        r_data = np.transpose(f.read(), (1,2,0)).squeeze()
    with rasterio.open(dir_path / "G.tiff") as f:
        g_data = np.transpose(f.read(), (1,2,0)).squeeze()    
    with rasterio.open(dir_path / "B.tiff") as f:
        b_data = np.transpose(f.read(), (1,2,0)).squeeze()
    with rasterio.open(dir_path / "IR.tiff") as f:
        ir_data = np.transpose(f.read(), (1,2,0)).squeeze()

    all_bands = np.stack([r_data, g_data, b_data, ir_data], axis=-1)
    all_bands = quantile_normalize(all_bands)
    return all_bands
        

def download_sentinel_openeo(connection, bbox, dir_path, pbar=None):

    s2_cube = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=("2024-01-01", "2024-06-30"),
        spatial_extent={
            "west": bbox["west"],
            "south": bbox["south"],
            "east": bbox["east"],
            "north": bbox["north"]
        },
        bands=["B04", "B03", "B02", "B08"],
        max_cloud_cover=30
    )
    
    pbar.set_postfix({"data": "(B04)"})
    R_band = s2_cube.band("B04")
    R_band.download(dir_path / "R.tiff")
    
    pbar.set_postfix({"data": "(B03)"})
    G_band = s2_cube.band("B03")
    G_band.download(dir_path / "G.tiff")
    
    pbar.set_postfix({"data": "(B02)"})
    B_band = s2_cube.band("B02")
    B_band.download(dir_path / "B.tiff")
    
    pbar.set_postfix({"data": "(B08)"})
    IR_band = s2_cube.band("B08")
    IR_band.download(dir_path / "IR.tiff")
    

def rasterize_city_features(city_features, bands_dir, label_dir):
            
    r_file = rasterio.open(bands_dir / "R.tiff")
    city_features = city_features.to_crs(r_file.crs)

    meta = r_file.meta.copy()
    meta.update(compress='lzw')
    with rasterio.open(label_dir / "footprint.tiff", "w", **meta) as f:
        buildings_raster = rasterize(
            [(geom, 1) for geom in city_features.geometry],
            fill=0,
            out_shape=r_file.shape,
            transform=f.transform,
            dtype='uint8'
        )
        f.write(buildings_raster, 1)
    

def download_band_data(city, city_gdf):
    
    # OpenEO connection
    print("Authenticating to OpenEO")
    connection = openeo.connect("openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc()

    if city == "berlin":
        bbox = BERLIN_BBOX
    else:     
    
        bbox = {
            "north": city_gdf.bbox_north[0],
            "south": city_gdf.bbox_south[0],
            "east": city_gdf.bbox_east[0],
            "west": city_gdf.bbox_west[0]
        }

    
    city_bands_dir = SENTINEL_DATASET_DIR / city.split(",")[0].lower()
    city_bands_dir.mkdir(parents=True, exist_ok=True)
    download_sentinel_openeo(connection, bbox, city_bands_dir, pbar=cities_tqdm)
            
    
def download_building_data(city, city_gdf):

    cities_tqdm.set_description(f"Downloading building data for {city}")
    
    city_bands_dir = SENTINEL_DATASET_DIR / city.split(",")[0].lower()
    city_label_dir = BUILDING_DATASET_DIR / city.split(",")[0].lower()
    city_label_dir.mkdir(parents=True, exist_ok=True)
    
    if city == "berlin":
        bbox = BERLIN_BBOX
    else:     
    
        bbox = {
            "north": city_gdf.bbox_north[0],
            "south": city_gdf.bbox_south[0],
            "east": city_gdf.bbox_east[0],
            "west": city_gdf.bbox_west[0]
        }

    city_features = ox.features_from_bbox(north=bbox["north"], south=bbox["south"], east=bbox["east"], west=bbox["west"], tags={'building':True})
    rasterize_city_features(city_features, city_bands_dir, city_label_dir)
    
        
def create_bands_plots(city):
    city_path_sentinel = SENTINEL_DATASET_DIR / city.split(",")[0].lower()
    
    all_bands = read_all_bands_data(city_path_sentinel)
    
    #RGB bands
    rgb_data = np.stack(
        [
        all_bands[..., 0],
        all_bands[..., 1],
        all_bands[..., 2], 
        ],
        axis=-1
    )

    #IRB bands
    irb_data = np.stack(
    [
        all_bands[..., 3],
        all_bands[..., 0],
        all_bands[..., 1], 
        ],
        axis=-1
    )
    
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_data)
    plt.savefig(city_path_sentinel / "rgb.png")
    
    plt.figure(figsize=(20, 20))
    plt.imshow(all_bands[..., 0], cmap="gray")
    plt.savefig(city_path_sentinel / "r.png")
    
    plt.figure(figsize=(20, 20))
    plt.imshow(all_bands[..., 1], cmap="gray")
    plt.savefig(city_path_sentinel / "g.png")
    
    plt.figure(figsize=(20, 20))
    plt.imshow(all_bands[..., 2], cmap="gray")
    plt.savefig(city_path_sentinel / "b.png")
    
    plt.figure(figsize=(20, 20))
    plt.imshow(all_bands[..., 3], cmap="gray")
    plt.savefig(city_path_sentinel / "ir.png")
    
    plt.figure(figsize=(20, 20))
    plt.imshow(irb_data)
    plt.savefig(city_path_sentinel / "irb.png")
    
    
def create_building_plots(city_bands_dir, city_building_dir):
    footprint_file = rasterio.open(city_building_dir / "footprint.tiff")
    building_data = np.transpose(footprint_file.read(), (1,2,0)).squeeze()
    r_data = rasterio.open(city_bands_dir / "R.tiff").read().squeeze()
    r_data = quantile_normalize(r_data)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(building_data, cmap="Blues")
    plt.savefig(city_building_dir / "footprint.png")  

    blue_cmap = matplotlib.colors.ListedColormap([(0,0,0,0), (0.3,0.3,1,0.7)])
    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(r_data, cmap="gray")
    plt.imshow(building_data, cmap=blue_cmap, alpha=1)
    plt.savefig(city_building_dir / "footprint_overlay.png")
        
if __name__ == '__main__':
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    SENTINEL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    BUILDING_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    cities_tqdm = tqdm(CITIES)
    
    for city in cities_tqdm:
        city_gdf = ox.geocode_to_gdf(city)
        city = city.split(",")[0].lower()
        
        cities_tqdm.set_description(f"Downloading band data for {city}")
        download_band_data(city, city_gdf)
        create_bands_plots(city)
                
        cities_tqdm.set_description(f"Downloading building data for {city}")
        download_building_data(city, city_gdf)
        create_building_plots(SENTINEL_DATASET_DIR / city, BUILDING_DATASET_DIR / city)