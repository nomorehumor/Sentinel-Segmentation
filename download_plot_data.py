from pathlib import Path
import osmnx as ox
import openeo
import rasterio
from rasterio.features import rasterize
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

DATASET_DIR = Path("datasets")
BUILDING_DATASET_DIR = DATASET_DIR / "building_footprints"
SENTINEL_DATASET_DIR = DATASET_DIR / "sentinel"
SAVE_DIR = DATASET_DIR / "training"
CITIES = ["Bologna, Italy", "Milan, Italy", "Split, Croatia", "Valencia, Spain", "Oslo, Norway", "Krasnodar, Russia", "Paris, France", "Barcelona, Spain", "Berlin, Germany"]


def normalize(img):
    masked_data = np.ma.masked_equal(img, 0)
    lq, uq = np.quantile(masked_data.compressed(), (0.01, 0.99))
    image_norm = np.clip(img, a_min=lq, a_max=uq)
    image_norm = (image_norm - lq) / (uq - lq)
    return image_norm


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
    all_bands = normalize(all_bands)
    return all_bands


def read_footprint_data(dir_path):
    with rasterio.open(dir_path / "footprint.tiff") as f:
        building_data = np.transpose(f.read(), (1,2,0)).squeeze()
    return building_data
        

def download_sentinel_openeo(connection, bbox, dir_path, pbar=None):

    s2_cube = connection.load_collection(
        "SENTINEL2_L2A",
        temporal_extent=("2024-05-01", "2024-05-30"),
        spatial_extent={
            "west": bbox["west"],
            "south": bbox["south"],
            "east": bbox["east"],
            "north": bbox["north"]
        },
        bands=["B04", "B03", "B02", "B08"],
        max_cloud_cover=20,
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
    

def download_band_data(city, city_gdf, city_tqdm):
    
    # OpenEO connection
    print("Authenticating to OpenEO")
    connection = openeo.connect("openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc()
    
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
    
    bbox = {
        "north": city_gdf.bbox_north[0],
        "south": city_gdf.bbox_south[0],
        "east": city_gdf.bbox_east[0],
        "west": city_gdf.bbox_west[0]
    }
    city_features = ox.features_from_bbox(north=bbox["north"], south=bbox["south"], east=bbox["east"], west=bbox["west"], tags={'building':True})
    rasterize_city_features(city_features, city_bands_dir, city_label_dir)     

    with rasterio.open(city_bands_dir / "R.tiff") as f:
        r_data = np.transpose(f.read(), (1,2,0)).squeeze()
        r_data = normalize(r_data)
    with rasterio.open(city_label_dir / "footprint.tiff") as f:
        building_data = np.transpose(f.read(), (1,2,0)).squeeze()
        
    plt.figure(figsize=(20, 20))
    plt.imshow(building_data, cmap="Blues")
    plt.savefig(city_label_dir / "footprint.png")  

    bounds = city_features.total_bounds
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(r_data, cmap="gray", extent=(bounds[0], bounds[2], bounds[1], bounds[3]))
    city_features.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.5, alpha=0.2)
    fig.savefig(city_label_dir / "footprint_overlay.png")
    
        
def create_bands_plots(city):
    # for city in CITIES:
    
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
    

        
if __name__ == '__main__':
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    SENTINEL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    BUILDING_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    cities_tqdm = tqdm(CITIES)
    
    for city in cities_tqdm:
        cities_tqdm.set_description(f"Downloading band data for {city}")
        city_gdf = ox.geocode_to_gdf(city)
        download_band_data(city, city_gdf, cities_tqdm)
        create_bands_plots(city)
                
        cities_tqdm.set_description(f"Downloading building data for {city}")
        download_building_data(city, city_gdf)