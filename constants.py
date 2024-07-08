from pathlib import Path

DATASET_DIR = Path("datasets")
BUILDING_DATASET_DIR = DATASET_DIR / "building_footprints"
SENTINEL_DATASET_DIR = DATASET_DIR / "sentinel"
TRAINING_DATASET_DIR = DATASET_DIR / "training"

RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
PARAMS_DIR = RESULTS_DIR / "params"

CITIES = ["Bologna, Italy", "Milan, Italy", "Split, Croatia", "Valencia, Spain", "Oslo, Norway", "Krasnodar, Russia", "Paris, France", "Barcelona, Spain", "Berlin, Germany", "Karlsruhe, Germany", "Porto, Portugal"]

BERLIN_BBOX  = {
    "north" : 52.574409,
    "south": 52.454927, 
    "west"  : 13.294333,
    "east": 13.500205 
}
TRAIN_PATCH_SIZE = 64

