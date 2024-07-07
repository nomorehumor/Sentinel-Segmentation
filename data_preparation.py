import torch
from torch.utils.data import TensorDataset
import rasterio
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from augmentation import augment_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")


from constants import SENTINEL_DATASET_DIR, BUILDING_DATASET_DIR, CITIES, SAVE_DIR

def create_dataset():
    patch_size = 32
    patches = torch.empty(0, patch_size, patch_size, 6)
    for city in CITIES:
        print(city)
        city_path = city.split(",")[0].lower()
        preprocess_tensor = load_city_bands(city_path)
        city_patches = create_patches(preprocess_tensor, patch_size)
        
        patches_cat = torch.empty(patches.shape[0] + city_patches.shape[0], patch_size, patch_size, 6)
        torch.cat((patches, city_patches), dim=0, out=patches_cat)
        patches = patches_cat
    create_torch_dataset(patches, patch_size)
    

def normalize(img):
    masked_data = np.ma.masked_equal(img, 0)
    lq, uq = np.quantile(masked_data.compressed(), (0.01, 0.99))
    image_norm = np.clip(img, a_min=lq, a_max=uq)
    image_norm = (image_norm - lq) / (uq - lq)
    return image_norm


def vis(img, quant_norm=True):
    if quant_norm:
        data = normalize(img)
    else:
        data = img
    if data.ndim == 2:
        plt.imshow(data, cmap="gray")
    elif data.ndim == 3:
        plt.imshow(data)


def load_city_bands(city_path):

    with rasterio.open(SENTINEL_DATASET_DIR / city_path / "R.tiff") as f:
        r_data = np.transpose(f.read(), (1,2,0)).squeeze()
    with rasterio.open(SENTINEL_DATASET_DIR /  city_path / "G.tiff") as f:
        g_data = np.transpose(f.read(), (1,2,0)).squeeze()    
    with rasterio.open(SENTINEL_DATASET_DIR /  city_path / "B.tiff") as f:
        b_data = np.transpose(f.read(), (1,2,0)).squeeze()
    with rasterio.open(SENTINEL_DATASET_DIR /  city_path / "IR.tiff") as f:
        ir_data = np.transpose(f.read(), (1,2,0)).squeeze()

    with rasterio.open(BUILDING_DATASET_DIR / city_path / "footprint.tiff") as f:
        building_data = np.transpose(f.read(), (1,2,0)).squeeze()


    all_bands = np.stack([
        normalize(r_data), 
        normalize(g_data), 
        normalize(b_data), 
        normalize(ir_data)
        ], axis=-1)

    stacked_data = np.stack([ir_data, r_data, g_data, b_data], axis=-1).reshape(-1, 4)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(stacked_data)
    labels = kmeans.labels_.reshape(ir_data.shape)

    cloud_label = kmeans.cluster_centers_[:, 0].argmax()
    cloud_mask = labels == cloud_label

    # Add cloud mask to input_output_tensor
    preprocess_data = np.dstack([all_bands, building_data, cloud_mask.astype(np.float32)])
    preprocess_tensor = torch.from_numpy(preprocess_data)  

    return preprocess_tensor


def is_cloud_present(img):
    if np.count_nonzero(img[:,:,5] == False) > 0:
        return True
    return False


def create_patches(preprocess_tensor, patch_size, plot=False):
    image_height, image_width = preprocess_tensor.size()[:2]

    patches_grid = preprocess_tensor.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).permute(0,1,3,4,2)
    patches_num_x, patches_num_y = patches_grid.size()[:2]
    patches = patches_grid.flatten(start_dim=0, end_dim=1)
    
    if plot:
        plt.figure(figsize=(10,10))
        plt.subplots_adjust(wspace=0, hspace=0)

        # Plot the patches in a grid
        for x in range(patches_num_x):
            for y in range(patches_num_y):
                ax = plt.subplot(patches_num_x, patches_num_y, x*patches_num_y + y + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis("off")
                
                patch_rgb = patches_grid[x, y, :, :, :3]
                plt.imshow(patch_rgb)

    return patches


def evaluate_dataset_positive_classes(dataset_label_tensors):
    # Evaluate the number of positive classes in the dataset
    flattened_dataset = dataset_label_tensors.flatten()
    return np.count_nonzero(flattened_dataset) / len(flattened_dataset)


def create_dataset_sets(input_tensor, output_tensor):
    output_tensor_flat = output_tensor.view(output_tensor.size(0), -1).mean(dim=1).numpy().round().astype(int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_val_idx, test_idx in splitter.split(np.zeros(len(input_tensor)), output_tensor_flat):
        inputs_train_val, inputs_test = input_tensor[train_val_idx], input_tensor[test_idx]
        output_train_val, output_test = output_tensor[train_val_idx], output_tensor[test_idx]
        
    splitter_train_val = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2 of original data
    for train_idx, val_idx in splitter_train_val.split(np.zeros(len(inputs_train_val)), output_train_val.view(output_train_val.size(0), -1).mean(dim=1).numpy().round().astype(int)):
        inputs_train, inputs_val = inputs_train_val[train_idx], inputs_train_val[val_idx]
        output_train, output_val = output_train_val[train_idx], output_train_val[val_idx]
    
    print("Distribution of positive classes in train dataset: ", evaluate_dataset_positive_classes(output_train))
    print("Distribution of positive classes in test dataset: ", evaluate_dataset_positive_classes(output_test))
    print("Distribution of positive classes in val dataset: ", evaluate_dataset_positive_classes(output_val))

    train = TensorDataset(inputs_train, output_train)
    val = TensorDataset(inputs_val, output_val)
    test = TensorDataset(inputs_test, output_test)
    
    return train, val, test

def normalize_data(data, channels=[]):
    means = []
    stds = []
    for i in channels:
        means.append(data[:,i,:,:].mean())
        stds.append(data[:,i,:,:].std())
    
    transform = transforms.Compose([
        transforms.Normalize(means, stds, inplace=True)
    ])
    transform(data)
    

def create_torch_dataset(patches, patch_size):
    valid_patches = []
    for i in range(len(patches)):
        if not is_cloud_present(patches[i]):
            valid_patches.append(i)
    print(f"{len(valid_patches)}/{len(patches)} patches are valid")

    input_tensor = patches[valid_patches, :, :, :4].permute(0,3,1,2) # change dims to (N, C, H, W)
    output_tensor = patches[valid_patches, :, :, 4]
    
    normalize_data(input_tensor, [0,1,2,3]) 
    
    train, val, test = create_dataset_sets(input_tensor, output_tensor)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(train, SAVE_DIR /  f'{str(patch_size)}_train.pt')
    torch.save(val, SAVE_DIR / f'{str(patch_size)}_val.pt')
    torch.save(test, SAVE_DIR / f'{str(patch_size)}_test.pt')

    print(f'Saved datasets to {SAVE_DIR}')

    augmented_train = augment_dataset(train)
    augmented_val = augment_dataset(val)
    augmented_test = augment_dataset(test)

    for transformation in augmented_train:
        torch.save(augmented_train[transformation], SAVE_DIR / f'{str(patch_size)}_augmented_{str(transformation)}_train.pt')
    for transformation in augmented_train:
        torch.save(augmented_val[transformation], SAVE_DIR / f'{str(patch_size)}_augmented_{str(transformation)}_val.pt')
    for transformation in augmented_train:
        torch.save(augmented_test[transformation], SAVE_DIR / f'{str(patch_size)}_augmented_{str(transformation)}_test.pt')

    print(f'Saved augmented datasets to {SAVE_DIR}')

    return train, test, val


if __name__== '__main__':
    create_dataset()