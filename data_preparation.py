import torch
from torch.utils.data import TensorDataset
import rasterio
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from constants import SENTINEL_DATASET_DIR, BUILDING_DATASET_DIR, CITIES

def create_dataset():
    patch_size = 32
    patches = []
    for city in CITIES:
        print(city)
        city_path = city.split(",")[0].lower()
        preprocess_tensor = load_city_bands(city_path)
        city_patch = create_patches(preprocess_tensor, patch_size)
        patches.append(city_patch)
  
    create_torch_dataset(patches)
    


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


    all_bands = np.stack([r_data, g_data, b_data, ir_data], axis=-1)
    all_bands = normalize(all_bands)
    vis(all_bands[:,:,:3], quant_norm=False)

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

    patch_size = patch_size

    patches_grid = preprocess_tensor.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).permute(0,1,3,4,2)
    patches_num_x, patches_num_y = patches_grid.size()[:2]
    # print(patches_grid.size())
    patches = patches_grid.flatten(start_dim=0, end_dim=1)
    # print(patches.size())
    
    if plot:
        plt.figure(figsize=(10,10))
        plt.subplots_adjust(wspace=0, hspace=0)

        # Plot the patches in a grid
        for x in range(patches_num_x):
            for y in range(patches_num_y):
                ax = plt.subplot(patches_num_x, patches_num_y, x*patches_num_y + y + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.axis("off")
                
                patch_rgb = patches_grid[x, y, :, :, :3]
                plt.imshow(patch_rgb)
                # plt.imshow(patches_grid[x, y, :, :, 4], cmap="gray")

    return patches


def create_torch_dataset(patches_list):

    valid_patches = []
    for patches in patches_list:
        for i in range(len(patches)):
            if not is_cloud_present(patches[i]):
                valid_patches.append(i)
        print(f"{len(valid_patches)}/{len(patches)} patches are valid")

    input_tensor = patches[valid_patches, :, :, :3].permute(0,3,1,2) # change dims to (N, C, H, W)
    output_tensor = patches[valid_patches, :, :, 4]
    dataset = TensorDataset(input_tensor, output_tensor)

    generator = torch.Generator().manual_seed(42)
    train, test, val = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=generator)

    torch.save(train, SAVE_DIR / 'train.pt')
    torch.save(val, SAVE_DIR / 'val.pt')
    torch.save(test, SAVE_DIR / 'test.pt')

    print(f'Saved datasets to {SAVE_DIR}')

    return train, test, val


if __name__== '__main__':
    create_dataset()