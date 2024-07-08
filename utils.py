from matplotlib import pyplot as plt
import numpy as np


def quantile_normalize(img):
    masked_data = np.ma.masked_equal(img, 0)
    lq, uq = np.quantile(masked_data.compressed(), (0.01, 0.99))
    image_norm = np.clip(img, a_min=lq, a_max=uq)
    image_norm = (image_norm - lq) / (uq - lq)
    return image_norm

def visualize(img, quant_norm=True):
    """
    Visualizes one-band or multiband image
    """
    if quant_norm:
        data = quantile_normalize(img)
    else:
        data = img
    if data.ndim == 2:
        plt.imshow(data, cmap="gray")
    elif data.ndim == 3:
        plt.imshow(data)
        
def evaluate_dataset_positive_classes(dataset_label_tensors):
    flattened_dataset = dataset_label_tensors.flatten()
    return np.count_nonzero(flattened_dataset) / len(flattened_dataset)