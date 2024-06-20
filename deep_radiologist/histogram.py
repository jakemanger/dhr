import torchio as tio
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def calculate_training_histogram(train_data_loader):
    """Calculate the intensity histogram, means, and standard deviations of the training data."""
    print('Calculating histogram of training data')
    histogram_bins = 256
    train_histogram = np.zeros(histogram_bins)
    range_min, range_max = -3, 3  # Define the range for the intensity values

    train_means = []
    train_sds = []
    filenames = []

    for batch in tqdm(train_data_loader, desc="Calculating training data histogram"):
        images = batch["image"][tio.DATA]
        batch_filenames = batch["filename"]
        for img, filename in zip(images, batch_filenames):
            train_histogram += np.histogram(img.numpy().flatten(), bins=histogram_bins, range=(range_min, range_max))[0]
            train_means.append(img.numpy().mean())
            train_sds.append(img.numpy().std())
            filenames.append(filename)

    return train_histogram, train_means, train_sds, filenames


def compare_histograms_and_means_sds(train_histogram, train_means, train_sds, inference_image, filenames):
    """Compare the training data histogram with the inference image histogram."""
    histogram_bins = 256
    range_min, range_max = -3, 3  # Define the range for the intensity values

    inference_histogram = np.histogram(inference_image.numpy().flatten(), bins=histogram_bins, range=(range_min, range_max))[0]

    bin_edges = np.linspace(range_min, range_max, histogram_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot histograms
    ax[0].plot(bin_centers, train_histogram, label='Training Data')
    ax[0].plot(bin_centers, inference_histogram, label='Inference Image', color='red')
    ax[0].set_xlabel('Intensity')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()

    # Calculate mean and standard deviation for the inference image
    inference_mean = inference_image.numpy().mean()
    inference_sd = inference_image.numpy().std()

    train_means.append(inference_mean)
    train_sds.append(inference_sd)
    filenames.append("Inference Image")

    x = np.arange(len(filenames))

    # Plot means and standard deviations
    ax[1].errorbar(x[:-1], train_means[:-1], yerr=train_sds[:-1], fmt='o', label='Training Images', ecolor='gray', capsize=5)
    ax[1].errorbar(x[-1], train_means[-1], yerr=train_sds[-1], fmt='o', color='red', label='Inference Image', ecolor='red', capsize=5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(filenames, rotation=90)
    ax[1].set_xlabel('Image Name')
    ax[1].set_ylabel('Intensity')
    ax[1].legend()

    plt.tight_layout()
    plt.show()
