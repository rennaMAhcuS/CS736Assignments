import matplotlib.pyplot as plt
import numpy as np
from h5py import File as h5pyFile

# Load image from the .mat file
file_path = "1_ImageDenoising/data/assignmentImageDenoising_chestCT.mat"

with h5pyFile(file_path, "r") as file:
    image_data = file["imageChestCT"][:]

image = np.array(image_data).astype(np.float32)
image = (image - np.min(image)) / (np.max(image) - np.min(image))


def extract_high_variance_patches(main_image, patch_size=(8, 8), num_patches=10000, variance_thresh=0.01):
    patches = []
    for i in range(main_image.shape[0] - patch_size[0] + 1):
        for j in range(main_image.shape[1] - patch_size[1] + 1):
            patch = main_image[i:i + patch_size[0], j:j + patch_size[1]]
            if np.var(patch) > variance_thresh:
                patches.append(patch.flatten())
    return np.array(patches[:num_patches])


def normalize_columns(matrix):
    return matrix / np.linalg.norm(matrix, axis=0, keepdims=True)


def learn_dictionary(patches, n_components=64, iterations=500):
    np.random.seed(42)
    dictionary = np.random.randn(patches.shape[1], n_components)
    dictionary = normalize_columns(dictionary)

    for _ in range(iterations):
        coefficients = np.linalg.lstsq(dictionary, patches.T, rcond=None)[0]
        dictionary = np.linalg.lstsq(coefficients.T, patches, rcond=None)[0].T
        dictionary = normalize_columns(dictionary)

    return dictionary


def visualize_dictionary(dictionary, patch_size=(8, 8)):
    fig, axes = plt.subplots(8, 8, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(dictionary[:, i].reshape(patch_size), cmap='gray')
        ax.axis('off')
    plt.show()


def add_gaussian_noise(main_image, std_ratio=0.1):
    noise_std = std_ratio * (main_image.max() - main_image.min())
    noisy_image = main_image + np.random.normal(0, noise_std, main_image.shape)
    return np.clip(noisy_image, 0, 1)


def denoise_image(noisy_image, dictionary, patch_size=(8, 8)):
    denoised_image = np.zeros_like(noisy_image)
    count = np.zeros_like(noisy_image)

    for i in range(noisy_image.shape[0] - patch_size[0] + 1):
        for j in range(noisy_image.shape[1] - patch_size[1] + 1):
            patch = noisy_image[i:i + patch_size[0], j:j + patch_size[1]].flatten()
            coefficients = np.linalg.lstsq(dictionary, patch, rcond=None)[0]
            reconstructed_patch = np.dot(dictionary, coefficients).reshape(patch_size)
            denoised_image[i:i + patch_size[0], j:j + patch_size[1]] += reconstructed_patch
            count[i:i + patch_size[0], j:j + patch_size[1]] += 1

    return np.divide(denoised_image, count, where=count != 0)


def main():
    # Image is already loaded and normalized from the MAT file
    patches = extract_high_variance_patches(image)
    dictionary = learn_dictionary(patches)
    visualize_dictionary(dictionary)

    noisy_image = add_gaussian_noise(image)
    denoised_image = denoise_image(noisy_image, dictionary)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(noisy_image, cmap='gray')
    axs[1].set_title('Noisy')
    axs[2].imshow(denoised_image, cmap='gray')
    axs[2].set_title('Denoised')
    plt.show()


if __name__ == '__main__':
    main()
