import matplotlib.pyplot as plt

def plot_mnist(img):
    plt.imshow(img, cmap="RdGy", vmin=-img.max(), vmax=img.max())
    plt.colorbar(label='Amplitude')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    plt.tight_layout()