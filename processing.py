import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "global_monthly_2018_01_mosaic_L15-0506E-1204N_2027_3374_13.tif"
img = Image.open(img_path).convert("RGB")
img = np.asarray(img).astype(np.float32) / 255.0

def rgb_to_gray(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return 0.299*r + 0.587*g + 0.114*b

gray = rgb_to_gray(img)

def convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# gaussian blur

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    return kernel

g_kernel = gaussian_kernel(size=5, sigma=1.0)
blurred = convolve2d(gray, g_kernel)

# canny edge detection

# Sobel filters
Kx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

Ky = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]
], dtype=np.float32)

Gx = convolve2d(blurred, Kx)
Gy = convolve2d(blurred, Ky)

magnitude = np.sqrt(Gx**2 + Gy**2)
magnitude = magnitude / magnitude.max()

angle = np.rad2deg(np.arctan2(Gy, Gx))
angle[angle < 0] += 180

# Non-maximum suppression

def non_max_suppression(mag, angle):
    H, W = mag.shape
    output = np.zeros((H, W), dtype=np.float32)

    for i in range(1, H-1):
        for j in range(1, W-1):

            q = 0
            r = 0

            a = angle[i, j]

            # 0 degrees
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]

            # 45 degrees
            elif 22.5 <= a < 67.5:
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]

            # 90 degrees
            elif 67.5 <= a < 112.5:
                q = mag[i+1, j]
                r = mag[i-1, j]

            # 135 degrees
            elif 112.5 <= a < 157.5:
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            if mag[i, j] >= q and mag[i, j] >= r:
                output[i, j] = mag[i, j]

    return output

thin_edges = non_max_suppression(magnitude, angle)

# double thresholding

def double_threshold(img, low=0.08, high=0.20):
    strong = 1.0
    weak = 0.5

    result = np.zeros_like(img)

    strong_pixels = img >= high
    weak_pixels = (img >= low) & (img < high)

    result[strong_pixels] = strong
    result[weak_pixels] = weak

    return result, weak, strong

thresholded, weak, strong = double_threshold(thin_edges, low=0.08, high=0.20)

# hysteresis

def hysteresis(img, weak=0.5, strong=1.0):
    H, W = img.shape
    result = img.copy()

    for i in range(1, H-1):
        for j in range(1, W-1):
            if result[i, j] == weak:
                neighborhood = result[i-1:i+2, j-1:j+2]

                if np.any(neighborhood == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0

    result[result == strong] = 1
    return result

edges = hysteresis(thresholded, weak, strong)

# figures

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Gaussian Blur")
plt.imshow(blurred, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Manual Canny Edges")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.tight_layout()

plt.savefig("pipeline_output.png", dpi=300, bbox_inches='tight')

plt.show()