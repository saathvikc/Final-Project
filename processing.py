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

# Additional edge detectors: Sobel (magnitude), Prewitt, Laplacian

# Prewitt kernels
Px = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

Py = np.array([
    [ 1,  1,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
], dtype=np.float32)

def sobel_edge_map(image):
    gx = convolve2d(image, Kx)
    gy = convolve2d(image, Ky)
    mag = np.sqrt(gx**2 + gy**2)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag

def prewitt_edge_map(image):
    gx = convolve2d(image, Px)
    gy = convolve2d(image, Py)
    mag = np.sqrt(gx**2 + gy**2)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag

def laplacian_edge_map(image):
    # 3x3 Laplacian kernel
    L = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)

    resp = convolve2d(image, L)
    mag = np.abs(resp)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag

# compute maps on the blurred grayscale image for fair comparison
sobel_map = sobel_edge_map(blurred)
prewitt_map = prewitt_edge_map(blurred)
laplacian_map = laplacian_edge_map(blurred)

# binary thresholded maps for visual comparison
def binary_from_mag(mag, thresh=0.2):
    return (mag >= thresh).astype(np.float32)

def otsu_threshold(image, nbins=256):
    # image assumed normalized to [0,1]
    img = (image * 255).astype(np.uint8)
    hist, bin_edges = np.histogram(img.ravel(), bins=nbins, range=(0, 255))
    total = img.size

    current_max, threshold = 0, 0
    sum_total = np.dot(np.arange(nbins), hist)
    sumB = 0
    wB = 0

    for i in range(nbins):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        # between class variance
        varBetween = wB * wF * (mB - mF) ** 2
        if varBetween > current_max:
            current_max = varBetween
            threshold = i

    return threshold / 255.0

sobel_bin = binary_from_mag(sobel_map, thresh=0.2)
prewitt_bin = binary_from_mag(prewitt_map, thresh=0.2)
laplacian_bin = binary_from_mag(laplacian_map, thresh=0.12)

# Morphological Closing

def dilate(binary_img, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(binary_img, pad, mode="constant")
    output = np.zeros_like(binary_img)

    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = 1 if np.any(region == 1) else 0

    return output


def erode(binary_img, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(binary_img, pad, mode="constant")
    output = np.zeros_like(binary_img)

    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = 1 if np.all(region == 1) else 0

    return output


def closing(binary_img, kernel_size=3):
    return erode(dilate(binary_img, kernel_size), kernel_size)


closed = closing(edges, kernel_size=5)

# Skeletonization

def skeletonize(binary_img):
    img = binary_img.copy().astype(np.uint8)
    changed = True

    while changed:
        changed = False
        pixels_to_remove = []

        # sub-iteration 1
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                P = [
                    img[i, j],
                    img[i-1, j],
                    img[i-1, j+1],
                    img[i, j+1],
                    img[i+1, j+1],
                    img[i+1, j],
                    img[i+1, j-1],
                    img[i, j-1],
                    img[i-1, j-1]
                ]

                if P[0] != 1:
                    continue

                neighbors = sum(P[1:])
                transitions = sum((P[k] == 0 and P[k+1] == 1) for k in range(1, 8))
                transitions += (P[8] == 0 and P[1] == 1)

                if (
                    2 <= neighbors <= 6 and
                    transitions == 1 and
                    P[1] * P[3] * P[5] == 0 and
                    P[3] * P[5] * P[7] == 0
                ):
                    pixels_to_remove.append((i, j))

        if pixels_to_remove:
            changed = True
            for i, j in pixels_to_remove:
                img[i, j] = 0

        pixels_to_remove = []

        # sub-iteration 2
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                P = [
                    img[i, j],
                    img[i-1, j],
                    img[i-1, j+1],
                    img[i, j+1],
                    img[i+1, j+1],
                    img[i+1, j],
                    img[i+1, j-1],
                    img[i, j-1],
                    img[i-1, j-1]
                ]

                if P[0] != 1:
                    continue

                neighbors = sum(P[1:])
                transitions = sum((P[k] == 0 and P[k+1] == 1) for k in range(1, 8))
                transitions += (P[8] == 0 and P[1] == 1)

                if (
                    2 <= neighbors <= 6 and
                    transitions == 1 and
                    P[1] * P[3] * P[7] == 0 and
                    P[1] * P[5] * P[7] == 0
                ):
                    pixels_to_remove.append((i, j))

        if pixels_to_remove:
            changed = True
            for i, j in pixels_to_remove:
                img[i, j] = 0

    return img


skeleton = skeletonize(closed)

# clean up

def remove_small_components(binary_img, min_size=30):
    img = binary_img.copy().astype(np.uint8)
    H, W = img.shape
    visited = np.zeros_like(img, dtype=bool)
    output = np.zeros_like(img)

    def get_neighbors(x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                if 0 <= nx < H and 0 <= ny < W:
                    neighbors.append((nx, ny))

        return neighbors

    for i in range(H):
        for j in range(W):
            if img[i, j] == 1 and not visited[i, j]:
                stack = [(i, j)]
                component = []
                visited[i, j] = True

                while stack:
                    x, y = stack.pop()
                    component.append((x, y))

                    for nx, ny in get_neighbors(x, y):
                        if img[nx, ny] == 1 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))

                if len(component) >= min_size:
                    for x, y in component:
                        output[x, y] = 1

    return output


clean_skeleton = remove_small_components(skeleton, min_size=75)



# figures

# plt.figure(figsize=(12, 8))

# plt.subplot(2, 2, 1)
# plt.title("Original")
# plt.imshow(img)
# plt.axis("off")

# plt.subplot(2, 2, 2)
# plt.title("Grayscale")
# plt.imshow(gray, cmap="gray")
# plt.axis("off")

# plt.subplot(2, 2, 3)
# plt.title("Gaussian Blur")
# plt.imshow(blurred, cmap="gray")
# plt.axis("off")

# plt.subplot(2, 2, 4)
# plt.title("Manual Canny Edges")
# plt.imshow(edges, cmap="gray")
# plt.axis("off")

# plt.tight_layout()
# plt.savefig("pipeline_output.png", dpi=300, bbox_inches='tight')
# plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Manual Canny Edges")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Morphological Closing")
plt.imshow(closed, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Skeletonized")
plt.imshow(skeleton, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Small Components Removed")
plt.imshow(clean_skeleton, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig("pipeline_output.png", dpi=300, bbox_inches="tight")
plt.show()

# Comparison figure for different edge detectors
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title("Canny (manual)")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Sobel (binary)")
plt.imshow(sobel_bin, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Prewitt (binary)")
plt.imshow(prewitt_bin, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Laplacian (binary)")
plt.imshow(laplacian_bin, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig("edge_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Compute Otsu thresholds for Sobel/Prewitt/Laplacian and show continuous maps
sobel_otsu_t = otsu_threshold(sobel_map)
prewitt_otsu_t = otsu_threshold(prewitt_map)
laplacian_otsu_t = otsu_threshold(laplacian_map)

print(f"Otsu thresholds -> Sobel: {sobel_otsu_t:.3f}, Prewitt: {prewitt_otsu_t:.3f}, Laplacian: {laplacian_otsu_t:.3f}")

sobel_otsu = binary_from_mag(sobel_map, thresh=sobel_otsu_t)
prewitt_otsu = binary_from_mag(prewitt_map, thresh=prewitt_otsu_t)
laplacian_otsu = binary_from_mag(laplacian_map, thresh=laplacian_otsu_t)

# Continuous magnitude comparison
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.title("Gradient Magnitude (Canny - pre-NMS)")
plt.imshow(magnitude, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Sobel Magnitude (continuous)")
plt.imshow(sobel_map, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Prewitt Magnitude (continuous)")
plt.imshow(prewitt_map, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Laplacian Response (continuous)")
plt.imshow(laplacian_map, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig("edge_continuous.png", dpi=300, bbox_inches="tight")
plt.show()

# Binary comparison using Otsu thresholds
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.title("Canny (manual)")
plt.imshow(edges, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title(f"Sobel (Otsu t={sobel_otsu_t:.2f})")
plt.imshow(sobel_otsu, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title(f"Prewitt (Otsu t={prewitt_otsu_t:.2f})")
plt.imshow(prewitt_otsu, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title(f"Laplacian (Otsu t={laplacian_otsu_t:.2f})")
plt.imshow(laplacian_otsu, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig("edge_binary_otsu.png", dpi=300, bbox_inches="tight")
plt.show()