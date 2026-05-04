import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

INPUT_DIR = Path("images")
OUTPUT_DIR = Path("outputs")
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def rgb_to_gray(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i : i + kh, j : j + kw]
            output[i, j] = np.sum(region * kernel)

    return output

# gaussian blur

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    return kernel


def gaussian_blur(image, sigma=1.0, size=None):
    if size is None:
        size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    kernel = gaussian_kernel(size=size, sigma=sigma)
    return convolve2d(image, kernel)


# Sobel filters
Kx = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)

Ky = np.array(
    [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ],
    dtype=np.float32,
)


# Prewitt kernels
Px = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)

Py = np.array(
    [
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ],
    dtype=np.float32,
)


def non_max_suppression(mag, angle):
    H, W = mag.shape
    output = np.zeros((H, W), dtype=np.float32)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 0
            r = 0

            a = angle[i, j]

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif 22.5 <= a < 67.5:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            elif 112.5 <= a < 157.5:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]

            if mag[i, j] >= q and mag[i, j] >= r:
                output[i, j] = mag[i, j]

    return output


def double_threshold(img, low=0.08, high=0.20):
    strong = 1.0
    weak = 0.5

    result = np.zeros_like(img)

    strong_pixels = img >= high
    weak_pixels = (img >= low) & (img < high)

    result[strong_pixels] = strong
    result[weak_pixels] = weak

    return result, weak, strong


def hysteresis(img, weak=0.5, strong=1.0):
    H, W = img.shape
    result = img.copy()

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if result[i, j] == weak:
                neighborhood = result[i - 1 : i + 2, j - 1 : j + 2]

                if np.any(neighborhood == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0

    result[result == strong] = 1
    return result


def gradient_mag_angle(image):
    gx = convolve2d(image, Kx)
    gy = convolve2d(image, Ky)
    mag = np.sqrt(gx**2 + gy**2)
    if mag.max() > 0:
        mag = mag / mag.max()
    angle = np.rad2deg(np.arctan2(gy, gx))
    angle[angle < 0] += 180
    return mag, angle


def percentile_thresholds(mag, low_pct=70, high_pct=90):
    vals = mag[mag > 0]
    if vals.size == 0:
        return 0.0, 0.0
    low = np.percentile(vals, low_pct)
    high = np.percentile(vals, high_pct)
    if high < low:
        high = low
    return low, high


def canny_edges(image, sigma=1.0, low_pct=70, high_pct=90):
    blurred_img = gaussian_blur(image, sigma=sigma)
    mag, angle = gradient_mag_angle(blurred_img)
    thin = non_max_suppression(mag, angle)
    low, high = percentile_thresholds(thin, low_pct, high_pct)
    thresholded, weak, strong = double_threshold(thin, low=low, high=high)
    edges = hysteresis(thresholded, weak, strong)
    return edges, mag, thin, (low, high)


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
    L = np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    resp = convolve2d(image, L)
    mag = np.abs(resp)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


def binary_from_mag(mag, thresh=0.2):
    return (mag >= thresh).astype(np.float32)


def otsu_threshold(image, nbins=256):
    img = (image * 255).astype(np.uint8)
    hist, _ = np.histogram(img.ravel(), bins=nbins, range=(0, 255))
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
        varBetween = wB * wF * (mB - mF) ** 2
        if varBetween > current_max:
            current_max = varBetween
            threshold = i

    return threshold / 255.0


def dilate(binary_img, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(binary_img, pad, mode="constant")
    output = np.zeros_like(binary_img)

    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            region = padded[i : i + kernel_size, j : j + kernel_size]
            output[i, j] = 1 if np.any(region == 1) else 0

    return output


def erode(binary_img, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(binary_img, pad, mode="constant")
    output = np.zeros_like(binary_img)

    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            region = padded[i : i + kernel_size, j : j + kernel_size]
            output[i, j] = 1 if np.all(region == 1) else 0

    return output


def closing(binary_img, kernel_size=3):
    return erode(dilate(binary_img, kernel_size), kernel_size)


def skeletonize(binary_img):
    img = binary_img.copy().astype(np.uint8)
    changed = True

    while changed:
        changed = False
        pixels_to_remove = []

        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                P = [
                    img[i, j],
                    img[i - 1, j],
                    img[i - 1, j + 1],
                    img[i, j + 1],
                    img[i + 1, j + 1],
                    img[i + 1, j],
                    img[i + 1, j - 1],
                    img[i, j - 1],
                    img[i - 1, j - 1],
                ]

                if P[0] != 1:
                    continue

                neighbors = sum(P[1:])
                transitions = sum((P[k] == 0 and P[k + 1] == 1) for k in range(1, 8))
                transitions += P[8] == 0 and P[1] == 1

                if (
                    2 <= neighbors <= 6
                    and transitions == 1
                    and P[1] * P[3] * P[5] == 0
                    and P[3] * P[5] * P[7] == 0
                ):
                    pixels_to_remove.append((i, j))

        if pixels_to_remove:
            changed = True
            for i, j in pixels_to_remove:
                img[i, j] = 0

        pixels_to_remove = []

        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                P = [
                    img[i, j],
                    img[i - 1, j],
                    img[i - 1, j + 1],
                    img[i, j + 1],
                    img[i + 1, j + 1],
                    img[i + 1, j],
                    img[i + 1, j - 1],
                    img[i, j - 1],
                    img[i - 1, j - 1],
                ]

                if P[0] != 1:
                    continue

                neighbors = sum(P[1:])
                transitions = sum((P[k] == 0 and P[k + 1] == 1) for k in range(1, 8))
                transitions += P[8] == 0 and P[1] == 1

                if (
                    2 <= neighbors <= 6
                    and transitions == 1
                    and P[1] * P[3] * P[7] == 0
                    and P[1] * P[5] * P[7] == 0
                ):
                    pixels_to_remove.append((i, j))

        if pixels_to_remove:
            changed = True
            for i, j in pixels_to_remove:
                img[i, j] = 0

    return img


def filter_long_thin_components(binary_img, skeleton_img, min_length=150, max_thickness=3.0):
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

                area = len(component)
                length = sum(1 for x, y in component if skeleton_img[x, y] == 1)
                thickness = area / max(length, 1)

                if length >= min_length and thickness <= max_thickness:
                    for x, y in component:
                        output[x, y] = 1

    return output


def list_image_files(input_dir):
    if not input_dir.exists():
        return []
    return sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
    )


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_image(image_path, output_root):
    img = Image.open(image_path).convert("RGB")
    img = np.asarray(img).astype(np.float32) / 255.0
    gray = rgb_to_gray(img)

    blurred = gaussian_blur(gray, sigma=1.0)

    edges_s1, mag_s1, thin_s1, (low1, high1) = canny_edges(
        gray, sigma=1.0, low_pct=70, high_pct=90
    )
    edges_s2, mag_s2, thin_s2, (low2, high2) = canny_edges(
        gray, sigma=2.0, low_pct=70, high_pct=90
    )
    edges = np.maximum(edges_s1, edges_s2)
    canny_mag = mag_s1

    print(
        f"[{image_path.name}] Canny percentiles -> "
        f"sigma=1.0: low={low1:.3f}, high={high1:.3f}; "
        f"sigma=2.0: low={low2:.3f}, high={high2:.3f}"
    )

    sobel_map = sobel_edge_map(blurred)
    prewitt_map = prewitt_edge_map(blurred)
    laplacian_map = laplacian_edge_map(blurred)

    sobel_bin = binary_from_mag(sobel_map, thresh=0.2)
    prewitt_bin = binary_from_mag(prewitt_map, thresh=0.2)
    laplacian_bin = binary_from_mag(laplacian_map, thresh=0.12)

    closed = closing(edges, kernel_size=5)
    skeleton = skeletonize(closed)
    road_candidates = filter_long_thin_components(
        closed, skeleton, min_length=150, max_thickness=3.0
    )
    road_skeleton = skeletonize(road_candidates)

    sobel_otsu_t = otsu_threshold(sobel_map)
    prewitt_otsu_t = otsu_threshold(prewitt_map)
    laplacian_otsu_t = otsu_threshold(laplacian_map)

    print(
        f"[{image_path.name}] Otsu thresholds -> "
        f"Sobel: {sobel_otsu_t:.3f}, Prewitt: {prewitt_otsu_t:.3f}, "
        f"Laplacian: {laplacian_otsu_t:.3f}"
    )

    sobel_otsu = binary_from_mag(sobel_map, thresh=sobel_otsu_t)
    prewitt_otsu = binary_from_mag(prewitt_map, thresh=prewitt_otsu_t)
    laplacian_otsu = binary_from_mag(laplacian_map, thresh=laplacian_otsu_t)

    out_dir = output_root / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title("Canny Edges (multi-scale)")
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
    plt.title("Road Skeleton (filtered)")
    plt.imshow(road_skeleton, cmap="gray")
    plt.axis("off")

    save_figure(fig, out_dir / "pipeline_output.png")

    fig = plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title("Canny (multi-scale)")
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

    save_figure(fig, out_dir / "edge_comparison.png")

    fig = plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title("Gradient Magnitude (Canny - pre-NMS)")
    plt.imshow(canny_mag, cmap="gray")
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

    save_figure(fig, out_dir / "edge_continuous.png")

    fig = plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title("Canny (multi-scale)")
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

    save_figure(fig, out_dir / "edge_binary_otsu.png")


def main():
    image_paths = list_image_files(INPUT_DIR)
    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        process_image(image_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()
