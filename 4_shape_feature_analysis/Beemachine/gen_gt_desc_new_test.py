import os
import cv2
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from brisque import BRISQUE
from pypiqe import piqe
from skimage import img_as_ubyte
from skimage.measure import regionprops, label
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import laplace
from skvideo.measure import niqe
from mahotas.features import zernike_moments
from multiprocessing import Pool, cpu_count
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Patch imresize if missing
if not hasattr(scipy.misc, "imresize"):
    def imresize(arr, size, interp=None, mode=None):
        if isinstance(size, float):  # scale factor
            new_shape = (int(arr.shape[0] * size), int(arr.shape[1] * size))
        else:
            new_shape = size[:2]
        arr_resized = resize(arr, new_shape, order=3, mode="reflect", anti_aliasing=True)
        arr_resized = (arr_resized * 255).astype(np.uint8)
        return arr_resized
    scipy.misc.imresize = imresize

# Patch for deprecated NumPy aliases (for backward compatibility)
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

# ──────────────────────────────────────────────────────────────────────────────
# Constants / Globals
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 320
BATCH_SIZE = 128

sift = cv2.SIFT_create()
orb = cv2.ORB_create()
bri_obj = BRISQUE(url=False)


# ──────────────────────────────────────────────────────────────────────────────
# Shape / Morphological Descriptors
# ──────────────────────────────────────────────────────────────────────────────

def extract_base_features(mask):
    if mask.sum() == 0:
        return {
            "area": 0.0, "perimeter": 0.0, "aspect_ratio": 0.0,
            "extent": 0.0, "solidity": 0.0, "eccentricity": 0.0,
            "orientation": 0.0, "circularity": 0.0,
            "elongation": 0.0, "compactness": 0.0
        }

    labeled = label(mask.astype(np.uint8))
    props = regionprops(labeled)
    if not props:
        return {k: 0.0 for k in extract_base_features(np.zeros((1,1), bool))}

    p = props[0]
    major = p.major_axis_length
    minor = p.minor_axis_length

    area = float(p.area)
    perimeter = max(float(p.perimeter), 1e-6)
    aspect_ratio = major / minor if minor > 0 else 0.0
    extent = float(p.extent)
    solidity = float(p.solidity)
    eccentricity = float(p.eccentricity)
    orientation = float(p.orientation)
    circularity = 4 * np.pi * area / (perimeter ** 2)
    elongation = 1 - (minor / major) if major > 0 else 0.0
    compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)

    return {
        "area": area, "perimeter": perimeter, "aspect_ratio": aspect_ratio,
        "extent": extent, "solidity": solidity, "eccentricity": eccentricity,
        "orientation": orientation, "circularity": circularity,
        "elongation": elongation, "compactness": compactness
    }


def compute_sift_features(image, mask=None):
    gray = cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray, mask)
    if des is None:
        des = np.full((0, 128), np.nan, dtype=np.float32)
    return kp, des


def compute_orb_features(image, mask=None):
    gray = cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_RGB2GRAY)
    kp, des = orb.detectAndCompute(gray, mask)
    if des is None:
        des = np.full((0, 32), np.nan, dtype=np.float32)
    return kp, des


def compute_hu_moments(mask):
    moments = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    return np.log(np.abs(hu) + 1e-12).astype(np.float32)


def compute_zernike_moments(mask, degree=8):
    radius = min(mask.shape) // 2
    if radius < 1:
        return np.zeros(25, dtype=np.float32)  # safe fallback
    mask_norm = mask.astype(float) / (mask.max() + 1e-10)
    zern = zernike_moments(mask_norm, radius=radius, degree=degree)
    return zern.astype(np.float32)


def compute_fourier_descriptors(mask, fourier_harmonics=20):
    if not isinstance(mask, np.ndarray):
        mask = mask.numpy().astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.full(fourier_harmonics, np.nan, dtype=np.float32)

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return np.full(fourier_harmonics, np.nan, dtype=np.float32)

    complex_contour = cnt[:, 0, 0] + 1j * cnt[:, 0, 1]
    cnt_centered = complex_contour - np.mean(complex_contour)
    fd = np.fft.fft(cnt_centered)

    if len(fd) < 2 or np.abs(fd[1]) == 0:
        return np.full(fourier_harmonics, np.nan, dtype=np.float32)

    fd = fd / np.abs(fd[1])
    fd_mag = np.abs(fd)
    truncated = fd_mag[:fourier_harmonics]

    if len(truncated) < fourier_harmonics:
        truncated = np.concatenate([truncated, np.full(fourier_harmonics - len(truncated), np.nan)])

    return truncated.astype(np.float32)


def extract_shape_features(image, mask):
    base = extract_base_features(mask)

    # SIFT
    sift_kp, sift_des = compute_sift_features(image, mask)
    sift_n = len(sift_kp)
    sift_mean = np.nanmean(sift_des, axis=0) if sift_des.shape[0] > 0 else np.full(128, np.nan)
    sift_dict = {f"sift_ds{i+1}": float(sift_mean[i]) for i in range(128)}
    sift_dict.update({
        "sift_kp_n": float(sift_n),
        "sift_kp_size": float(max([k.size for k in sift_kp]) if sift_kp else 0)
    })

    # ORB
    orb_kp, orb_des = compute_orb_features(image, mask)
    orb_n = len(orb_kp)
    orb_mean = np.nanmean(orb_des, axis=0) if orb_des.shape[0] > 0 else np.full(32, np.nan)
    orb_dict = {f"orb_ds{i+1}": float(orb_mean[i]) for i in range(32)}
    orb_dict["orb_kp_n"] = float(orb_n)

    # Hu
    hu = compute_hu_moments(mask)
    hu_dict = {f"hu{i+1}": float(v) for i, v in enumerate(hu)}

    # Zernike — restored
    zern = compute_zernike_moments(mask)
    zern_dict = {f"zernike_{i+1}": float(v) for i, v in enumerate(zern)}

    # Fourier
    fourier = compute_fourier_descriptors(mask)
    fourier_dict = {f"fourier_{i+1}": float(v) for i, v in enumerate(fourier)}

    features = {**base, **sift_dict, **orb_dict, **hu_dict, **zern_dict, **fourier_dict}
    return {k: np.float32(v) for k, v in features.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Visual / Perceptual Quality Descriptors
# ──────────────────────────────────────────────────────────────────────────────

def pad_to_size(img, target_size=512):
    if img.ndim == 2:
        h, w = img.shape
        pad_h = target_size - h
        pad_w = target_size - w
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    elif img.ndim == 3:
        h, w, _ = img.shape
        pad_h = target_size - h
        pad_w = target_size - w
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    raise ValueError("Expected 2D or 3D image")


def extract_visual_features(image, mask):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)

    mask_exp = mask[..., None].astype(image.dtype)
    img_cropped = image * mask_exp

    gray = rgb2gray(img_cropped)
    region_mask = mask.astype(bool)

    if not region_mask.any():
        return {
            "brightness": 0.0, "contrast": 0.0, "sharpness": 0.0,
            "colorfulness": 0.0, "brisque": 0.0, "niqe": 0.0, "piqe": 0.0
        }

    gray_pixels = gray[region_mask]
    brightness = float(gray_pixels.mean())
    contrast   = float(gray_pixels.std())

    lap = laplace(gray)
    lap_var = float(lap[region_mask].var())

    R, G, B = cv2.split(img_cropped.astype(np.float32))
    rg = np.abs(R - G)[region_mask]
    yb = np.abs(0.5 * (R + G) - B)[region_mask]
    std_rg, std_yb = float(rg.std()), float(yb.std())
    mean_rg, mean_yb = float(rg.mean()), float(yb.mean())
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

    img_pad = pad_to_size(img_cropped)
    brisque_score = bri_obj.score(img_as_ubyte(img_pad))

    gray_u8 = img_as_ubyte(gray)
    niqe_score = niqe(gray_u8)
    piqe_score, _, _, _ = piqe(gray_u8)

    return {
        "brightness":   np.float32(brightness),
        "contrast":     np.float32(contrast),
        "sharpness":    np.float32(lap_var),
        "colorfulness": np.float32(colorfulness),
        "brisque":      np.float32(brisque_score),
        "niqe":         np.float32(niqe_score),
        "piqe":         np.float32(piqe_score)
    }


def extract_combined_features(image, mask):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy().astype(np.uint8)

    image_f = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image_u8 = img_as_ubyte(image_f)

    shape_feats = extract_shape_features(image_u8, mask)
    visual_feats = extract_visual_features(image_f, mask)

    return {**shape_feats, **visual_feats}


# ──────────────────────────────────────────────────────────────────────────────
# Full Feature Extraction (all parts + ratios)
# ──────────────────────────────────────────────────────────────────────────────

def extract_all_features(image, mask):
    masks = {
        "head":    (mask == 2),
        "thorax":  (mask == 3),
        "abdomen": (mask == 1),
        "full":    (mask > 0)
    }

    feats = {part: extract_combined_features(image, m) for part, m in masks.items()}

    h = feats["head"]["area"]
    t = feats["thorax"]["area"]
    a = feats["abdomen"]["area"]
    total = h + t + a + 1e-10

    ratios = {
        "head_to_thorax_area":    h / (t + 1e-6),
        "thorax_to_abdomen_area": t / (a + 1e-6),
        "head_to_total_area":     h / total,
        "thorax_to_total_area":   t / total,
        "abdomen_to_total_area":  a / total,
    }

    record = {}
    for part, fdict in feats.items():
        record.update({f"{part}_{k}": v for k, v in fdict.items()})
    record.update(ratios)

    return record


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PartWholeDataset(Dataset):
    def __init__(self, root, images_dir="aug_images", masks_dir="aug_masks", image_size=320):
        self.images_dir = os.path.join(root, images_dir)
        self.masks_dir  = os.path.join(root, masks_dir)
        self.image_paths = sorted(os.listdir(self.images_dir))
        self.image_paths = [os.path.join(self.images_dir, f) for f in self.image_paths]
        self.image_size = image_size

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.masks_dir, base + "_m.png")

        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask, img_path


# ──────────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR = r"/home/c/choton/beemachine/datasets/Beemachine_Partwhole_v5/"
    train_path = os.path.join(DATA_DIR, "train")
    val_path = os.path.join(DATA_DIR, "valid")
    test_path = os.path.join(DATA_DIR, "test")


    train_dataset = PartWholeDataset(
        root=train_path,
        images_dir="aug_images",
        masks_dir="aug_masks",
        image_size=IMAGE_SIZE
    )

    val_dataset = PartWholeDataset(
        root=val_path,
        images_dir="images",
        masks_dir="masks",
        image_size=IMAGE_SIZE
    )

    test_dataset = PartWholeDataset(
        root=test_path,
        images_dir="images",
        masks_dir="masks",
        image_size=IMAGE_SIZE
    )

    print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}, Test images: {len(test_dataset)}")

    def process_item(idx):
        image, mask, img_path = train_dataset[idx]
        features = extract_all_features(image, mask)
        record = {"image": os.path.basename(img_path)}
        record.update(features)
        return record

    def extract_features_for_dataset(dataset, output_csv):
        print(f"Extracting features for {len(dataset)} images...")
        num_procs = max(1, cpu_count() - 1)

        with Pool(processes=num_procs) as pool:
            records = list(tqdm(
                pool.imap(process_item, range(len(dataset))),
                total=len(dataset),
                desc="Extracting"
            ))

        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"Saved → {output_csv}")
        return df

    feat_path = r"./bee_gt_shape_features_grok_new"
    os.makedirs(feat_path, exist_ok=True)


    # train_csv = os.path.join(feat_path, "bee_gt_features_train.csv")
    val_csv = os.path.join(feat_path, "bee_gt_features_valid.csv")
    test_csv = os.path.join(feat_path, "bee_gt_features_test.csv")


    # train_df = extract_features_for_dataset(train_dataset, train_csv)
    val_df = extract_features_for_dataset(val_dataset, val_csv)
    test_df = extract_features_for_dataset(test_dataset, test_csv)