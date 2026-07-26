# Hyperparameters
DEVICE = 2
BATCH_SIZE = 128
IMAGE_SIZE = 320
DATA_DIR = r'/home/c/choton/beemachine/datasets/Others/fish-vista'

# Import libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE)
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy
# from scipy.fft import fft

from mahotas.features import zernike_moments
from brisque import BRISQUE # NR-IQA
from pypiqe import piqe # NR-IQA

from skimage import img_as_ubyte
from skimage.measure import regionprops, label, shannon_entropy
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import laplace
from sklearn.model_selection import train_test_split
from skvideo.measure import niqe # NR-IQA

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Other DL related libraries
import timm
import segmentation_models_pytorch as smp
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, infer_device
from ultralytics import SAM

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

# Create SIFT and ORB detectors once
sift = cv2.SIFT_create()
orb = cv2.ORB_create()
bri_obj = BRISQUE(url=False)

def extract_base_features(mask):
    """Compute geometric, Zernike, Fourier, and texture shape descriptors from a binary mask."""
    
    features = ["area", "perimeter", "aspect_ratio", "extent", "solidity", "eccentricity", 
        "orientation", "circularity", "elongation", "compactness"]
    
    if mask is None or mask.sum() == 0:
        return {f: 0 for f in features}

    # --- Region properties ---
    # mask = mask.astype(np.uint8)
    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) == 0:
        return {f: 0 for f in features}
    p = props[0]
    major_axis = p.major_axis_length
    minor_axis = p.minor_axis_length

    # ----- base shape features -----
    area = p.area
    perimeter = max(p.perimeter, 1e-6) # Ignoring too small perimeters
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0 # L_major / L_minor
    extent = p.extent
    solidity = p.solidity
    eccentricity = p.eccentricity
    orientation = p.orientation
    circularity = 4 * np.pi * area / (perimeter ** 2)
    elongation = 1 - (minor_axis / major_axis) if major_axis > 0 else 0
    # convexity = p.perimeter_convex / perimeter
    compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)

    # ----- Assemble features -----
    features_d = {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "orientation": orientation,
        "circularity": circularity,
        "elongation": elongation,
        "compactness": compactness
    }
    return features_d

def compute_sift_features(image, mask=None):
    # if isinstance(image, torch.Tensor):
    #     image = image.detach().cpu().numpy().transpose(1, 2, 0) # Transform tensor to numpy image
    # if isinstance(mask, torch.Tensor):
    #     mask = mask.numpy().astype(np.uint8)
    gray= cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_RGB2GRAY) # converts image into uint8 and mask as input
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if descriptors is None:
        descriptors = np.full((0, 128), np.nan, dtype=np.float32)  # return empty array if no keypoints
    return keypoints, descriptors

def compute_orb_features(image, mask=None):
    # if isinstance(image, torch.Tensor):
    #     image = image.detach().cpu().numpy().transpose(1, 2, 0) # Transform tensor to numpy image
    # if isinstance(mask, torch.Tensor):
    #     mask = mask.numpy().astype(np.uint8)
    gray= cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    if descriptors is None:
        descriptors = np.full((0, 32), np.nan, dtype=np.float32)  # return empty array if no keypoints
    return keypoints, descriptors

def compute_hu_moments(mask):
    # if not isinstance(mask, np.ndarray):
    #     mask = mask.numpy().astype(np.uint8)
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    hu = np.log(np.abs(hu) + 1e-12) # log-scale for stability
    return hu

def compute_zernike_moments(mask, degree=8):
    # if not isinstance(mask, np.ndarray):
    #     mask = mask.numpy().astype(np.uint8)
    radius = min(mask.shape) // 2
    mask_norm = mask / mask.max() if mask.max() > 0 else mask
    zern = zernike_moments(mask_norm, radius=radius, degree=degree)
    return zern

# *** Updated fourier descriptors (Dec 4, 2025)
def compute_fourier_descriptors(mask, image=None, fourier_harmonics=20, visualize=False):
    if not isinstance(mask, np.ndarray): # Ensure proper mask format
        mask = mask.numpy().astype(np.uint8)
    # --- 2. Find largest contour (object part) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.full(fourier_harmonics, np.nan, dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        # Too few points for Fourier transform
        return np.full(fourier_harmonics, np.nan, dtype=np.float32)
    
    # Translation invariance: center contour
    complex_contour = cnt[:,0,0] + 1j * cnt[:,0,1]
    fd = np.fft.fft(complex_contour)
    
    if visualize: # ** IMPORTANT: Visualization uses raw contour (so you can see the real shape), descriptors are centered.
        # Convert image if needed
        H, W = mask.shape
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy().transpose(1, 2, 0)
            elif isinstance(image, Image.Image):
                image = np.array(image.convert('RGB'))
            elif image.dtype != np.uint8:  # NumPy float → uint8
                image = (image*255).astype(np.uint8)
            img_draw = image.copy()
        else:
            img_draw = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.drawContours(img_draw, [cnt.astype(np.int32)], -1, (0, 255, 0), 2)

        fd_recon = fd.copy()
        keep = fourier_harmonics
        if 2 * keep < len(fd_recon):
            fd_recon[keep:-keep] = 0 # Safe zeroing
        else:
            fd_recon[keep:] = 0
        recon = np.fft.ifft(fd_recon)
        pts = np.column_stack((recon.real, recon.imag)).astype(np.int32)

        for i in range(len(pts)):
            cv2.line(img_draw, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (255, 0, 255), 1)
        plt.figure(figsize=(16, 6))
        plt.imshow(img_draw)
        plt.axis('off')
        plt.title("Shape Descriptors Overlay")
        plt.legend(
            handles=[
                Patch(facecolor='green', edgecolor='green'),
                Patch(facecolor='magenta', edgecolor='magenta')
            ],
            labels=["Contour", "Fourier Reconstruction"],
            loc='upper right'
        )
        plt.show()
    
    cnt_centered = complex_contour - np.mean(complex_contour)
    fd = np.fft.fft(cnt_centered)
    if len(fd) < 2 or np.abs(fd[1]) == 0:
        return np.full(fourier_harmonics, np.nan, dtype=np.float32)

    # Scale invariance: divide by first descriptor magnitude
    fd = fd / np.abs(fd[1])

    # Rotation invariance: use only magnitudes
    fd_normalized = np.abs(fd)

    # Keep only first N harmonics
    fd_truncated = fd_normalized[:fourier_harmonics]
    if len(fd_truncated) < fourier_harmonics:
        fd_truncated = np.concatenate([fd_truncated, np.full((fourier_harmonics - len(fd_truncated)), np.nan)])
    return fd_truncated

def extract_shape_features(image, mask):
    # Compute base features
    features = extract_base_features(mask)

    # Compute sift features
    sift_kp, sift_ds = compute_sift_features(image, mask)
    sift_sizes = [k.size for k in sift_kp]
    if sift_ds.shape[0] > 0:
        sift_mean_ds = np.nanmean(sift_ds, axis=0)
    else:
        sift_mean_ds = np.full(128, np.nan)
    sift_dict = {f'sift_ds{i+1}': sift_mean_ds[i] for i in range(len(sift_mean_ds))}
    sift_dict['sift_kp_n'] = len(sift_kp)
    sift_dict['sift_kp_size'] = np.max(sift_sizes) if sift_sizes else 0

    # Compute orb features
    orb_kp, orb_ds = compute_orb_features(image, mask)
    if orb_ds.shape[0] > 0:
        orb_mean_ds = np.nanmean(orb_ds, axis=0)
    else:
        orb_mean_ds = np.full(32, np.nan)
    orb_dict = {f'orb_ds{i+1}': orb_mean_ds[i] for i in range(len(orb_mean_ds))}
    orb_dict['orb_kp_n'] = len(orb_kp)

    # Compute hu moments
    hu_moments = compute_hu_moments(mask)
    hu_dict = {f"hu{i+1}": hu_moments[i] for i in range(len(hu_moments))}

    # Compute Zernike moments
    zern_moments = compute_zernike_moments(mask, degree=8)
    zern_dict = {f"zernike_{i+1}": zern_moments[i] for i in range(len(zern_moments))}

    # Compute fourier descriptors
    fourier_ds = compute_fourier_descriptors(mask, fourier_harmonics=20)
    fourier_dict = {f"fourier_{i+1}": fourier_ds[i] for i in range(len(fourier_ds))}

    features.update(sift_dict)
    features.update(orb_dict)
    features.update(hu_dict)
    features.update(zern_dict)
    features.update(fourier_dict)
    converted = {k: np.float32(v) for k, v in features.items()}
    return converted

def extract_visual_features(image, mask):
    # --- 1. Ensure binary uint8 mask ---
    if not isinstance(mask, np.ndarray):
        mask = mask.numpy().astype(np.uint8)
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
    elif isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    img_cropped = np.zeros_like(image)
    img_cropped[mask==1] = image[mask==1]
    # plt.imshow(img_cropped)

    # --- Brightness ---
    brightness = np.mean(img_cropped)

    # --- Contrast (standard deviation of luminance) ---
    gray = rgb2gray(img_cropped)
    contrast = np.std(gray)

    # --- Sharpness (variance of Laplacian) ---
    gray_8u = (gray * 255).astype(np.uint8)
    lap_var = cv2.Laplacian(gray_8u, cv2.CV_64F).var()

    # --- Colorfulness (Hasler & Süsstrunk, 2003) ---
    (R, G, B) = cv2.split(img_cropped)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

    # --- Entropy (texture complexity) ---
    entropy = shannon_entropy(gray)

    # BRISQUE
    bri_obj = BRISQUE(url=False)
    brisque_score = bri_obj.score(img_cropped)

    # NIQE
    niqe_score = niqe(gray)

    # PIQE
    piqe_score, activityMask, noticeableArtifactMask, noiseMask = piqe(gray)

    # --- Aggregate descriptors ---
    descriptors = {
        "brightness": np.float32(brightness),
        "contrast": np.float32(contrast),
        "sharpness": np.float32(lap_var),
        "colorfulness": np.float32(colorfulness),
        "entropy": np.float32(entropy),
        "brisque": np.float32(brisque_score),
        "niqe": np.float32(niqe_score.item()),
        "piqe": np.float32(piqe_score)
    }
    return descriptors

def extract_combined_features(image, mask): 
    # ---- Convert once ----
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().astype(np.uint8)
    if (mask is None) or (mask.size == 0) or (not mask.any()):
        return {}
    mask_u8 = mask.astype(np.uint8)
    image_f = (image - image.min()) / (image.max() - image.min() + 1e-8) # Linearly rescale to [0, 1] and avoid division by zero
    image_u8 = img_as_ubyte(image_f)

    combined_features = extract_shape_features(image_u8, mask_u8)
    vis_features = extract_visual_features(image_f, mask_u8)
    combined_features.update(vis_features)
    return combined_features

# Load the GroundingDINO model
model_dino_id = "IDEA-Research/grounding-dino-base"
device = infer_device()
processor = AutoProcessor.from_pretrained(model_dino_id)
model_dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_dino_id).to(device)

# Load the SAM 2 model
model_sam2 = SAM("/home/c/choton/fishvistamachine/codes/sam2_ultralytics_weights/sam2.1_b.pt")

def generate_full_body_mask(image_path, text_prompt):
    no_box, no_mask = False, False 
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image: {image_path}, {e}")
        return None

    text_labels = [[text_prompt]]
    W, H = image.size
    nan_mask = np.full((H, W), np.nan, dtype=float)
    
    try:
        # Grounding DINO
        inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model_dino.device)
        with torch.no_grad():
            outputs = model_dino(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.35,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )       
        boxes = results[0].get("boxes", None)
    except Exception as e:
        print(f"GroundingDINO failure on {image_path}: {e}")
        boxes = None
        no_box = True    
    if boxes is None or len(boxes) == 0:
        # ← fallback: full-image box for SAM2
        # print("No object detected using DINO for image:", image_path)
        w, h = image.size
        box = np.array([0, 0, w, h], dtype=np.float32)
        no_box = True
    else:
        box = boxes[0].cpu().numpy()    
    try:
        # Use SAM 2 and generate masks
        results = model_sam2(image_path, bboxes=[box], verbose=False) # This will return a single, most confident mask for the given prompt
        masks = results[0].masks.data.cpu().numpy()
        if masks.shape[0] > 1: 
            # choose best mask, e.g. by area
            areas = [mask.sum() for mask in masks]
            idx = np.argmax(areas)
            mask = masks[idx]
        else:
            mask = masks[0]
    except Exception as e:
        print(f"SAM2 failure on {image_path}: {e}")
        mask = nan_mask
        no_mask = True
    if (mask is None) or (mask.size == 0) or (not mask.any()):
        print("No mask detected using SAM 2 for image:", image_path)
        mask = nan_mask
        no_mask = True
    return mask, no_box, no_mask #, image_mask

# Load the classification splits and check the shape
train_csv = os.path.join(DATA_DIR, r'classification_train.csv')
val_csv = os.path.join(DATA_DIR, r'classification_val.csv')
test_csv = os.path.join(DATA_DIR, r'classification_test.csv')
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

classes = sorted(train_df['standardized_species'].unique())
num_classes = len(classes)
class_to_idx = {cls: i for i, cls in enumerate(classes)}
print(f'Shape of FishVista classification datasets,  train: {train_df.shape}, validation: {val_df.shape}, test): {test_df.shape}')
print(f'Columns of the test dataset:', list(test_df.columns))

class FishVistaDataset(Dataset):
    def __init__(self, dataframe, class_to_idx, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(DATA_DIR, "Images", row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row["standardized_species"]]
        if self.transform:
            image = self.transform(image)
        return image, img_path, label

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
])

train_dataset = FishVistaDataset(train_df, class_to_idx, val_transform)
val_dataset = FishVistaDataset(val_df, class_to_idx, val_transform)
test_dataset = FishVistaDataset(test_df, class_to_idx, val_transform)

print(f"Classes: {num_classes} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

image, img_path, _ = train_dataset[8]
text_prompt = 'fish'
mask, no_box, no_mask = generate_full_body_mask(img_path, text_prompt)
if mask is None:
    mask = np.full((IMAGE_SIZE, IMAGE_SIZE), np.nan, dtype=np.float32)
if mask.ndim > 2:
    mask = mask.squeeze() # Make the mask 2D
mask = torch.from_numpy(np.array(mask, dtype=np.float32))
mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W) for interpolate
mask = torch.nn.functional.interpolate(mask, size=(IMAGE_SIZE,IMAGE_SIZE), mode='nearest')  # keep discrete values
mask = mask.squeeze(0).squeeze(0)  # back to (H,W)

s_feats = extract_combined_features(image=image, mask=mask)
print(s_feats)
s_df = pd.DataFrame.from_dict([s_feats])

def extract_features_for_dataset(dataset, output_csv, text_prompt='fish'):
    records = []
    no_boxes, no_masks = 0, 0
    print(f"Extracting shape features for {len(dataset)} images...")
    pbar = tqdm(dataset, desc=f"empty_boxes={no_boxes}, empty_masks={no_masks}")
    for image, img_path, class_id in pbar:
        mask, no_box, no_mask = generate_full_body_mask(img_path, text_prompt)
        if mask is None:
            mask = np.full((IMAGE_SIZE, IMAGE_SIZE), np.nan, dtype=np.float32)
        if mask.ndim > 2:
            mask = mask.squeeze() # Make the mask 2D
        mask = torch.from_numpy(np.array(mask, dtype=np.float32))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W) for interpolate
        mask = torch.nn.functional.interpolate(mask, size=(IMAGE_SIZE,IMAGE_SIZE), mode='nearest')  # keep discrete values
        mask = mask.squeeze(0).squeeze(0)  # back to (H,W)

        if no_box:
            no_boxes += 1
            pbar.set_description(f"empty_boxes={no_boxes}, empty_masks={no_masks}")
        if no_mask:
            no_masks += 1
            pbar.set_description(f"empty_boxes={no_boxes}, empty_masks={no_masks}")            
        features = extract_combined_features(image=image, mask=mask)
        record = {"image": os.path.basename(img_path),
                  "class_id": class_id}
        record.update(features)
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved shape features to: {output_csv}")
    return df

feat_path = r"./fishvista_zeroshot_fullbody_features_concise"
os.makedirs(feat_path, exist_ok=True)

train_csv = os.path.join(feat_path, r"fishvista_zeroshot_fullbody_features_train.csv")
df = extract_features_for_dataset(train_dataset, train_csv)

val_csv = os.path.join(feat_path, r"fishvista_zeroshot_fullbody_features_val.csv")
df = extract_features_for_dataset(val_dataset, val_csv)

test_csv = os.path.join(feat_path, r"fishvista_zeroshot_fullbody_features_test.csv")
df = extract_features_for_dataset(test_dataset, test_csv)