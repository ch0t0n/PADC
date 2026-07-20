import os
import cv2
import random
import json
# from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import scipy
import cv2
from skimage.measure import regionprops, label, shannon_entropy
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from brisque import BRISQUE
from skvideo.measure import niqe
from pypiqe import piqe
from mahotas.features import zernike_moments
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import timm
import segmentation_models_pytorch as smp

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

# =========================
# GLOBAL SETTINGS
# =========================
DEVICE_ID = 0
IMAGE_SIZE = 224
EPOCHS = 20
LR = 1e-4
OPTIMIZER = "AdamW"
DATA_DIR = "/home/c/choton/beemachine/datasets/Others/CUB_200_2011/"
SEED = 42
NUM_WORKERS = 4
NUM_PARTS = 12 # Total number of parts including full-body

# Segmentation config
SEGMENTATION_NAME = "deeplabv3plus"
SEGMENTATION_ENCODER = "resnext50_32x4d"
SEGMENTATION_WEIGHTS = r"/home/c/choton/beemachine/codes/AG_vision_2026/2_segmentation/CUB/deeplabv3plus/lightning_logs/version_2/checkpoints/epoch=199-step=13400.ckpt"

# =========================
# PER-MODEL CONFIG
# =========================
MODEL_CONFIGS = {
    "convnext_nano.in12k": {"batch_size": 128},
    "deit_base_distilled_patch16_224.fb_in1k": {"batch_size": 128},
    "efficientnetv2_rw_m.agc_in1k": {"batch_size": 64},
    "efficientnetv2_rw_s.ra2_in1k": {"batch_size": 128},
    "inception_next_tiny.sail_in1k": {"batch_size": 128},
    "resnet101.a1_in1k": {"batch_size": 128},
    "seresnext101_32x4d.gluon_in1k": {"batch_size": 128},
    "swin_base_patch4_window7_224.ms_in1k": {"batch_size": 64},
}

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
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().astype(np.uint8)
    mask_u8 = mask.astype(np.uint8)
    image_f = (image - image.min()) / (image.max() - image.min() + 1e-8) # Linearly rescale to [0, 1] and avoid division by zero
    image_u8 = img_as_ubyte(image_f)

    combined_features = extract_shape_features(image_u8, mask_u8)
    vis_features = extract_visual_features(image_f, mask_u8)
    combined_features.update(vis_features)
    return combined_features

def extract_all_features(image, mask):
    """
    Generalized feature extraction for N body parts.
    mask: 2D array where each unique integer >0 is a body part ID.
    """

    # Get unique body-part IDs (excluding background 0)
    part_ids = sorted([pid for pid in np.unique(mask) if pid > 0])

    # Extract features for each part
    part_features = {}
    for pid in part_ids:
        part_mask = mask == pid
        feats = extract_combined_features(image, part_mask)
        part_features[pid] = feats

    # Full body features
    full_mask = mask > 0
    full_feats = extract_combined_features(image, full_mask)

    # Compute area sums
    areas = {pid: part_features[pid]["area"] for pid in part_ids}
    total_area = sum(areas.values()) + 1e-6

    # ----------- Ratio Features -----------
    # Pairwise ratios: area_i / area_j
    ratio_feats = {}
    for i in part_ids:
        for j in part_ids:
            if i != j:
                ratio_feats[f"area_ratio_part{i}_to_part{j}"] = areas[i] / (areas[j] + 1e-6)

    # Area-to-total ratios
    for pid in part_ids:
        ratio_feats[f"area_ratio_part{pid}_to_total"] = areas[pid] / total_area

    # ----------- Build Record -----------
    record = {}

    # Add each part's features
    for pid in part_ids:
        feats = part_features[pid]
        for k, v in feats.items():
            record[f"part{pid}_{k}"] = v

    # Add full-body features
    for k, v in full_feats.items():
        record[f"full_{k}"] = v

    # Add ratio features
    record.update(ratio_feats)
    return record

# Load classifier
def load_classifier(name, weights, num_classes, device_id):
    # Define the classification model and load weights
    model = timm.create_model(name, pretrained=False, num_classes=num_classes).to(f"cuda:{device_id}")
    state_dict_c = torch.load(weights, map_location=f"cuda:{device_id}")
    model.load_state_dict(state_dict_c) # Load the classifier state dict
    return model

# Load segmentation model
def load_segmentation(name, encoder_name, checkpoint_path):    
    # Define the segmentation model and load weights
    model = smp.create_model(
                name,
                encoder_name=encoder_name,
                in_channels = 3, # 3 for RGB images
                classes = NUM_PARTS
            ).to(f"cuda:{DEVICE_ID}")
    # checkpoint_path = r"/home/c/choton/beemachine/codes/everyday_test/oct1_25/segmentation_models/lightning_logs/fpn/lightning_logs/version_0/checkpoints/epoch=199-step=37400.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=rf"cuda:{DEVICE_ID}")

    # Sometimes Lightning adds "state_dict"
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

        # Remove "model." or "net." prefixes if present
        new_state_dict = {k.replace("model.", "").replace("net.", ""): v 
                        for k, v in state_dict.items() if k not in ["mean", "std"]}

        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    # model.to(f"cuda:{DEVICE_ID}")
    return model


# =========================
# UTILITIES
# =========================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_optimizer(name, model, lr):
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)

# =========================
# DATASET
# =========================
class CUBDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(DATA_DIR, "images", row["filepath"])
        image = Image.open(img_path).convert("RGB")
        label = row["label"]
        if self.transform:
            image = self.transform(image)
        return image, label, idx

def load_splits():
    images = pd.read_csv(os.path.join(DATA_DIR, "images.txt"), sep=" ", names=["id", "filepath"])
    labels = pd.read_csv(os.path.join(DATA_DIR, "image_class_labels.txt"), sep=" ", names=["id", "label"])
    splits = pd.read_csv(os.path.join(DATA_DIR, "train_test_split.txt"), sep=" ", names=["id", "is_train"])
    
    df = images.merge(labels, on="id").merge(splits, on="id")
    df["label"] -= 1

    train_df, temp_df = train_test_split(df, test_size=0.25, stratify=df["label"], random_state=SEED)
    val_ratio = 0.15 / (0.15 + 0.10)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_ratio),
                                       stratify=temp_df["label"], random_state=SEED)
    return train_df, val_df, test_df, df["label"].nunique()

def build_loaders(train_df, val_df, test_df, batch_size):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    train_ds = CUBDataset(train_df, train_tf)
    val_ds = CUBDataset(val_df, val_tf)
    test_ds = CUBDataset(test_df, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, test_loader

# =========================
# MODEL DEFINITION
# =========================

class ShapeEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(128, embed_dim)

    def forward(self, mask_tensor):
        """
        mask_tensor: (B, 3, 64, 64)
        """
        feat = self.encoder(mask_tensor)
        feat = feat.flatten(1)
        z = self.fc(feat)
        return z  

class GatedFusion(nn.Module):
    def __init__(self, img_dim, shape_dim):
        super().__init__()

        self.shape_proj = nn.Linear(shape_dim, img_dim)

        self.gate = nn.Sequential(
            nn.Linear(img_dim + img_dim, img_dim),
            nn.LayerNorm(img_dim),
            nn.Sigmoid()
        )

    def forward(self, z_img, z_shape):
        z_shape_proj = self.shape_proj(z_shape)

        concat = torch.cat([z_img, z_shape_proj], dim=1)
        g = self.gate(concat)

        fused = z_img + g * (z_shape_proj - z_img)
        return fused

class DeepShapeFusionModel(nn.Module):
    def __init__(self, num_classes, classifier_name, feature_size, shape_embed_dim=256):
        super().__init__()
        self.num_shape_feats = feature_size

        # Modules
        self.seg_module = load_segmentation(
            name=SEGMENTATION_NAME, 
            encoder_name=SEGMENTATION_ENCODER, 
            checkpoint_path=SEGMENTATION_WEIGHTS
        )
        self.backbone = timm.create_model(model_name=classifier_name, pretrained=True, num_classes=0)
        self.shape_encoder = ShapeEncoder(shape_embed_dim)

        self.fusion = GatedFusion(
            img_dim=self.backbone.num_features,
            shape_dim=shape_embed_dim
        )

        total_feats = self.backbone.num_features + self.num_shape_feats # + shape_embed_dim

        self.classifier = nn.Sequential(
                                nn.LayerNorm(total_feats),
                                nn.Dropout(0.3),
                                nn.Linear(total_feats, num_classes)
                            )

        for p in self.seg_module.parameters():
            p.requires_grad = False
        self.seg_module.eval()

    def forward(self, x, shape_feats=None):

        # ---- Segmentation ----
        with torch.no_grad():
            mask_logits = self.seg_module(x)
            masks = torch.argmax(mask_logits, dim=1)  # (B, H, W) with labels 0,1,2,3
        # mask_logits = self.seg_module(x)
        # mask_probs = torch.softmax(mask_logits, dim=1)  # example single part

        # Extract part embeddings
        parts = mask_logits[:, 1:, :, :]  # remove background, will try mask_probs as well
        mask_binary = parts.sum(dim=1, keepdim=True)
        edge = torch.abs(
            torch.nn.functional.avg_pool2d(mask_binary, 3, stride=1, padding=1)
            - mask_binary
        )
        shape_tensor = torch.cat([mask_binary, edge, mask_binary], dim=1)
        z_shape = self.shape_encoder(shape_tensor)

        # ---- Extract hand-crafted shape descriptors ----
        # ---- Shape features ----
        if shape_feats is None:
            batch_size = x.size(0)
            batch_feats = []
            for i in range(batch_size):
                img_np = x[i].detach().cpu().numpy().transpose(1, 2, 0)
                mask_np = masks[i].detach().cpu().numpy()
                feats_dict = extract_all_features(img_np, mask_np)
                # Convert dict to tensor (relies on consistent insertion order for 937 features)
                feat_tensor = torch.tensor(list(feats_dict.values()), dtype=torch.float32)
                batch_feats.append(feat_tensor)
            shape_feats = torch.stack(batch_feats).to(x.device)  # (B, 937)
        # shape_feats = torch.tensor(shape_feats).to(x.device)        

        # ---- Image Encoding ----
        z_img = self.backbone(x)
        fused = self.fusion(z_img, z_shape)
        combined = torch.cat([fused, shape_feats], dim=1)  # (B, total_feats)

        # ---- Classification ----
        out = self.classifier(combined)
        return out, mask_logits, z_shape


# =========================
# TRAIN / VALID
# =========================
def run_epoch(model, loader, criterion, device, optimizer=None, desc="[Train]"):
    training = optimizer is not None
    model.train() if training else model.eval()

    running_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(training):
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item())
    pbar.close()
    return running_loss / total, correct / total

# =========================
# TRAIN LOOP
# =========================
def train_model(classifier_name, batch_size):
    print(f"\n===== Training {classifier_name} | Batch Size={batch_size} =====")

    device = torch.device(f"cuda:{DEVICE_ID}")
    train_df, val_df, test_df, num_classes = load_splits()
    train_loader, val_loader, test_loader = build_loaders(train_df, val_df, test_df, batch_size)

    # model = timm.create_model(classifier_name, pretrained=True, num_classes=num_classes).to(device)
    model = DeepShapeFusionModel(num_classes=num_classes, shape_embed_dim=512)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(OPTIMIZER, model, LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    log_dir = f"./{model_name}_logs"
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, desc="[Train]")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, desc="[Val]")

        scheduler.step(val_loss)

        # Logging to console (same format as original)
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Logging to history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Checkpoints
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)

            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, checkpoint_path)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, device, desc="[Val]")
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    print(f"Experiment finished. Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Save logs
    pd.DataFrame(history).to_csv(os.path.join(log_dir, "training_log.csv"), index=False)
    with open(os.path.join(log_dir, "training_log.json"), "w") as f:
        json.dump(history, f, indent=4)

# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)
    for model_name, cfg in MODEL_CONFIGS.items():
        train_model(model_name, cfg["batch_size"])

if __name__ == "__main__":
    main()