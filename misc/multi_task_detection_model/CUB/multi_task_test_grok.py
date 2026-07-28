import os
import numpy as np
from tqdm import tqdm
import pandas as pd

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

import segmentation_models_pytorch as smp

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME    = "manet"
ENCODER_NAME  = "resnext50_32x4d"
EPOCHS        = 100
BATCH_SIZE    = 64
IMAGE_SIZE    = 320
SEED          = 42
DEVICE_IDS    = [3]

DATASET_ROOT = r'/home/c/choton/beemachine/datasets/Others/CUB_200_2011/'

imgs_dir    = os.path.join(DATASET_ROOT, 'images')
mask_dir    = os.path.join(DATASET_ROOT, 'AnnotationMasksPerclass')
part_label_path = os.path.join(DATASET_ROOT, 'part_labels.txt')
classes_path    = os.path.join(DATASET_ROOT, 'classes.txt')
image_names_path = os.path.join(DATASET_ROOT, 'images.txt')

# ==========================================
# HELPERS
# ==========================================
def read_part_labels(path):
    with open(path, "r") as f:
        labels = [l.strip() for l in f if l.strip()]
    labels.insert(0, "background")
    return labels

classes_pd = pd.read_csv(classes_path, sep=" ", header=None, names=["id", "name"])
species_list = classes_pd["name"].tolist()

labels = read_part_labels(part_label_path)
num_parts = len(labels)
print(f"Part labels (including background): {labels}")
print(f"Number of part classes: {num_parts}")

# ==========================================
# DATASET (aligned with cub_manet.py + added species label)
# ==========================================
class PartWholeMultiTaskDataset(Dataset):
    def __init__(self, image_size, part_labels):
        self.image_size = image_size
        self.part_labels = part_labels

        # ── 1. Discover which classes actually have part masks ────────────────
        available_class_dirs = [
            d for d in os.listdir(mask_dir)
            if os.path.isdir(os.path.join(mask_dir, d)) and d.isdigit()
        ]
        self.annotated_class_ids = sorted({int(d) for d in available_class_dirs})   # e.g. [1, 3, 7, ..., 192]

        print(f"Classes with part annotations: {len(self.annotated_class_ids)}")
        print("Class IDs:", self.annotated_class_ids)

        # Create mapping: original class id → consecutive label 0,1,2,...
        self.class_id_to_label = {cid: idx for idx, cid in enumerate(self.annotated_class_ids)}
        self.num_species = len(self.annotated_class_ids)

        # ── 2. Load image metadata ────────────────────────────────────────────
        image_names = pd.read_csv(image_names_path, sep=" ", header=None, names=["id", "name"])

        # Only keep images from annotated classes
        self.image_paths = []
        self.species_labels = []   # will store the consecutive 0-based label

        print("Collecting images from classes with part masks...")
        for _, row in tqdm(image_names.iterrows(), total=len(image_names)):
            img_rel_path = row["name"]          # e.g. 001.Black_Footed_Albatross/xxx.jpg
            class_name = os.path.basename(os.path.dirname(img_rel_path))
            try:
                orig_class_id = species_list.index(class_name) + 1
            except ValueError:
                continue

            if orig_class_id not in self.annotated_class_ids:
                continue

            img_path = os.path.join(imgs_dir, img_rel_path)

            # Quick existence + has-mask check
            if not os.path.isfile(img_path):
                continue

            image_stem = os.path.splitext(os.path.basename(img_rel_path))[0]
            mask_class_dir = os.path.join(mask_dir, str(orig_class_id))
            mask_files = [f for f in os.listdir(mask_class_dir) if image_stem in f]

            if len(mask_files) == 0:
                continue

            self.image_paths.append(img_path)
            # Store the remapped consecutive label
            self.species_labels.append(self.class_id_to_label[orig_class_id])

        print(f"Final number of images with part masks: {len(self.image_paths)}")

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        species_label = self.species_labels[idx]   # already 0-based consecutive

        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        # ── Load part mask (same logic as before) ─────────────────────────────
        image_stem = os.path.splitext(os.path.basename(img_path))[0]
        # We need original class id → find it from the remapped label
        orig_class_id = self.annotated_class_ids[species_label]
        mask_class_dir = os.path.join(mask_dir, str(orig_class_id))

        mask_files = [f for f in os.listdir(mask_class_dir) if image_stem in f]
        mask_dict = {}
        for lbl in self.part_labels:
            for fname in mask_files:
                if lbl in fname:
                    mask_dict[lbl] = os.path.join(mask_class_dir, fname)

        mask = np.zeros((self.image_size, self.image_size), dtype=np.int64)

        for i, part_name in enumerate(self.part_labels):
            if part_name in mask_dict:
                pm = Image.open(mask_dict[part_name]).convert("L")
                pm = pm.resize((self.image_size, self.image_size), Image.NEAREST)
                pm_np = np.array(pm) > 127
                mask[pm_np] = i

        mask = np.clip(mask, 0, len(self.part_labels) - 1)

        return (
            image,
            torch.from_numpy(mask).long(),
            torch.tensor(species_label, dtype=torch.long)
        )


# ==========================================
# CREATE DATASET & SPLIT (same as original cub_manet.py)
# ==========================================
torch.manual_seed(SEED)
np.random.seed(SEED)

full_dataset = PartWholeMultiTaskDataset(
    image_size=IMAGE_SIZE,
    part_labels=labels
)

n_total = len(full_dataset)
n_train = int(0.75 * n_total)
n_val   = int(0.15 * n_total)
n_test  = n_total - n_train - n_val

train_set, val_set, test_set = random_split(
    full_dataset,
    [n_train, n_val, n_test]
)

print(f"Train: {len(train_set):4d}  Val: {len(val_set):4d}  Test: {len(test_set):4d}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# ==========================================
# MULTI-TASK MODEL (unchanged)
# ==========================================
class MultiTaskPartModel(L.LightningModule):
    def __init__(self, arch, encoder_name, num_parts, num_species):
        super().__init__()
        self.save_hyperparameters()

        aux_params = dict(
            pooling='avg',
            dropout=0.3,
            classes=num_species,
        )

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=3,
            classes=num_parts,
            aux_params=aux_params
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(params["std"]).view(1, 3, 1, 1))

        self.seg_loss_fn = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True)
        self.cls_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)

    def _shared_step(self, batch, stage):
        img, mask, species_label = batch
        mask_logits, species_logits = self(img)

        loss_seg = self.seg_loss_fn(mask_logits, mask)
        loss_cls = self.cls_loss_fn(species_logits, species_label)
        total_loss = loss_seg + loss_cls

        # Metrics
        pred_mask = mask_logits.argmax(dim=1)
        acc = (species_logits.argmax(dim=1) == species_label).float().mean()

        ious = []
        for c in range(self.hparams.num_parts):
            p = (pred_mask == c)
            t = (mask == c)
            inter = (p & t).sum().float()
            union = (p | t).sum().float()
            iou_c = inter / union if union > 0 else 1.0
            ious.append(iou_c)
        iou = sum(ious) / len(ious)

        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc",  acc,       on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_iou",  iou,       on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_iou"}


# ==========================================
# RUN TRAINING
# ==========================================
if __name__ == "__main__":
    num_species = full_dataset.num_species
    print(f"Number of species classes: {num_species}")

    model = MultiTaskPartModel(
        arch=MODEL_NAME,
        encoder_name=ENCODER_NAME,
        num_parts=num_parts,
        num_species=num_species
    )

    logger = CSVLogger(save_dir="logs/", name="cub_manet_multitask_randomsplit")

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=DEVICE_IDS,
        precision="16-mixed",
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                monitor="val_iou",
                mode="max",
                save_top_k=1,
                filename="best-{epoch:03d}-{val_iou:.4f}",
                auto_insert_metric_name=False
            ),
            LearningRateMonitor(logging_interval="step")
        ],
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")