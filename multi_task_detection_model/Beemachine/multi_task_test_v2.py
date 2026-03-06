import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 4, 5, 7" 
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import lightning as L 
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import segmentation_models_pytorch as smp

# ==========================================
# 1. DATASET
# ==========================================
class SpeciesPartDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, image_size, num_classes):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.num_classes = num_classes 

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4925, 0.4475, 0.3490), 
                                 std=(0.2392, 0.2265, 0.2213))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "images"]
        species_label = self.df.loc[idx, "label"]
        
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base + "_m.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, base + ".png")
            
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)
        
        # Safety: Clip mask values to stay within range [0, num_classes-1]
        mask_np[mask_np >= self.num_classes] = 0
        
        return img, torch.from_numpy(mask_np), torch.tensor(species_label, dtype=torch.long)

# ==========================================
# 2. MULTI-TASK LIGHTNING MODULE
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

        self.seg_loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.cls_loss_fn = nn.CrossEntropyLoss()
        
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def _compute_metrics(self, mask_logits, mask, species_logits, species_label, stage):
        # 1. Classification Accuracy
        acc = (species_logits.argmax(1) == species_label).float().mean()
        
        # 2. Segmentation IoU (Macro)
        preds = mask_logits.argmax(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds.long(), mask.long(), mode="multiclass", num_classes=self.hparams.num_parts
        )
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        
        # Logging per step and per epoch
        self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_iou", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return tp, fp, fn, tn

    def training_step(self, batch, batch_idx):
        img, mask, species_label = batch
        mask_logits, species_logits = self(img)
        
        loss_seg = self.seg_loss_fn(mask_logits, mask)
        loss_cls = self.cls_loss_fn(species_logits, species_label)
        total_loss = loss_seg + loss_cls
        
        self._compute_metrics(mask_logits, mask, species_logits, species_label, "train")
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        img, mask, species_label = batch
        mask_logits, species_logits = self(img)
        
        loss_seg = self.seg_loss_fn(mask_logits, mask)
        loss_cls = self.cls_loss_fn(species_logits, species_label)
        
        tp, fp, fn, tn = self._compute_metrics(mask_logits, mask, species_logits, species_label, "val")
        
        # Still store for epoch-end aggregation if needed
        self.validation_step_outputs.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
        return loss_seg + loss_cls

    def on_validation_epoch_end(self):
        # Optional: You can do a more robust dataset-wide IoU here
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        img, mask, species_label = batch
        mask_logits, species_logits = self(img)
        
        # Reuse the metric helper we built earlier
        self._compute_metrics(mask_logits, mask, species_logits, species_label, "test")

    def on_test_epoch_end(self):
        # This is called automatically at the end of the test loop
        print("\n" + "="*30)
        print("TESTING COMPLETE")
        # Lightning will automatically print the logged values in a table format

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Settings
    EPOCHS = 100
    BATCH_SIZE = 128
    IMAGE_SIZE = 320
    DEVICE_IDS = [1, 2, 4, 5, 7]
    DATA_DIR = r"/home/c/choton/beemachine/datasets/Beemachine_Partwhole_v5/"
    
    # Load DFs
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_aug_labels.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_labels.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_labels.csv'))
    
    # --- IMPORTANT: Ensure 'num_classes' (Species) is defined from your LabelEncoder ---
    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df["species"])
    val_df["label"] = le.transform(val_df["species"])
    test_df["label"] = le.transform(test_df["species"])
    num_classes = len(le.classes_)
    print(f"Total images, Train: {len(train_df['label'])}, Validation: {len(val_df['label'])}, Test: {len(test_df['label'])}")
    print(f"Total classes: {num_classes}")

    # Datasets
    train_dataset = SpeciesPartDataset(df=train_df, 
        img_dir=os.path.join(DATA_DIR, "train", "aug_images"),
        mask_dir=os.path.join(DATA_DIR, "train", "aug_masks"),
        image_size=IMAGE_SIZE, num_classes=4)
    
    val_dataset = SpeciesPartDataset(df=val_df, 
        img_dir=os.path.join(DATA_DIR, "valid", "images"),
        mask_dir=os.path.join(DATA_DIR, "valid", "masks"),
        image_size=IMAGE_SIZE, num_classes=4)
    
    test_dataset = SpeciesPartDataset(df=test_df, 
        img_dir=os.path.join(DATA_DIR, "test", "images"),
        mask_dir=os.path.join(DATA_DIR, "test", "masks"),
        image_size=IMAGE_SIZE, num_classes=4)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    print(f"Images in dataset, train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    # Model
    model = MultiTaskPartModel(
        arch="deeplabv3plus",
        encoder_name="resnext50_32x4d",
        num_parts=4,
        num_species=num_classes
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=DEVICE_IDS,
        precision="16-mixed",
        callbacks=[
            ModelCheckpoint(monitor="val_iou", mode="max", save_top_k=1),
            LearningRateMonitor(logging_interval="step")
        ]
    )

    trainer.fit(model, train_loader, val_loader)

    results = trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    # 3. Accessing the numbers directly if you need to print them in a specific format
    test_acc = results[0]['test_acc_epoch']
    test_iou = results[0]['test_iou_epoch']

    print(f"\nFinal Test Results:")
    print(f"--> Classification Accuracy: {test_acc*100:.2f}%")
    print(f"--> Part Segmentation IoU:   {test_iou:.4f}")