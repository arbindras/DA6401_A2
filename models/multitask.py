import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown

from .vgg11 import VGG11Encoder, ConvBlock


class MultiTaskPerceptionModel(nn.Module):
    def __init__(
        self,
        num_breeds=37,
        seg_classes=3,
        in_channels=3,
        classifier_path="classifier.pth",
        localizer_path="localizer.pth",
        unet_path="unet.pth",
    ):
        super().__init__()

        # -------------------------
        # 📥 Download weights
        # -------------------------
        gdown.download(id="1s7Rju2nm6c4lnq9YFpyWACuby00befjz", output=classifier_path, quiet=False)
        gdown.download(id="1BTmgw6VFCx23JqGXGwYAA8540RwWUVYK", output=localizer_path, quiet=False)
        gdown.download(id="1hR5M9aKldMGJ0pnv77WavT64gbIkGNmh", output=unet_path, quiet=False)

        # -------------------------
        # 🧠 Encoder
        # -------------------------
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # -------------------------
        # 🏷 Classification head
        # -------------------------
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_breeds),
        )

        # -------------------------
        # 📦 Localization head
        # -------------------------
        self.localizer_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4),
        )

        # -------------------------
        # 🧩 UNet Decoder (EXACT match)
        # -------------------------
        self.up4 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = ConvBlock(128, 64)

        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

        # -------------------------
        # 🔥 Load weights
        # -------------------------
        self._load_weights(classifier_path, localizer_path, unet_path)

    # ============================================================
    # 🔥 WEIGHT LOADING (FIXED)
    # ============================================================
    def _load_weights(self, cls_path, loc_path, unet_path):

        def load(path):
            try:
                sd = torch.load(path, map_location="cpu")
                if isinstance(sd, dict) and "model_state" in sd:
                    return sd["model_state"]
                return sd
            except:
                return None

        cls_sd = load(cls_path)
        loc_sd = load(loc_path)
        unet_sd = load(unet_path)

        # -------------------------
        # Encoder + classifier
        # -------------------------
        if cls_sd:
            self.encoder.load_state_dict(
                {k.replace("encoder.", ""): v for k, v in cls_sd.items() if "encoder." in k},
                strict=False,
            )

            self.classifier_head.load_state_dict(
                {k.replace("classifier_head.", ""): v for k, v in cls_sd.items() if "classifier_head." in k},
                strict=False,
            )

            print("✅ Loaded classifier weights")

        # -------------------------
        # Localizer
        # -------------------------
        if loc_sd:
            try:
                self.localizer_head.load_state_dict(loc_sd, strict=False)
                print("✅ Loaded localizer weights")
            except:
                print("⚠️ Localizer partial load")

        # -------------------------
        # UNet (FULL LOAD)
        # -------------------------
        if unet_sd:
            try:
                self.load_state_dict(unet_sd, strict=False)
                print("✅ Loaded UNet weights")
            except Exception as e:
                print("⚠️ UNet load failed:", e)

    # ============================================================
    # 🚀 FORWARD
    # ============================================================
    def forward(self, x):
        B, _, H, W = x.shape

        # Encoder with features
        bottleneck, feats = self.encoder(x, return_features=True)

        # 🔥 FIX: Undo pool5 effect (VERY IMPORTANT)
        bottleneck = F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False)

        # -------------------------
        # Classification
        # -------------------------
        pooled = F.adaptive_avg_pool2d(bottleneck, (7, 7))
        cls_out = self.classifier_head(pooled)

        # -------------------------
        # Localization
        # -------------------------
        loc_raw = torch.sigmoid(self.localizer_head(pooled))

        cx = loc_raw[:, 0] * W
        cy = loc_raw[:, 1] * H
        w  = loc_raw[:, 2] * W
        h  = loc_raw[:, 3] * H

        loc_out = torch.stack([cx, cy, w, h], dim=1)

        # -------------------------
        # Segmentation (REAL SKIPS)
        # -------------------------
        x4 = feats["enc4_2"]
        x3 = feats["enc3_2"]
        x2 = feats["enc2"]
        x1 = feats["enc1"]

        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        seg_out = self.seg_head(d1)
        seg_out = F.interpolate(seg_out, size=(H, W), mode="bilinear", align_corners=False)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }