import torch
# import torch.nn as nn
import torch.nn.functional as F
# import gdown

# from .vgg11 import VGG11Encoder, ConvBlock


# class MultiTaskPerceptionModel(nn.Module):
#     def __init__(
#         self,
#         num_breeds=37,
#         seg_classes=3,
#         in_channels=3,
#         classifier_path="classifier.pth",
#         localizer_path="localizer.pth",
#         unet_path="unet.pth",
#     ):
#         super().__init__()

#         # -------------------------
#         # 📥 Download weights
#         # -------------------------
#         gdown.download(id="1s7Rju2nm6c4lnq9YFpyWACuby00befjz", output=classifier_path, quiet=False)
#         gdown.download(id="1BTmgw6VFCx23JqGXGwYAA8540RwWUVYK", output=localizer_path, quiet=False)
#         gdown.download(id="1hR5M9aKldMGJ0pnv77WavT64gbIkGNmh", output=unet_path, quiet=False)

#         # -------------------------
#         # 🧠 Encoder
#         # -------------------------
#         self.encoder = VGG11Encoder(in_channels=in_channels)

#         # -------------------------
#         # 🏷 Classification head
#         # -------------------------
#         self.classifier_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(512, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_breeds),
#         )

#         # -------------------------
#         # 📦 Localization head
#         # -------------------------
#         self.localizer_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(512, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 4),
#         )

#         # -------------------------
#         # 🧩 UNet Decoder (EXACT match)
#         # -------------------------
#         self.up4 = nn.ConvTranspose2d(512, 512, 2, 2)
#         self.dec4 = ConvBlock(1024, 512)

#         self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
#         self.dec3 = ConvBlock(512, 256)

#         self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
#         self.dec2 = ConvBlock(256, 128)

#         self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
#         self.dec1 = ConvBlock(128, 64)

#         self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

#         # -------------------------
#         # 🔥 Load weights
#         # -------------------------
#         self._load_weights(classifier_path, localizer_path, unet_path)

#     # ============================================================
#     # 🔥 WEIGHT LOADING (FIXED)
#     # ============================================================
#     def _load_weights(self, cls_path, loc_path, unet_path):

#         def load(path):
#             try:
#                 sd = torch.load(path, map_location="cpu")
#                 if isinstance(sd, dict) and "model_state" in sd:
#                     return sd["model_state"]
#                 return sd
#             except:
#                 return None

#         cls_sd = load(cls_path)
#         loc_sd = load(loc_path)
#         unet_sd = load(unet_path)

#         # -------------------------
#         # Encoder + classifier
#         # -------------------------
#         if cls_sd:
#             self.encoder.load_state_dict(
#                 {k.replace("encoder.", ""): v for k, v in cls_sd.items() if "encoder." in k},
#                 strict=False,
#             )

#             self.classifier_head.load_state_dict(
#                 {k.replace("classifier_head.", ""): v for k, v in cls_sd.items() if "classifier_head." in k},
#                 strict=False,
#             )

#             print("✅ Loaded classifier weights")

#         # -------------------------
#         # Localizer
#         # -------------------------
#         if loc_sd:
#             try:
#                 self.localizer_head.load_state_dict(loc_sd, strict=False)
#                 print("✅ Loaded localizer weights")
#             except:
#                 print("⚠️ Localizer partial load")

#         # -------------------------
#         # UNet (FULL LOAD)
#         # -------------------------
#         if unet_sd:
#             try:
#                 self.load_state_dict(unet_sd, strict=False)
#                 print("✅ Loaded UNet weights")
#             except Exception as e:
#                 print("⚠️ UNet load failed:", e)

#     # ============================================================
#     # 🚀 FORWARD
#     # ============================================================
#     def forward(self, x):
#         B, _, H, W = x.shape

#         # Encoder with features
#         bottleneck, feats = self.encoder(x, return_features=True)

#         # 🔥 FIX: Undo pool5 effect (VERY IMPORTANT)
#         bottleneck = F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False)

#         # -------------------------
#         # Classification
#         # -------------------------
#         pooled = F.adaptive_avg_pool2d(bottleneck, (7, 7))
#         cls_out = self.classifier_head(pooled)

#         # -------------------------
#         # Localization
#         # -------------------------
#         loc_raw = torch.sigmoid(self.localizer_head(pooled))

#         cx = loc_raw[:, 0] * W
#         cy = loc_raw[:, 1] * H
#         w  = loc_raw[:, 2] * W
#         h  = loc_raw[:, 3] * H

#         loc_out = torch.stack([cx, cy, w, h], dim=1)

#         # -------------------------
#         # Segmentation (REAL SKIPS)
#         # -------------------------
#         x4 = feats["enc4_2"]
#         x3 = feats["enc3_2"]
#         x2 = feats["enc2"]
#         x1 = feats["enc1"]

#         d4 = self.up4(bottleneck)
#         d4 = torch.cat([d4, x4], dim=1)
#         d4 = self.dec4(d4)

#         d3 = self.up3(d4)
#         d3 = torch.cat([d3, x3], dim=1)
#         d3 = self.dec3(d3)

#         d2 = self.up2(d3)
#         d2 = torch.cat([d2, x2], dim=1)
#         d2 = self.dec2(d2)

#         d1 = self.up1(d2)
#         d1 = torch.cat([d1, x1], dim=1)
#         d1 = self.dec1(d1)

#         seg_out = self.seg_head(d1)
#         seg_out = F.interpolate(seg_out, size=(H, W), mode="bilinear", align_corners=False)

#         return {
#             "classification": cls_out,
#             "localization": loc_out,
#             "segmentation": seg_out,
#         }

from models.layers import CustomDropout

import torch
import torch.nn as nn
import gdown

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.vgg11 import VGG11Encoder, ConvBlock

# class MultiTaskPerceptionModel(nn.Module):
#     """
#     Multi-task perception model that composes VGG11Classifier, VGG11Localizer,
#     and VGG11UNet and loads each from its own checkpoint.

#     Using the actual trained model classes guarantees the architecture matches
#     the saved state dicts exactly, so weight loading is correct.
#     """

#     def __init__(
#         self,
#         num_breeds: int = 37,
#         seg_classes: int = 3,
#         in_channels: int = 3,
#         classifier_path: str = "classifier.pth",
#         localizer_path: str = "localizer.pth",
#         unet_path: str = "unet.pth",
#     ):
#         super().__init__()

#         # -------------------------
#         # 📥 Download weights
#         # -------------------------
#         gdown.download(id="1JCg2CNXKXc63YLHP3OsFoUg9I_bwXsqI", output=classifier_path, quiet=False)
#         gdown.download(id="1E_8FcKhZSu_HlQ-wB28iE1U3J7aVLhMw", output=localizer_path, quiet=False)
#         gdown.download(id="1hR5M9aKldMGJ0pnv77WavT64gbIkGNmh", output=unet_path, quiet=False)

#         # -------------------------
#         # 🧠 Individual models
#         # Each has the EXACT architecture that was used during training,
#         # so load_state_dict will match layer names and shapes.
#         # -------------------------
#         self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
#         self.localizer  = VGG11Localizer(in_channels=in_channels)
#         self.segmenter  = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

#         # -------------------------
#         # 🔥 Load weights
#         # -------------------------
#         self._load(self.classifier, classifier_path, "classifier")
#         self._load(self.localizer,  localizer_path,  "localizer")
#         self._load(self.segmenter,  unet_path,       "UNet")

#     # ============================================================
#     # 🔥 Weight loading
#     # ============================================================
#     def _load(self, model: nn.Module, path: str, name: str) -> None:
#         """Load a checkpoint into *model*, supporting both raw and wrapped state dicts."""
#         try:
#             sd = torch.load(path, map_location="cpu")
#             # train.py wraps the state dict under "model_state"
#             if isinstance(sd, dict) and "model_state" in sd:
#                 sd = sd["model_state"]
#             model.load_state_dict(sd, strict=True)
#             print(f"✅ Loaded {name} weights")
#         except Exception as e:
#             print(f"⚠️ {name} strict load failed ({e}), trying strict=False")
#             try:
#                 sd = torch.load(path, map_location="cpu")
#                 if isinstance(sd, dict) and "model_state" in sd:
#                     sd = sd["model_state"]
#                 model.load_state_dict(sd, strict=False)
#                 print(f"⚠️ Loaded {name} weights (partial)")
#             except Exception as e2:
#                 print(f"❌ {name} load completely failed: {e2}")

#     # ============================================================
#     # 🚀 Forward
#     # ============================================================
#     def forward(self, x: torch.Tensor) -> dict:
#         """
#         Args:
#             x: [B, C, H, W] input images.
#         Returns:
#             dict with keys:
#               "classification" → [B, num_breeds] logits
#               "localization"   → [B, 4] normalized (cx, cy, w, h) in [0, 1]
#               "segmentation"   → [B, seg_classes, H, W] logits
#         """
#         # cls_out = self.classifier(x)
#         # loc_out = self.localizer(x)
#         # seg_out = self.segmenter(x)
#         cls_out = self.classifier(x)

#         # convert normalized → pixel coordinates
#         B, _, H, W = x.shape
#         loc = self.localizer(x)  # normalized [0,1]

#         cx = loc[:, 0] * W
#         cy = loc[:, 1] * H
#         w  = loc[:, 2] * W
#         h  = loc[:, 3] * H

#         loc_out = torch.stack([cx, cy, w, h], dim=1)

#         seg_out = self.segmenter(x)

#         return {
#             "classification": cls_out,
#             "localization":   loc_out,
#             "segmentation":   seg_out,
#         }

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

        import gdown

        # Download weights
        gdown.download(id="1R_UMGJ82-myRodJZdV4FJSJy3JmJowpF", output=classifier_path, quiet=False)
        gdown.download(id="18EuKWuB4EA9V_FCbvhvBOQVRs_XWcPju", output=localizer_path, quiet=False)
        gdown.download(id="14t-6HQkNiTdWyj8MD3D7knTPazRCiNfo", output=unet_path, quiet=False)

        # ✅ Use pretrained architectures
        from models.classification import VGG11Classifier
        from models.localization import VGG11Localizer
        from models.segmentation import VGG11UNet

        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer  = VGG11Localizer(in_channels=in_channels)
        self.segmenter  = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # ✅ Load weights properly
        self._load(self.classifier, classifier_path, "classifier")
        self._load(self.localizer,  localizer_path,  "localizer")
        self._load(self.segmenter,  unet_path,       "unet")

        # ✅ Set to evaluation mode (CRITICAL)
        self.classifier.eval()
        self.localizer.eval()
        self.segmenter.eval()

    def _load(self, model, path, name):
        sd = torch.load(path, map_location="cpu")

        if isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]

        try:
            model.load_state_dict(sd, strict=True)
            print(f"✅ Loaded {name} weights")
        except:
            model.load_state_dict(sd, strict=False)
            print(f"⚠️ Partial load for {name}")

    def forward(self, x):
        B, _, H, W = x.shape

        # Classification
        cls_out = self.classifier(x)

        # Localization → convert to pixel coords
        # loc = self.localizer(x)
        # loc_out = torch.stack([
        #     loc[:, 0] * W,
        #     loc[:, 1] * H,
        #     loc[:, 2] * W,
        #     loc[:, 3] * H,
        # ], dim=1)
        loc_out = self.localizer(x)

        # Segmentation
        seg_out = self.segmenter(x)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }