# """Unified multi-task model
# """

# from vgg11 import ConvBlock, VGG11Encoder
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiTaskPerceptionModel(nn.Module):
#     """Shared-backbone multi-task model."""

#     def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
#         """
#         Initialize the shared backbone/heads using these trained weights.
#         Args:
#             num_breeds: Number of output classes for classification head.
#             seg_classes: Number of output classes for segmentation head.
#             in_channels: Number of input channels.
#             classifier_path: Path to trained classifier weights.
#             localizer_path: Path to trained localizer weights.
#             unet_path: Path to trained unet weights.
#         """
#         super(MultiTaskPerceptionModel, self).__init__()
#         self.encoder = VGG11Encoder(in_channels=in_channels)

#         self.classifier_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
#             nn.Flatten(),
#             nn.Linear(512*1*1, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_breeds)
#         )
#         self.localizer_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
#             nn.Flatten(),
#             nn.Linear(512*1*1, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024,4)
#         )

#         self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
#         self.dec4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

#         self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.dec3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

#         self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

#         self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

#         self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

#             # Load pretrained weights for encoder and heads
#         self._load_weights(classifier_path, localizer_path, unet_path)
#     def _load_weights(self, classifier_path, localizer_path, unet_path):
#         try:
#             classifier_state = torch.load(classifier_path, map_location='cpu')
#             localizer_state = torch.load(localizer_path, map_location='cpu')
#             unet_state = torch.load(unet_path, map_location='cpu')

#             # Load encoder weights (assuming same architecture)
#             self.encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in classifier_state.items() if k.startswith('encoder.')}, strict=False   )
#             # Load classifier head weights
            
#             self.classifier_head.load_state_dict({k.replace('classifier_head.', ''): v for k, v in classifier_state.items() if k.startswith('classifier_head.')}, strict=False)

#             # Load localizer head weights
#             self.localizer_head.load_state_dict({k.replace('localizer_head.', ''): v for k, v in localizer_state.items() if k.startswith('localizer_head.')}, strict=False)
#         except Exception as e:
#             print(f"⚠️ Warning: Could not fully load pretrained weights: {e}")
           

#     def forward(self, x: torch.Tensor):
#         """Forward pass for multi-task model.
#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].
#         Returns:
#             A dict with keys:
#             - 'classification': [B, num_breeds] logits tensor.
#             - 'localization': [B, 4] bounding box tensor.
#             - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
#         """
#         B, _, H, W = x.shape

#         bottleneck = None
#         feats = None

#         # If encoder supports returning features via keyword
#         try:
#             out = self.encoder(x, return_features=True)
#             # accept either (bottleneck, feats) or dict
#             if isinstance(out, tuple) and len(out) == 2:
#                 bottleneck, feats = out
#             elif isinstance(out, dict):
#                 # some encoders might return dict of features only
#                 feats = out
#                 # attempt to get bottleneck from a known key
#                 bottleneck = feats.get("bottleneck") or feats.get("enc5_2") or feats.get("out")
#             else:
#                 # single tensor returned
#                 bottleneck = out
#         except TypeError:


#         bottleneck, feats = self.encoder(x, return_features=True)
#         pooled = F.adaptive_avg_pool2d(bottleneck, (7, 7))

#         classfication_out = self.classifier_head(pooled)
#         localizer_out = torch.sigmoid(self.localizer_head(pooled))

#         x_center = localizer_out[:, 0] * W
#         y_center = localizer_out[:, 1] * H
#         width = localizer_out[:, 2] * W
#         height = localizer_out[:, 3] * H
#         localizer_out = torch.stack([x_center, y_center, width, height], dim=1)

#         x4 = feats['enc4_2']
#         x3 = feats['enc3_2']
#         x2 = feats['enc2']
#         x1 = feats['enc1']

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
#         d1 = F.relu(self.dec1(d1))  

#         seg_out = self.seg_head(d1)
#         seg_out = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=False)

#         return {
#             'classification': classfication_out,
#             'localization': localizer_out,
#             'segmentation': seg_out
#         }

# x = torch.randn(2, 3, 224, 224)
# model = MultiTaskPerceptionModel()
# out = model(x)

# print(out["classification"].shape)  # [2, 37]
# print(out["localization"].shape)    # [2, 4]
# print(out["segmentation"].shape)    # [2, 3, H, W]

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg11 import VGG11Encoder  # must exist in your project
import gdown  # for downloading pretrained weights

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model with robust encoder handling and defensive weight loading."""

    def __init__(self,
                 num_breeds: int = 37,
                 seg_classes: int = 3,
                 in_channels: int = 3,
                 classifier_path: str = "classifier.pth",
                 localizer_path: str = "localizer.pth",
                 unet_path: str = "unet.pth"):
        gdown.download(id="1fpIHkdzmWmdMsOaFgrpgMuO_E7Pir_Il",output=classifier_path, quiet=False)
        gdown.download(id="1McBzlxWtX_RndFIgnCZxs6FDWkP0ZXOz", output=localizer_path, quiet=False)
        gdown.download(id="1vX0Rb6fPFkDbUmDSc-6rE7nS5JnYqEBq", output=unet_path, quiet=False)
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head (uses an internal adaptive pool as well)
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_breeds)
        )

        # Localizer head
        self.localizer_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4)
        )

        # Simple decoder for segmentation (mirrors your earlier decoder)
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

        # Try to load pretrained weights (defensive)
        self._load_weights(classifier_path, localizer_path, unet_path)

    def _safe_filter_state(self, state_dict, prefix):
        """Return a new dict with keys that start with prefix, with prefix stripped."""
        out = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        return out

    def _load_weights(self, classifier_path, localizer_path, unet_path):
        # Defensive: only attempt to load if files exist and are dict-like
        def try_load(path):
            try:
                return torch.load(path, map_location="cpu")
            except Exception:
                return None

        cls_sd = try_load(classifier_path)
        loc_sd = try_load(localizer_path)
        unet_sd = try_load(unet_path)

        # If the saved files are full checkpoints (dict with "model_state"), extract that
        def extract_model_state(maybe_ckpt):
            if maybe_ckpt is None:
                return None
            if isinstance(maybe_ckpt, dict) and "model_state" in maybe_ckpt:
                return maybe_ckpt["model_state"]
            # assume it's already a state_dict
            return maybe_ckpt

        cls_sd = extract_model_state(cls_sd)
        loc_sd = extract_model_state(loc_sd)
        unet_sd = extract_model_state(unet_sd)

        # Load encoder weights from classifier checkpoint if available
        if cls_sd is not None:
            # try common prefixes
            candidates = []
            # direct keys that look like encoder.* or features.* etc.
            if any(k.startswith("encoder.") for k in cls_sd.keys()):
                enc_map = self._safe_filter_state(cls_sd, "encoder.")
                try:
                    self.encoder.load_state_dict(enc_map, strict=False)
                    print("Loaded encoder weights from classifier checkpoint (prefix 'encoder.').")
                except Exception as e:
                    print("Warning loading encoder from classifier checkpoint:", e)
            elif any(k.startswith("features.") for k in cls_sd.keys()):
                # some saved VGGs use 'features.' prefix
                enc_map = self._safe_filter_state(cls_sd, "features.")
                try:
                    self.encoder.load_state_dict(enc_map, strict=False)
                    print("Loaded encoder weights from classifier checkpoint (prefix 'features.').")
                except Exception as e:
                    print("Warning loading encoder from classifier checkpoint (features.):", e)
            else:
                # fallback: try to load any matching keys (best-effort)
                try:
                    self.encoder.load_state_dict({k: v for k, v in cls_sd.items() if k in self.encoder.state_dict()}, strict=False)
                    print("Partially loaded encoder weights from classifier checkpoint (best-effort).")
                except Exception as e:
                    print("Warning: could not load encoder weights from classifier checkpoint:", e)

            # classifier head
            try:
                # keys may be 'classifier_head.0.weight' etc.
                ch_map = self._safe_filter_state(cls_sd, "classifier_head.")
                if ch_map:
                    self.classifier_head.load_state_dict(ch_map, strict=False)
                    print("Loaded classifier head weights from classifier checkpoint.")
                else:
                    # try matching by name
                    self.classifier_head.load_state_dict({k: v for k, v in cls_sd.items() if k in self.classifier_head.state_dict()}, strict=False)
                    print("Partially loaded classifier head weights (best-effort).")
            except Exception as e:
                print("Warning loading classifier head:", e)

        # localizer head
        if loc_sd is not None:
            try:
                lh_map = self._safe_filter_state(loc_sd, "localizer_head.")
                if lh_map:
                    self.localizer_head.load_state_dict(lh_map, strict=False)
                    print("Loaded localizer head weights from localizer checkpoint.")
                else:
                    self.localizer_head.load_state_dict({k: v for k, v in loc_sd.items() if k in self.localizer_head.state_dict()}, strict=False)
                    print("Partially loaded localizer head weights (best-effort).")
            except Exception as e:
                print("Warning loading localizer head:", e)

        # segmentation / decoder weights (from unet checkpoint)
        if unet_sd is not None:
            try:
                # try to load decoder layers by matching names
                dec_map = {k: v for k, v in unet_sd.items() if any(k.startswith(prefix) for prefix in ("dec1.", "dec2.", "dec3.", "dec4.", "seg_head.", "up"))}
                if dec_map:
                    # map keys to current names if necessary (best-effort)
                    self_state = self.state_dict()
                    filtered = {k: v for k, v in dec_map.items() if k in self_state}
                    if filtered:
                        self.load_state_dict(filtered, strict=False)
                        print("Partially loaded decoder/segmentation weights from unet checkpoint.")
            except Exception as e:
                print("Warning loading unet weights:", e)

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape

        # Try to get bottleneck + intermediate features from encoder if supported
        bottleneck = None
        feats = None

        # If encoder supports returning features via keyword
        try:
            out = self.encoder(x, return_features=True)
            # accept either (bottleneck, feats) or dict
            if isinstance(out, tuple) and len(out) == 2:
                bottleneck, feats = out
            elif isinstance(out, dict):
                # some encoders might return dict of features only
                feats = out
                # attempt to get bottleneck from a known key
                bottleneck = feats.get("bottleneck") or feats.get("enc5_2") or feats.get("out")
            else:
                # single tensor returned
                bottleneck = out
        except TypeError:
            # encoder doesn't accept return_features; call it normally
            out = self.encoder(x)
            if isinstance(out, tuple) and len(out) == 2:
                bottleneck, feats = out
            elif isinstance(out, dict):
                feats = out
                bottleneck = feats.get("bottleneck") or feats.get("enc5_2") or next(iter(feats.values()))
            else:
                bottleneck = out

        # If we still don't have feats, try to extract them by re-running encoder blocks
        # (This requires VGG11Encoder to expose submodules; if not available, skip)
        if feats is None:
            # best-effort: try to call a method that returns features
            if hasattr(self.encoder, "get_features"):
                try:
                    feats = self.encoder.get_features(x)
                except Exception:
                    feats = {}
            else:
                feats = {}

        # Classification and localization use pooled bottleneck
        pooled = F.adaptive_avg_pool2d(bottleneck, (7, 7))
        classification_out = self.classifier_head(pooled)

        localizer_out = torch.sigmoid(self.localizer_head(pooled))
        x_center = localizer_out[:, 0] * W
        y_center = localizer_out[:, 1] * H
        width = localizer_out[:, 2] * W
        height = localizer_out[:, 3] * H
        localizer_out = torch.stack([x_center, y_center, width, height], dim=1)

        # For segmentation decoder we expect certain feature keys; fallback to empty tensors if missing
        x4 = feats.get("enc4_2_pooled") if isinstance(feats, dict) else None
        x3 = feats.get("enc3_2_pooled") if isinstance(feats, dict) else None
        x2 = feats.get("enc2_pooled") if isinstance(feats, dict) else None
        x1 = feats.get("enc1_pooled") if isinstance(feats, dict) else None

        # If any feature is missing, try to derive shapes from bottleneck by simple upsampling
        if x4 is None:
            x4 = F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False)
        if x3 is None:
            x3 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        if x2 is None:
            x2 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        if x1 is None:
            x1 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)

        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = F.relu(self.dec4(d4))

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = F.relu(self.dec3(d3))

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = F.relu(self.dec2(d2))

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = F.relu(self.dec1(d1))

        seg_out = self.seg_head(d1)
        seg_out = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=False)

        return {
            "classification": classification_out,
            "localization": localizer_out,
            "segmentation": seg_out
        }

# Quick smoke test (will work only if VGG11Encoder exists and matches API)
# if __name__ == "__main__":
#     x = torch.randn(2, 3, 224, 224)
#     model = MultiTaskPerceptionModel()
#     out = model(x)
#     print(out["classification"].shape)  # [2, num_breeds]
#     print(out["localization"].shape)    # [2, 4]
#     print(out["segmentation"].shape)    # [2, seg_classes, H, W]