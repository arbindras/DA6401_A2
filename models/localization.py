# """Localization modules
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .layers import CustomDropout
# from .vgg11 import VGG11Encoder

# class VGG11Localizer(nn.Module):
#     """VGG11-based localizer."""

#     def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
#         """
#         Initialize the VGG11Localizer model.

#         Args:
#             in_channels: Number of input channels.
#             dropout_p: Dropout probability for the localization head.
#         """
        
#         super(VGG11Localizer, self).__init__()

#         self.encoder = VGG11Encoder(in_channels=in_channels)
#         for param in self.encoder.parameters():
#             param.requires_grad = False

#         self.localization_head = nn.Sequential(
#            # nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
#             nn.Flatten(),

#             nn.Linear(512*7*7, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             CustomDropout(dropout_p),

#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             CustomDropout(dropout_p),

#             nn.Linear(512, 4)  # Final localization layer for bounding box coordinates
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass for localization model.
#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].

#         Returns:
#             Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
#         """
#         B, _, H, W = x.shape
#         x = self.encoder(x) # [B, 512, H/16, W/16]
#         x = F.adaptive_avg_pool2d(x, (7, 7))  # Ensure the spatial dimensions are 7x7
#         out = self.localization_head(x) # [B, 4]

#         # Convert from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
#         out = torch.sigmoid(out)  # Ensure outputs are in [0, 1]
#         x_center = out[:, 0] * W
#         y_center = out[:, 1] * H
#         width = out[:, 2] * W
#         height = out[:, 3] * H
        
#         #bbox = torch.stack([x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2], dim=1)
#         bbox = torch.stack([x_center, y_center, width, height], dim=1)

#         return bbox

# """Localization modules
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.layers import CustomDropout
# from models.vgg11 import VGG11Encoder

# class VGG11Localizer(nn.Module):
#     """VGG11-based localizer."""

#     def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, pretrained: bool = True):
#         """
#         Initialize the VGG11Localizer model.

#         Args:
#             in_channels: Number of input channels.
#             dropout_p: Dropout probability for the localization head.
#         """
        
#         super().__init__()

#         self.encoder = VGG11Encoder(in_channels=in_channels)
#         # for param in self.encoder.parameters():
#         #     param.requires_grad = False

#         if pretrained and in_channels == 3:
#             vgg = tv_models.vgg11(weight=tv_models.VGG11_Weights.IMAGENET1K_V1)
#             try:
#                 self.encoder.features.load_state_dict(vgg.features.state_dict())
#                 print("Loaded pretrained VGG11 weights into encoder.")
#             except Exception as e:
#                 print(f"Error loading pretrained weights: {e}")

#         self.localization_head = nn.Sequential(
#             nn.Flatten(),

#             nn.Linear(512*7*7, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             CustomDropout(dropout_p),

#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             CustomDropout(dropout_p),

#             nn.Linear(512, 4)  # Final localization layer for bounding box coordinates
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass for localization model.
#         Args:
#             x: Input tensor of shape [B, in_channels, H, W].

#         Returns:
#             Bounding box [B, 4] in normalized (cx, cy, w, h) format, values in [0, 1].
#             Matches the ground-truth format produced by OxfordIIITPetDataset._mask_to_bbox.
#         """
#         x = self.encoder(x)                          # [B, 512, H/32, W/32]
#         x = F.adaptive_avg_pool2d(x, (7, 7))         # [B, 512, 7, 7]
#         out = self.localization_head(x)               # [B, 4]
#         out = torch.sigmoid(out)                      # Normalized [0, 1]
#         return out                                    # (cx, cy, w, h) normalized


# x = torch.randn(2, 3, 224, 224)
# model = VGG11Localizer()
# out = model(x)
# print(out.shape)   # EXPECT: [2, 4]
# print(out.min(), out.max())  # EXPECT: values in [0, 1]


"""Localization modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import CustomDropout
from models.vgg11 import VGG11

IMG_SIZE = 224 

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11(in_channels=in_channels)

        self.localization_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 4),   # (cx, cy, w, h)
        )
    def load_from_classifier(self, classifier_path: str):
        """Load encoder weights from a trained VGG11Classifier checkpoint."""
        sd = torch.load(classifier_path, map_location="cpu")
        if isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]
    
        # Extract only encoder keys, strip the "encoder." prefix
        encoder_sd = {
            k.replace("encoder.", ""): v
            for k, v in sd.items() if k.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_sd, strict=False)
        print(f"✅ Loaded classifier encoder → localizer. Missing: {missing}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        out  = torch.sigmoid(self.localization_head(feat))  # [B, 4] normalized

        # Convert to pixel space — autograder expects image-space coordinates
        scale = torch.tensor([IMG_SIZE, IMG_SIZE, IMG_SIZE, IMG_SIZE],
                              dtype=torch.float32, device=x.device)
        out = out * scale
        return out  # [B, 4] pixel-space cxcywh