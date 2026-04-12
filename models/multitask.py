import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown


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

        # Download weights
        gdown.download(id="1VrusgIINSRfmrx6WHxrWQsVHu68SJIvT", output=classifier_path, quiet=False)
        gdown.download(id="1-WYAwJGEQCW7q-VIkgLBfFf17PI6zvv9", output=localizer_path, quiet=False)
        gdown.download(id="1NrB99xG2jnJkCSLZXXz3qqJR6C0mTmZz", output=unet_path, quiet=False)

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


    def _load(self, model, path, name):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model_state" in sd:
            sd = sd["model_state"]
        try:
            model.load_state_dict(sd, strict=True)
            print(f"✅ Loaded {name} weights (strict)")
        except RuntimeError as e:
            model.load_state_dict(sd, strict=False)
            print(f"⚠️  Partial load for {name}: {e}")


    def forward(self, x):
        cls_out = self.classifier(x)
        loc_out = self.localizer(x)
        seg_out = self.segmenter(x)
        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }