"""
Pre-trained CNN Feature Extractor module.
Supports MobileNetV3 and ResNet backbones to transform raw images into dense visual feature vectors.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Any, Tuple

class CNNFeatureExtractor:
    def __init__(self, backbone: str = "mobilenet_v3_small", device: str = None):
        self.backbone_name = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Standard ImageNet normalization and augmentation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self._init_model()

    def _init_model(self):
        if self.backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=weights)
            # MobileNetV3 classifier: [Linear(576, 1024), Hardswish, Dropout, Linear(1024, 1000)]
            # We preserve the full model for top-k classification logits,
            # and extract feature vector from classifier[0] (1024-dim)
            self.feature_dim = 1024
            self.categories = weights.meta["categories"]
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
            self.feature_dim = 2048
            self.categories = weights.meta["categories"]
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def extract_features(self, pil_image: Image.Image) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract visual feature vector and top semantic concept categories from an image.
        
        Returns:
            feature_tensor: Tensor of shape (1, feature_dim)
            metadata: Top detected categories, confidence scores, image dimensions
        """
        # Ensure RGB format
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            
        orig_w, orig_h = pil_image.size
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        if self.backbone_name == "mobilenet_v3_small":
            # Extract penultimate representation
            x = self.model.features(input_tensor)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            # Run through first layer of classifier: Linear(576, 1024) + Hardswish
            feat = self.model.classifier[0](x)
            feat = self.model.classifier[1](feat) # 1024-dim
            
            # Also compute full logits for semantic clues
            logits = self.model.classifier(x)
        else: # resnet50
            # Run up to avgpool
            x = self.model.conv1(input_tensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            feat = torch.flatten(x, 1) # 2048-dim
            logits = self.model.fc(feat)
            
        probs = torch.softmax(logits, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, 5)
        
        top_concepts = []
        for prob, idx in zip(top5_probs[0].tolist(), top5_indices[0].tolist()):
            cat_name = self.categories[idx] if idx < len(self.categories) else f"class_{idx}"
            top_concepts.append({
                "concept": cat_name,
                "confidence": round(prob, 4)
            })

        metadata = {
            "backbone": self.backbone_name,
            "feature_dim": self.feature_dim,
            "image_size": [orig_w, orig_h],
            "top_concepts": top_concepts,
            "device": self.device
        }

        return feat, metadata
