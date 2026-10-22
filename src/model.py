"""Deep Convolutional Neural Network for Satellite Image Classification.

Matches the architecture trained in K12515096.ipynb achieving 97.15% accuracy.
"""

from typing import Dict, List, Optional
import os
import torch
import torch.nn as nn

# Standard EuroSAT RGB 10 classes ordered by alphabetical label indexing
CLASS_NAMES: List[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "AnnualCrop": "Regularly cultivated agricultural land with annual crops (cereals, vegetables).",
    "Forest": "Dense natural woodland and forested canopy areas.",
    "HerbaceousVegetation": "Natural grasslands, meadows, and wild low vegetation cover.",
    "Highway": "Paved major transport corridors, motorways, and infrastructure lines.",
    "Industrial": "Commercial complexes, manufacturing plants, logistics hubs, and large buildings.",
    "Pasture": "Managed grazing meadows and livestock agricultural grassland.",
    "PermanentCrop": "Long-term plantations including orchards, olive groves, and vineyards.",
    "Residential": "Urban and suburban settlements, houses, apartment blocks, and city centers.",
    "River": "Inland flowing water bodies, river channels, and canals.",
    "SeaLake": "Open coastal waters, inland lakes, reservoirs, and natural marine basins.",
}


class SateliteCNN(nn.Module):
    """Custom 8-convolutional-layer deep neural network with Batch Normalization,

    Dropout regularization, and Adaptive Average Pooling for 64x64 satellite imagery.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # === First stack of 2 conv layers ===
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1_2 = nn.BatchNorm2d(32)

        # === Second stack of 2 conv layers ===
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_2 = nn.BatchNorm2d(64)

        # === Third stack of 2 conv layers ===
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3_2 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.1)

        # === Fourth stack of 2 conv layers ===
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4_2 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout2d(0.2)

        # === Pooling & Activation ===
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()

        # === Classification Head ===
        self.linear1 = nn.Linear(256 * 2 * 2, 512)
        self.batchnorm_linear = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === Block 1 ===
        x = self.relu(self.batchnorm1_1(self.conv1_1(x)))
        x = self.relu(self.batchnorm1_2(self.conv1_2(x)))
        x = self.maxpool(x)

        # === Block 2 ===
        x = self.relu(self.batchnorm2_1(self.conv2_1(x)))
        x = self.relu(self.batchnorm2_2(self.conv2_2(x)))
        x = self.maxpool(x)

        # === Block 3 ===
        x = self.relu(self.batchnorm3_1(self.conv3_1(x)))
        x = self.relu(self.batchnorm3_2(self.conv3_2(x)))
        x = self.maxpool(x)
        x = self.dropout1(x)

        # === Block 4 ===
        x = self.relu(self.batchnorm4_1(self.conv4_1(x)))
        x = self.relu(self.batchnorm4_2(self.conv4_2(x)))
        x = self.dropout2(x)

        # === Adaptive Pool + Dense Layers ===
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.relu(self.batchnorm_linear(self.linear1(x)))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def load_model(
    weights_path: str = "app/satellite_model_9715.pth",
    device: Optional[torch.device] = None,
    num_classes: int = 10,
) -> SateliteCNN:
    """Instantiate and load pretrained weights into SateliteCNN."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    model = SateliteCNN(num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
