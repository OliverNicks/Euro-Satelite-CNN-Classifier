"""Unit tests for SateliteCNN model architecture, weight checkpoint, and inference.

Supports both pytest and standard library unittest:
    python -m unittest tests/test_model.py
    pytest tests/
"""

import os
import unittest
from PIL import Image
import torch

from src.dataset import get_inference_transforms
from src.model import CLASS_NAMES, SateliteCNN, load_model
from src.predict import predict_image


class TestSateliteModel(unittest.TestCase):
    """Test suite for neural network layers, parameters, weights, and inference."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.weights_path = os.path.join("app", "satellite_model_9715.pth")
        assert os.path.exists(cls.weights_path), f"Checkpoint missing at {cls.weights_path}"
        cls.device = torch.device("cpu")

    def test_model_forward_pass_dimensions(self) -> None:
        """Verify tensor shapes through all 8 conv layers and classification head."""
        model = SateliteCNN(num_classes=10)
        model.eval()

        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (batch_size, 10))

    def test_model_parameter_count(self) -> None:
        """Ensure parameter count aligns with the 4-block SateliteCNN architecture (~1.7M)."""
        model = SateliteCNN(num_classes=10)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertTrue(1_500_000 < total_params < 2_000_000, f"Params: {total_params}")

    def test_checkpoint_loading(self) -> None:
        """Verify saved state dictionary loads cleanly into SateliteCNN."""
        model = load_model(weights_path=self.weights_path, device=self.device)
        self.assertIsInstance(model, SateliteCNN)

    def test_transforms_pipeline(self) -> None:
        """Verify inference transforms resize and normalize PIL image to (3, 64, 64)."""
        img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        transform = get_inference_transforms()
        tensor = transform(img)

        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertIsInstance(tensor, torch.Tensor)

    def test_predict_sample_forest(self) -> None:
        """Verify prediction on samples/Forest.jpg accurately classifies as Forest."""
        sample_path = os.path.join("samples", "Forest.jpg")
        if not os.path.exists(sample_path):
            self.skipTest("Sample image Forest.jpg not found.")

        model = load_model(weights_path=self.weights_path, device=self.device)
        result = predict_image(sample_path, model, self.device, top_k=3)

        self.assertEqual(result["predicted_class"], "Forest")
        self.assertGreater(result["confidence"], 70.0)
        self.assertEqual(len(result["top_predictions"]), 3)

    def test_all_classes_mapped(self) -> None:
        """Ensure exactly 10 EuroSAT classes are defined."""
        self.assertEqual(len(CLASS_NAMES), 10)
        self.assertIn("AnnualCrop", CLASS_NAMES)
        self.assertIn("SeaLake", CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
