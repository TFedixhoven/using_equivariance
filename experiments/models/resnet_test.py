"""Unit tests for the Color Equivariant ResNet."""

import unittest

import torch
from experiments.classification.resnet import ResNet18, ResNet44

_create_dummy_input = lambda: torch.rand(2, 3, 32, 32)
_create_dummy_input_large = lambda: torch.rand(2, 3, 224, 224)


class TestResNet(unittest.TestCase):
    """Unit tests for the Color Equivariant ResNet."""

    def test_resnet18(self) -> None:
        """Test the forward pass of a regular ResNet18."""

        input = _create_dummy_input_large()

        model = ResNet18()
        with torch.no_grad():
            y = model(input)

        self.assertIsNotNone(y)

    def test_resnet18_separable(self) -> None:
        """Test the forward pass of a regular ResNet18."""

        input = _create_dummy_input_large()

        model = ResNet18(separable=True)
        with torch.no_grad():
            y = model(input)

        self.assertIsNotNone(y)

    def test_resnet44(self) -> None:
        """Test the forward pass of a regular ResNet44."""

        input = _create_dummy_input()

        model = ResNet44()
        with torch.no_grad():
            y = model(input)

        self.assertIsNotNone(y)

    def test_resnet44_separable(self) -> None:
        """Test the forward pass of a regular ResNet44."""

        input = _create_dummy_input()

        model = ResNet44(separable=True)
        with torch.no_grad():
            y = model(input)

        self.assertIsNotNone(y)


if __name__ == "__main__":
    unittest.main()
