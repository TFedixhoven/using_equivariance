"""Unit tests for the P4M equivariant ResNet architectures."""

import unittest

import torch
from experiments.classification.resnet_p4m import (
    P4MResNet18,
    P4MResNet44,
    gcP4MResNet18,
    gcP4MResNet44,
)

_create_dummy_input = lambda: torch.rand(2, 3, 32, 32)
_create_dummy_input_exact = lambda: torch.rand(2, 3, 33, 33)
_create_dummy_input_large = lambda: torch.rand(2, 3, 224, 224)
_create_dummy_input_large_exact = lambda: torch.rand(2, 3, 225, 225)


class TestP4MResNet(unittest.TestCase):
    """Unit tests for the Color Equivariant ResNet."""

    def test_resnet18_p4m_inexact(self) -> None:
        """Test the forward pass of a P4M ResNet18.

        Should not be exactly equivariant for a 224x224 input."""

        input = _create_dummy_input_large()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = P4MResNet18()
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        y, _ = torch.max(y, dim=2)
        y_rot, _ = torch.max(y_rot, dim=2)

        # Test equivariance
        self.assertFalse(torch.allclose(y, torch.rot90(y_rot, -1, [2, 3]), atol=1e-4))

    def test_resnet18_p4m_inexact_groupcosetpool(self) -> None:
        """Test the forward pass of a separable P4M ResNet18.

        Should not be exactly equivariant for a 224x224 input."""

        input = _create_dummy_input_large()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = P4MResNet18(groupcosetmaxpool=True)
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        self.assertFalse(torch.allclose(y, y_rot, atol=1e-4))

    def test_gc_resnet18_p4m_inexact(self) -> None:
        """Test the forward pass of a separable P4M ResNet18.

        Should not be exactly equivariant for a 224x224 input."""

        input = _create_dummy_input_large()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = gcP4MResNet18()
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        y, _ = torch.max(y, dim=2)
        y_rot, _ = torch.max(y_rot, dim=2)

        self.assertFalse(torch.allclose(y, torch.rot90(y_rot, -1, [2, 3]), atol=1e-4))

    def test_gc_resnet18_p4m_inexact_groupcosetpool(self) -> None:
        """Test the forward pass of a separable P4M ResNet18.

        Should not be exactly equivariant for a 224x224 input."""

        input = _create_dummy_input_large()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = gcP4MResNet18(groupcosetmaxpool=True)
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        self.assertFalse(torch.allclose(y, y_rot, atol=1e-4))

    def test_resnet18_p4m_exact(self) -> None:
        """Test the forward pass of a P4M ResNet18.

        Should be exactly equivariant for a 225x225 input."""

        input = _create_dummy_input_large_exact()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = P4MResNet18()
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        y, _ = torch.max(y, dim=2)
        y_rot, _ = torch.max(y_rot, dim=2)

        self.assertTrue(torch.allclose(y, torch.rot90(y_rot, -1, [2, 3]), atol=1e-4))

    def test_resnet18_p4m_exact_groupcosetpool(self) -> None:
        """Test the forward pass of a P4M ResNet18.

        Should be exactly equivariant for a 225x225 input."""

        input = _create_dummy_input_large_exact()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = P4MResNet18(groupcosetmaxpool=True)
        with torch.no_grad():
            y, _ = model(input)
            y_rot, _ = model(input_rotated)

        self.assertTrue(torch.allclose(y, y_rot, atol=1e-4))

    def test_gc_resnet18_p4m_exact(self) -> None:
        """Test the forward pass of a separable P4M ResNet18.

        Should be exactly equivariant for a 225x225 input."""

        input = _create_dummy_input_large_exact()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = gcP4MResNet18()
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        y, _ = torch.max(y, dim=2)
        y_rot, _ = torch.max(y_rot, dim=2)

        self.assertTrue(torch.allclose(y, torch.rot90(y_rot, -1, [2, 3]), atol=1e-4))

    def test_gc_resnet18_p4m_exact_groupcosetpool(self) -> None:
        """Test the forward pass of a separable P4M ResNet18.

        Should be exactly equivariant for a 225x225 input."""

        input = _create_dummy_input_large_exact()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = gcP4MResNet18(groupcosetmaxpool=True)
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        self.assertFalse(torch.allclose(y, y_rot, atol=1e-4))

    def test_resnet44_p4m_inexact(self) -> None:
        """Test the forward pass of a P4M ResNet44.

        Should not be exactly equivariant for a 224x224 input."""

        input = _create_dummy_input()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = P4MResNet44()
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        y, _ = torch.max(y, dim=2)
        y_rot, _ = torch.max(y_rot, dim=2)

        # Test equivariance
        self.assertFalse(torch.allclose(y, torch.rot90(y_rot, -1, [2, 3]), atol=1e-4))

    def test_resnet44_p4m_inexact_groupcosetpool(self) -> None:
        """Test the forward pass of a separable P4M ResNet44.

        Should not be exactly equivariant for a 224x224 input."""

        input = _create_dummy_input()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = P4MResNet44(groupcosetmaxpool=True)
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        self.assertFalse(torch.allclose(y, y_rot, atol=1e-4))

    def test_gc_resnet44_p4m_inexact(self) -> None:
        """Test the forward pass of a separable P4M ResNet44.

        Should not be exactly equivariant for a 224x224 input."""

        input = _create_dummy_input()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = gcP4MResNet44()
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        y, _ = torch.max(y, dim=2)
        y_rot, _ = torch.max(y_rot, dim=2)

        self.assertFalse(torch.allclose(y, torch.rot90(y_rot, -1, [2, 3]), atol=1e-4))

    def test_gc_resnet44_p4m_inexact_groupcosetpool(self) -> None:
        """Test the forward pass of a separable P4M ResNet44.

        Should not be exactly equivariant for a 224x224 input."""

        input = _create_dummy_input()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = gcP4MResNet44(groupcosetmaxpool=True)
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        self.assertFalse(torch.allclose(y, y_rot, atol=1e-4))

    def test_resnet44_p4m_exact(self) -> None:
        """Test the forward pass of a P4M ResNet44.

        Should be exactly equivariant for a 225x225 input."""

        input = _create_dummy_input_exact()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = P4MResNet44()
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        y, _ = torch.max(y, dim=2)
        y_rot, _ = torch.max(y_rot, dim=2)

        self.assertTrue(torch.allclose(y, torch.rot90(y_rot, -1, [2, 3]), atol=1e-4))

    def test_resnet44_p4m_exact_groupcosetpool(self) -> None:
        """Test the forward pass of a P4M ResNet44.

        Should be exactly equivariant for a 225x225 input."""

        input = _create_dummy_input_exact()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = P4MResNet44(groupcosetmaxpool=True)
        with torch.no_grad():
            y, _ = model(input)
            y_rot, _ = model(input_rotated)

        self.assertTrue(torch.allclose(y, y_rot, atol=1e-4))

    def test_gc_resnet44_p4m_exact(self) -> None:
        """Test the forward pass of a separable P4M ResNet44.

        Should be exactly equivariant for a 225x225 input."""

        input = _create_dummy_input_exact()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = gcP4MResNet44()
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        y, _ = torch.max(y, dim=2)
        y_rot, _ = torch.max(y_rot, dim=2)

        self.assertTrue(torch.allclose(y, torch.rot90(y_rot, -1, [2, 3]), atol=1e-4))

    def test_gc_resnet44_p4m_exact_groupcosetpool(self) -> None:
        """Test the forward pass of a separable P4M ResNet44.

        Should be exactly equivariant for a 225x225 input."""

        input = _create_dummy_input_exact()
        input_rotated = torch.rot90(input, 1, [2, 3])

        model = gcP4MResNet44(groupcosetmaxpool=True)
        with torch.no_grad():
            _, y = model(input)
            _, y_rot = model(input_rotated)

        self.assertFalse(torch.allclose(y, y_rot, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
