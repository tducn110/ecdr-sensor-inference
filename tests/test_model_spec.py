"""test_model_spec.py — Test architecture specifications, shapes, parameters, MAC, and RF.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Sections 3–7, Table 2).
"""

import unittest
import numpy as np
import torch

from ecdr.model import (
    ECDRConv1DNet,
    OperationalSequentialNumpyCNN,
    compute_layer_specs,
)


class TestModelSpecifications(unittest.TestCase):
    def test_layer_specs_k12(self):
        specs = compute_layer_specs(num_classes=12, input_length=128)

        # Output lengths
        self.assertEqual(specs["conv1"]["out_len"], 124)
        self.assertEqual(specs["conv2"]["out_len"], 122)
        self.assertEqual(specs["conv3"]["out_len"], 120)

        # Receptive fields
        self.assertEqual(specs["conv1"]["rf"], 5)
        self.assertEqual(specs["conv2"]["rf"], 7)
        self.assertEqual(specs["conv3"]["rf"], 9)
        self.assertEqual(specs["gap"]["rf"], 128)

        # Parameters
        self.assertEqual(specs["conv1"]["params"], 992)
        self.assertEqual(specs["conv2"]["params"], 6208)
        self.assertEqual(specs["conv3"]["params"], 12352)
        self.assertEqual(specs["head"]["params"], 780)
        self.assertEqual(specs["total"]["conv_params"], 19552)
        self.assertEqual(specs["total"]["total_params"], 20332)

        # MAC counts
        self.assertEqual(specs["conv1"]["mac"], 119040)
        self.assertEqual(specs["conv2"]["mac"], 749568)
        self.assertEqual(specs["conv3"]["mac"], 1474560)
        self.assertEqual(specs["head"]["mac"], 768)
        self.assertEqual(specs["total"]["conv_mac"], 2343168)
        self.assertEqual(specs["total"]["total_mac"], 2343936)

    def test_layer_specs_k6(self):
        specs = compute_layer_specs(num_classes=6, input_length=128)
        self.assertEqual(specs["total"]["conv_params"], 19552)
        self.assertEqual(specs["head"]["params"], 6 * 65)  # 390
        self.assertEqual(specs["total"]["total_params"], 19552 + 390)  # 19942
        self.assertEqual(specs["total"]["total_mac"], 2343168 + 64 * 6)  # 2343552

    def test_torch_model_parameter_count(self):
        torch_net = ECDRConv1DNet(num_classes=12)
        total_p = sum(p.numel() for p in torch_net.parameters())
        self.assertEqual(total_p, 20332)

        conv_p = (
            sum(p.numel() for p in torch_net.conv1.parameters())
            + sum(p.numel() for p in torch_net.conv2.parameters())
            + sum(p.numel() for p in torch_net.conv3.parameters())
        )
        self.assertEqual(conv_p, 19552)

    def test_torch_vs_numpy_sequential_forward(self):
        torch.manual_seed(42)
        torch_net = ECDRConv1DNet(num_classes=12)
        numpy_net = OperationalSequentialNumpyCNN.from_torch_module(torch_net)

        np.random.seed(42)
        x = np.random.randn(6, 128).astype(np.float32)

        with torch.no_grad():
            torch_out = torch_net(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()

        numpy_out = numpy_net.forward(x.astype(np.float64))

        # Check maximum absolute error between float32 torch and float64 numpy
        max_err = np.max(np.abs(torch_out - numpy_out["logits"]))
        self.assertLess(max_err, 1e-5)
        self.assertEqual(int(torch_out.argmax()), numpy_out["pred_class"])

    def test_nonfinite_input_rejection(self):
        torch_net = ECDRConv1DNet(num_classes=12)
        numpy_net = OperationalSequentialNumpyCNN.from_torch_module(torch_net)

        x_nan = np.zeros((6, 128), dtype=np.float64)
        x_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            numpy_net.forward(x_nan)

        x_inf = np.zeros((6, 128), dtype=np.float64)
        x_inf[0, 0] = np.inf
        with self.assertRaises(ValueError):
            numpy_net.forward(x_inf)

    def test_wrong_shape_rejection(self):
        torch_net = ECDRConv1DNet(num_classes=12)
        numpy_net = OperationalSequentialNumpyCNN.from_torch_module(torch_net)

        with self.assertRaises(ValueError):
            numpy_net.forward(np.zeros((6, 64), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
