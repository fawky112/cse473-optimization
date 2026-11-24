import unittest

import torch
import torch.nn as nn

from src.model import LeakyGELUCNN


class TestModel(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(42)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = LeakyGELUCNN().to(self.device)

        self.images = torch.randn(
            16, 1, 28, 28,
            device=self.device
        )

        self.labels = torch.randint(
            0, 10, (16,),
            device=self.device
        )


    def test_parameter_count(self):

        parameters = sum(
            p.numel()
            for p in self.model.parameters()
        )

        self.assertEqual(parameters, 475434)


    def test_block1_shape(self):

        output = self.model.block1(self.images)

        self.assertEqual(
            output.shape,
            (16, 32, 14, 14)
        )


    def test_block2_shape(self):

        x = self.model.block1(self.images)

        output = self.model.block2(x)

        self.assertEqual(
            output.shape,
            (16, 64, 7, 7)
        )


    def test_output_shape(self):

        output = self.model(self.images)

        self.assertEqual(output.shape, (16, 10))


    def test_softmax(self):

        with torch.no_grad():

            probabilities = self.model.predict_proba(
                self.images
            )

        sums = probabilities.sum(dim=1)

        self.assertTrue(
            torch.allclose(
                sums,
                torch.ones_like(sums),
                atol=1e-6
            )
        )


    def test_backpropagation(self):

        criterion = nn.CrossEntropyLoss()

        output = self.model(self.images)

        loss = criterion(output, self.labels)

        loss.backward()

        print("\nGradient Norm Analysis")
        print("Device:", self.device)
        print("Loss:", loss.item())

        for name, parameter in self.model.named_parameters():

            self.assertIsNotNone(parameter.grad)

            self.assertTrue(
                torch.isfinite(parameter.grad).all().item()
            )

            grad_norm = parameter.grad.norm().item()

            print(f"{name:<35} {grad_norm:.8f}")


if __name__ == "__main__":

    unittest.main()