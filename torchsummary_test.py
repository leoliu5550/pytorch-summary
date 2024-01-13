import unittest
import os
print(os.getcwd())
from torchsummary import summary
from test_models import SingleInputNet, MultipleInputNet, MultipleInputNetDifferentDtypes
import torch

gpu_if_available = "cuda:0" if torch.cuda.is_available() else "cpu"

class torchsummaryTests(unittest.TestCase):
    def test_single_input(self):
        model = SingleInputNet()
        input = (1, 28, 28)
        result = summary(model, input, device="cpu")
        total_params, trainable_params = result["total_params"],result["trainable_params"]
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)

    def test_multiple_input(self):
        model = MultipleInputNet()
        input1 = (1, 300)
        input2 = (1, 300)
        result = summary(model,[input1, input2], device="cpu")
        total_params, trainable_params = result["total_params"],result["trainable_params"]
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

    def test_single_layer_network(self):
        model = torch.nn.Linear(2, 5)
        input = (1, 2)
        result = summary(model, input, device="cpu")
        total_params, trainable_params = result["total_params"],result["trainable_params"]
        self.assertEqual(total_params, 15)
        self.assertEqual(trainable_params, 15)

    def test_single_layer_network_on_gpu(self):
        model = torch.nn.Linear(2, 5)
        if torch.cuda.is_available():
            model.cuda()
        input = (1, 2)
        result = summary(model, input, device="cpu")
        total_params, trainable_params = result["total_params"],result["trainable_params"]
        self.assertEqual(total_params, 15)
        self.assertEqual(trainable_params, 15)

    def test_multiple_input_types(self):
        model = MultipleInputNetDifferentDtypes()
        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = [torch.FloatTensor, torch.LongTensor]
        result = summary(model, [input1, input2], device="cpu")
        total_params, trainable_params = result["total_params"],result["trainable_params"]
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)


class torchsummarystringTests(unittest.TestCase):
    def test_single_input(self):
        model = SingleInputNet()
        input = (1, 28, 28)

        result = summary(model, input, device="cpu")
        total_params, trainable_params = result["total_params"],result["trainable_params"]
        self.assertEqual(type(result["summary_str"]), str)
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)


if __name__ == '__main__':
    unittest.main(buffer=True)
