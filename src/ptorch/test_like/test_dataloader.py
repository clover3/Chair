import torch
from torch.utils.data import DataLoader, TensorDataset
import unittest

from ptorch.splade_tree.datasets.pep_dataloaders import PEPPairsDataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset for demonstration purposes
        self.data = torch.randn(100, 3)  # 100 samples, 3 features each
        self.targets = torch.randint(0, 2, (100,))  # 100 binary targets
        self.dataset = TensorDataset(self.data, self.targets)
        max_seq_length = 256
        tokenizer_type = "bert-base-uncased"
        self.batch_size = 10
        self.dataloader = PEPPairsDataLoader(tokenizer_type, max_seq_length)

    def test_batch_size(self):
        for data, targets in self.dataloader:
            # Check if each batch has the correct batch size
            self.assertEqual(data.size(0), self.batch_size)
            self.assertEqual(targets.size(0), self.batch_size)

    def test_data_type(self):
        for data, targets in self.dataloader:
            # Check if the data and targets are of expected type (torch.Tensor in this case)
            self.assertIsInstance(data, torch.Tensor)
            self.assertIsInstance(targets, torch.Tensor)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
