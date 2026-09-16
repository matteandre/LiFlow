import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from liflow.datasets.dataloader.SemanticKITTITemporal import TemporalKITTISet
import warnings

warnings.filterwarnings('ignore')

__all__ = ['SemanticKittiDataModule']

class SparseSegmentCollation:

    def __call__(self, data):
        batch = list(zip(*data))

        return {'pcd_full': torch.stack(batch[0]).float(),
            'pcd_part' : torch.stack(batch[1]).float(),
            'filename': batch[2],
        }

class SemanticKittiDataModule(LightningDataModule):
    def __init__(self, data_dir, seqs, num_points, max_range, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.seqs = seqs
        self.num_points = num_points
        self.max_range = max_range
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.collate = SparseSegmentCollation()


    def prepare_data(self):
        # Augmentations
        pass

    def setup(self, stage=None):
        # Create datasets
        pass

    def train_dataloader(self):

        data_set = TemporalKITTISet(
            data_dir=self.data_dir,
            seqs=self.seqs['train'],
            split='train',
            num_points=self.num_points,
            max_range=self.max_range)
        loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.num_workers, collate_fn=self.collate)
        return loader

    def val_dataloader(self, pre_training=True):

        data_set = TemporalKITTISet(
            data_dir=self.data_dir,
            seqs=self.seqs['validation'],
            split='validation',
            num_points=self.num_points,
            max_range=self.max_range)
        loader = DataLoader(data_set, self.batch_size,
                            num_workers=self.num_workers, collate_fn=self.collate)
        return loader

    def test_dataloader(self):

        data_set = TemporalKITTISet(
            data_dir=self.data_dir,
            seqs=self.seqs['validation'],
            split='validation',
            num_points=self.num_points,
            max_range=self.max_range)
        loader = DataLoader(data_set, self.batch_size,
                             num_workers=self.num_workers, collate_fn=self.collate)
        return loader

dataloaders = {
    'KITTI': SemanticKittiDataModule,
}
