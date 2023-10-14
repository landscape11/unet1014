import argparse
import datetime
import os
import socket
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from dsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from model_seg import UNetWrapper, SegmentationAugmentation
from train_seg import LunaTrainingApp

class BenchmarkLuna2dSegmentationDataset(TrainingLuna2dSegmentationDataset):
    def __len__(self):
        return 5000

class LunaBenchmarkApp(LunaTrainingApp):
    def initTrainDl(self):
        train_ds = BenchmarkLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def main(self):
        train_dl = self.initTrainDl()

        for epoch_ndx in range(1, 2):
            self.doTraining(epoch_ndx, train_dl)

if __name__ == '__main__':
    LunaBenchmarkApp().main()
