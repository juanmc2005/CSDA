from typing import Union, Tuple, Text, Literal, Optional

import torch
from pyannote.audio import Model
from pyannote.audio.core.task import Problem
from pyannote.audio.pipelines.utils import get_devices
from pyannote.audio.tasks import Segmentation
from pyannote.database import Protocol
from torch.utils.data._utils.collate import default_collate
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform


class SupervisedSegmentation(Segmentation):
    def validation_step(self, batch, batch_idx: int):
        """Compute validation area under the ROC curve
        (copied from `SegmentationTaskMixin` to remove the plotting)

        Parameters
        ----------
        batch : dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        """

        X, y = batch["X"], batch["y"]
        # X = (batch_size, num_channels, num_samples)
        # y = (batch_size, num_frames, num_classes) or (batch_size, num_frames)

        y_pred = self.model(X)
        _, num_frames, _ = y_pred.shape
        # y_pred = (batch_size, num_frames, num_classes)

        # postprocess
        y_pred = self.validation_postprocess(y, y_pred)

        # - remove warm-up frames
        # - downsample remaining frames
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        preds = y_pred[:, warm_up_left : num_frames - warm_up_right : 10]
        target = y[:, warm_up_left : num_frames - warm_up_right : 10]

        # torchmetrics tries to be smart about the type of machine learning problem
        # pyannote.audio is more explicit so we have to reshape target and preds for
        # torchmetrics to be happy... more details can be found here:
        # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#input-types

        if self.specifications.problem == Problem.BINARY_CLASSIFICATION:
            # target: shape (batch_size, num_frames), type binary
            # preds:  shape (batch_size, num_frames, 1), type float

            # torchmetrics expects:
            # target: shape (N,), type binary
            # preds:  shape (N,), type float

            self.model.validation_metric(preds.reshape(-1), target.reshape(-1))

        elif self.specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
            # target: shape (batch_size, num_frames, num_classes), type binary
            # preds:  shape (batch_size, num_frames, num_classes), type float

            # torchmetrics expects
            # target: shape (N, ), type binary
            # preds:  shape (N, ), type float

            self.model.validation_metric(preds.reshape(-1), target.reshape(-1))

        elif self.specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
            # target: shape (batch_size, num_frames, num_classes), type binary
            # preds:  shape (batch_size, num_frames, num_classes), type float

            # torchmetrics expects:
            # target: shape (N, ), type int
            # preds:  shape (N, num_classes), type float

            # TODO: implement when pyannote.audio gets its first mono-label segmentation task
            raise NotImplementedError()

        self.model.log(
            f"{self.ACRONYM}@val_auroc",
            self.model.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


class PseudoLabelingSegmentation(SupervisedSegmentation):
    def __init__(
        self,
        protocol: Protocol,
        pseudo_label_generator: Model,
        pseudo_label_device: Optional[torch.device] = None,
        duration: float = 2.0,
        max_num_speakers: int = None,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        overlap: dict = Segmentation.OVERLAP_DEFAULTS,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        loss: Literal["bce", "mse"] = "bce",
        vad_loss: Literal["bce", "mse"] = None,
        pseudo_label_augmentation: BaseWaveformTransform = None,
    ):
        super().__init__(
            protocol,
            duration,
            max_num_speakers,
            warm_up,
            overlap,
            balance,
            weight,
            batch_size,
            num_workers,
            pin_memory,
            augmentation,
            loss,
            vad_loss
        )
        self.pl_generator = pseudo_label_generator
        self.pl_augmentation = pseudo_label_augmentation
        self.pl_device = pseudo_label_device
        if self.pl_device is None:
            self.pl_device = get_devices(needs=1)[0]
        self.pl_generator.to(self.pl_device)

    @property
    def sample_rate(self) -> int:
        return self.model.hparams.sample_rate

    @torch.inference_mode()
    def collate_fn(self, batch):
        collated_batch = default_collate(batch)
        train_X = collated_batch["X"]
        pl_X = collated_batch["X"]
        if self.augmentation is not None:
            train_X = self.augmentation(train_X, sample_rate=self.sample_rate)
        if self.pl_augmentation is not None:
            pl_X = self.pl_augmentation(pl_X, sample_rate=self.sample_rate)
        pseudo_labels = self.pl_generator(pl_X.to(self.pl_device))
        pseudo_labels = (pseudo_labels >= 0.5).int()
        collated_batch["X"] = train_X
        collated_batch["y"] = pseudo_labels
        return collated_batch
