import argparse
from pathlib import Path
from types import MethodType
from typing import Optional, Text

import pytorch_lightning as pl
from pyannote.audio.core.model import Model
from pyannote.audio.pipelines.utils import get_model, get_devices
from pyannote.audio.tasks import Segmentation
from pyannote.database import get_protocol, Protocol
from pyannote.database.util import FileFinder
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

import utils
from augmentation import get_augmentation, get_audio_file_paths, get_noise_augmentation
from evaluation import get_cder
from task import PseudoLabelingSegmentation

parser = argparse.ArgumentParser()
parser.add_argument("protocol", type=str, help="A pyannote database protocol")
parser.add_argument("-m", "--model", type=str, default="AMI_pretrained_model.ckpt", help="A pre-trained model")
parser.add_argument("-e", "--eval-subset", type=str, default="development",
                    help="A subset to evaluate the model (train | development | test)")
parser.add_argument("-s", "--supervision", required=True, type=str, help="(labels | pseudo)")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
parser.add_argument("-b", "--batch-size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--noise", type=str, nargs="+", default=[], help="Paths to directories with noise files")
parser.add_argument("--rir", type=str, help="Path to impulse response files")
parser.add_argument("--clean-pseudo-labels", dest="clean_pseudo_labels", action="store_true",
                    help="Obtain pseudo-labels without augmentation aug (whole1|2)")
parser.add_argument("--clean-training", dest="clean_training", action="store_true",
                    help="Remove augmentation from training (whole1|4). It doesn't affect pseudo-labels")
parser.add_argument("--noise-snr-train", type=str, default="0-5",
                    help="Sound-to-noise range in dB for noise augmentation "
                         "during training. min-max (e.g. 10-15)")
parser.add_argument("--noise-snr-pl", type=str, default="5-10",
                    help="Sound-to-noise range in dB for noise augmentation "
                         "for pseudo-label calculation. min-max (e.g. 10-15)")
parser.add_argument("--logdir", type=str, default=".",
                    help="Path to logging directory. Defaults to current directory")
args = parser.parse_args()
assert args.supervision in ["labels", "pseudo"]
args.logdir = Path(args.logdir)
args.noise_snr_train = utils.format_float_range(args.noise_snr_train)
args.noise_snr_pl = utils.format_float_range(args.noise_snr_pl)


# TODO refactor common code between whole and ours
def configure_optimizers(model):
    return Adam(model.parameters(), lr=args.learning_rate)


def get_task(
    model: Model,
    protocol: Protocol,
    weight: Optional[Text] = None,
    augmentation: Optional[BaseWaveformTransform] = None,
):
    if args.supervision == "labels":
        return Segmentation(
            protocol=protocol,
            duration=model.specifications.duration,
            max_num_speakers=len(model.specifications.classes),
            batch_size=args.batch_size,
            weight=weight,
            overlap={"probability": 0},
            num_workers=4,
            augmentation=augmentation,
        )
    else:
        pl_generator = get_model(args.model)
        pl_generator.load_state_dict(model.state_dict())
        pl_augmentation = None
        if not args.clean_pseudo_labels and args.noise:
            pl_augmentation = get_noise_augmentation(
                get_audio_file_paths(args.noise),
                min_snr=args.noise_snr_pl[0],
                max_snr=args.noise_snr_pl[1],
                p=1.0
            )
        return PseudoLabelingSegmentation(
            protocol=protocol,
            pseudo_label_generator=pl_generator,
            duration=model.specifications.duration,
            max_num_speakers=len(model.specifications.classes),
            batch_size=args.batch_size,
            weight=weight,
            overlap={"probability": 0},
            num_workers=0,
            augmentation=augmentation,
            pseudo_label_augmentation=pl_augmentation,
        )


protocol = get_protocol(args.protocol, preprocessors={"audio": FileFinder()})
augmentation = None
if not args.clean_training:
    augmentation = get_augmentation(
        noise_files=get_audio_file_paths(args.noise),
        rir_root=args.rir,
        noise_min_snr=args.noise_snr_train[0],
        noise_max_snr=args.noise_snr_train[1],
        rir_p=0.5,
    )
segmentation = get_model(args.model)
# Create a segmentation task and configure the model to train with it
segmentation.task = get_task(segmentation, protocol, augmentation=augmentation)
segmentation.configure_optimizers = MethodType(
    configure_optimizers, segmentation
)
segmentation.setup(stage="fit")

# Configure training
checkpoint_callback = ModelCheckpoint(
    dirpath=args.logdir / "best_ckpt", monitor="seg@val_auroc", mode="max", save_top_k=1
)
logger = TensorBoardLogger(save_dir=args.logdir / "training_logs", name=None)
trainer = pl.Trainer(
    gpus=1,
    max_epochs=args.epochs,
    num_sanity_val_steps=0,
    logger=logger,
    checkpoint_callback=True,
    callbacks=[checkpoint_callback],
)
trainer.fit(segmentation)

# Load best checkpoint
best_ckpt = trainer.checkpoint_callback.best_model_path
segmentation = Model.from_pretrained(best_ckpt)
segmentation.task = get_task(segmentation, protocol, augmentation=augmentation)
segmentation.setup(stage="fit")

# Evaluate best checkpoint
metric = get_cder(segmentation, protocol, args.eval_subset, args.batch_size, get_devices(1)[0])
cder = abs(metric)
conf = metric["confusion"] / metric["total"]
miss = metric["missed detection"] / metric["total"]
fa = metric["false alarm"] / metric["total"]
print(f"{args.eval_subset} CDER: {100 * cder:.4f}%")
