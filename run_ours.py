import argparse
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines.utils import get_model, get_devices
from pyannote.core import Timeline, Segment
from pyannote.database import get_protocol
from pyannote.database.util import FileFinder
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim import Adam

import utils
from augmentation import get_augmentation, get_audio_file_paths, get_noise_augmentation
from evaluation import get_cder
from protocols import SplitFileProtocol
from task import SupervisedSegmentation, PseudoLabelingSegmentation

parser = argparse.ArgumentParser()
parser.add_argument("protocol", type=str,
                    help="A pyannote database protocol. The train set will be used as training sequence")
parser.add_argument("-e", "--eval-subset", type=str, default="development",
                    help="A subset to evaluate the model (train | development | test)")
parser.add_argument("-m", "--model", type=str, default="AMI_pretrained_model.ckpt",
                    help="The pre-trained model (m_0)")
parser.add_argument("-s", "--supervision", required=True, type=str,
                    help="The type of supervision (labels | pseudo)")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
parser.add_argument("-b", "--batch-size", type=int, default=64)
parser.add_argument("--clean-pseudo-labels", dest="clean_pseudo_labels", action="store_true",
                    help="Obtain pseudo-labels without augmentation aug (ours1|2)")
parser.add_argument("--eval-matrix", dest="eval_matrix", action="store_true",
                    help="Calculate performance on all train conversations at each step in the training sequence")
parser.add_argument("--noise", type=str, nargs="+", default=[], help="Paths to directories with noise files")
parser.add_argument("--rir", type=str, help="Path to impulse response files")
parser.add_argument("--logdir", type=str, help="Path to logging directory")
parser.add_argument("--noise-snr-train", type=str, default="0-5",
                    help="Sound-to-noise range in dB for noise augmentation "
                         "during training. min-max (e.g. 10-15)")
parser.add_argument("--noise-snr-pl", type=str, default="5-10",
                    help="Sound-to-noise range in dB for noise augmentation "
                         "for pseudo-label calculation. min-max (e.g. 10-15)")
args = parser.parse_args()
assert args.supervision in ["labels", "pseudo"]
if args.logdir is not None:
    args.logdir = Path(args.logdir)
args.noise_snr_train = utils.format_float_range(args.noise_snr_train)
args.noise_snr_pl = utils.format_float_range(args.noise_snr_pl)


# TODO refactor common code between whole and ours
def configure_optimizers(model):
    return Adam(model.parameters(), lr=args.learning_rate)


protocol = get_protocol(args.protocol, preprocessors={"audio": FileFinder()})
file_splitter = utils.FileSplitter()

noise_paths = get_audio_file_paths(args.noise)
augmentation = get_augmentation(
    noise_paths,
    args.rir,
    noise_min_snr=args.noise_snr_train[0],
    noise_max_snr=args.noise_snr_train[1],
    rir_p=0.5,
)

pl_augmentation = None
if not args.clean_pseudo_labels and noise_paths is not None:
    pl_augmentation = get_noise_augmentation(
        noise_paths,
        min_snr=args.noise_snr_pl[0],
        max_snr=args.noise_snr_pl[1],
        p=1.0,
    )

pl_generator = get_model(args.model) if args.supervision == "pseudo" else None

device = get_devices(needs=1)[0]
audio = Audio(sample_rate=16000, mono=True)
training_files = list(protocol.train())
num_train_files = len(training_files)

performance = np.zeros(num_train_files)
confusion = np.zeros_like(performance)
miss = np.zeros_like(performance)
fa = np.zeros_like(performance)

train_performance = np.zeros_like(performance)
train_confusion = np.zeros_like(performance)
train_miss = np.zeros_like(performance)
train_fa = np.zeros_like(performance)

performance_matrix = np.zeros((num_train_files, num_train_files))
confusion_matrix = np.zeros_like(performance_matrix)
miss_matrix = np.zeros_like(performance_matrix)
fa_matrix = np.zeros_like(performance_matrix)


continual_protocol = SplitFileProtocol()
segmentation = get_model(args.model)
train_samples, val_samples = [], []
for i_file, file in enumerate(training_files):
    print(f"---------- {file['uri']} ----------")
    if args.supervision == "pseudo":
        file["annotated"] = Timeline([Segment(0, audio.get_duration(file))])
    # Split file into train and dev
    train, dev = file_splitter.split(file)
    # Create a protocol with file's train and dev
    continual_protocol.update(train, dev)
    # Create a configured segmentation task
    if args.supervision == "labels":
        task = SupervisedSegmentation(
            protocol=continual_protocol,
            duration=segmentation.specifications.duration,
            max_num_speakers=len(segmentation.specifications.classes),
            batch_size=args.batch_size,
            weight=None,
            overlap={"probability": 0},
            num_workers=1,
            augmentation=augmentation,
        )
    else:
        # Use a snapshot of the segmentation model at this point to generate pseudo-labels
        pl_generator.load_state_dict(segmentation.state_dict())
        task = PseudoLabelingSegmentation(
            protocol=continual_protocol,
            pseudo_label_generator=pl_generator,
            pseudo_label_device=device,
            duration=segmentation.specifications.duration,
            max_num_speakers=len(segmentation.specifications.classes),
            batch_size=args.batch_size,
            weight=None,
            overlap={"probability": 0},
            num_workers=0,  # Can't use GPU on other threads
            augmentation=augmentation,
            pseudo_label_augmentation=pl_augmentation,
        )

    # Configure the model to train with this task
    segmentation.task = task
    segmentation.configure_optimizers = MethodType(
        configure_optimizers, segmentation
    )
    segmentation.setup(stage="fit")

    # Save a checkpoint of the current model state
    prev_state_dict = segmentation.state_dict()

    # Configure training
    early_stop = EarlyStopping("seg@val_auroc", patience=3, mode="max")
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1000,
        checkpoint_callback=False,
        num_sanity_val_steps=0,
        logger=False,
        callbacks=[early_stop],
        weights_summary=None,
    )

    # Calculate development AUROC before training
    auroc_before = trainer.validate(segmentation, verbose=False)[0]["seg@val_auroc"]
    # Train model
    trainer.fit(segmentation)
    # Calculate development AUROC after training
    auroc_after = trainer.validate(segmentation, verbose=False)[0]["seg@val_auroc"]

    print(f"\nAUROC: {100 * auroc_before:.4f}% -> {100 * auroc_after:.4f}%")

    # Keep best or last checkpoint
    if auroc_before > auroc_after:
        segmentation.load_state_dict(prev_state_dict)

    metric = get_cder(segmentation, protocol, args.eval_subset, args.batch_size, device)
    cder = abs(metric)
    performance[i_file] = cder
    confusion[i_file] = metric["confusion"] / metric["total"]
    miss[i_file] = metric["missed detection"] / metric["total"]
    fa[i_file] = metric["false alarm"] / metric["total"]
    if args.eval_matrix:
        report = get_cder(segmentation, protocol, "train", args.batch_size, device).report()
        performance_matrix[i_file] = np.array(report["discrete diarization error rate"][:-1]).squeeze()
        confusion_matrix[i_file] = np.array(report["confusion"]["%"][:-1]).squeeze()
        miss_matrix[i_file] = np.array(report["missed detection"]["%"][:-1]).squeeze()
        fa_matrix[i_file] = np.array(report["false alarm"]["%"][:-1]).squeeze()
        train_performance[i_file] = np.array(report["discrete diarization error rate"]).squeeze()[-1]
        train_confusion[i_file] = np.array(report["confusion"]["%"]).squeeze()[-1]
        train_miss[i_file] = np.array(report["missed detection"]["%"]).squeeze()[-1]
        train_fa[i_file] = np.array(report["false alarm"]["%"]).squeeze()[-1]
    print(f"{args.eval_subset} CDER: {100 * cder:.4f}%")

print("Confusion:")
print(repr(confusion))
print()
print("Missed detection:")
print(repr(miss))
print()
print("False alarm:")
print(repr(fa))
print()
print("CDER:")
print(repr(performance))
print()
print(f"Mean end CDER: {100 * np.mean(performance[:, -1]):.4f}")
print(f"Std end CDER: {100 * np.std(performance[:, -1]):.4f}")

if args.logdir is not None:
    for res, name in zip([performance, confusion, miss, fa], ["cder", "confusion", "miss", "fa"]):
        pd.DataFrame(res).to_csv(
            args.logdir / f"{args.eval_subset}_{name}.csv", header=False, index=False
        )
    if args.eval_matrix:
        for matrix, name in zip(
            [performance_matrix, confusion_matrix, miss_matrix, fa_matrix],
            ["cder", "confusion", "miss", "fa"]
        ):
            np.save(args.logdir / f"train_{name}_matrix.npy", matrix)
            pd.DataFrame(np.mean(matrix, axis=0)).to_csv(
                args.logdir / f"train_avg_{name}_matrix.csv", header=False, index=False,
            )
            pd.DataFrame(np.std(matrix, axis=0)).to_csv(
                args.logdir / f"train_std_{name}_matrix.csv", header=False, index=False,
            )
        for res, name in zip(
            [train_performance, train_confusion, train_miss, train_fa],
            ["cder", "confusion", "miss", "fa"]
        ):
            pd.DataFrame(res).to_csv(
                args.logdir / f"train_{name}.csv", header=False, index=False
            )
