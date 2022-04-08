from typing import List, Optional, Text, Union
from pathlib import Path
from torch_audiomentations.utils.file import find_audio_files
from torch_audiomentations import AddBackgroundNoise, ApplyImpulseResponse, Compose
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform


def get_audio_file_paths(
    paths: List[Union[Path, Text]]
) -> Optional[List[Path]]:
    if not paths:
        return None
    noise_paths = []
    for path in paths:
        noise_paths.extend(find_audio_files(path))
    return noise_paths


def get_noise_augmentation(
    paths: List[Path],
    min_snr: float,
    max_snr: float,
    p: float
) -> Optional[BaseWaveformTransform]:
    return AddBackgroundNoise(
        paths,
        min_snr_in_db=min_snr,
        max_snr_in_db=max_snr,
        mode="per_example",
        p=p,
    )


def get_rir_augmentation(
    root: Union[Path, Text],
    p: float
) -> Optional[BaseWaveformTransform]:
    return ApplyImpulseResponse(
        root,
        compensate_for_propagation_delay=True,
        mode="per_example",
        p=p,
    )


def get_augmentation(
    noise_files: Optional[List[Union[Path, Text]]],
    rir_root: Optional[Union[Path, Text]],
    noise_min_snr: float = 1.0,
    noise_max_snr: float = 4.0,
    noise_p: float = 1.0,
    rir_p: float = 0.4,
) -> Optional[BaseWaveformTransform]:
    transforms = []
    if noise_files is not None:
        transforms.append(get_noise_augmentation(
            noise_files,
            min_snr=noise_min_snr,
            max_snr=noise_max_snr,
            p=noise_p,
        ))
    if rir_root is not None:
        transforms.append(get_rir_augmentation(rir_root, p=rir_p))
    if not transforms:
        return None
    return transforms[0] if len(transforms) == 1 else Compose(
        shuffle=False, transforms=transforms
    )
