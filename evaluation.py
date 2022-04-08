from typing import Text

import torch
from pyannote.audio import Inference
from pyannote.audio.core.model import Model
from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
from pyannote.audio.utils.signal import binarize
from pyannote.database import Protocol
from tqdm import tqdm


def get_cder(
    model: Model,
    protocol: Protocol,
    subset: Text,
    batch_size: int,
    device: torch.device
) -> DiscreteDiarizationErrorRate:
    metric = DiscreteDiarizationErrorRate()
    inference = Inference(
        model,
        skip_aggregation=True,
        device=device,
        step=0.5,
        batch_size=batch_size,
    )
    for file in tqdm(list(getattr(protocol, subset)()), desc=f"Evaluating on {subset}"):
        metric(file["annotation"], binarize(inference(file)), uem=file["annotated"])
    return metric
