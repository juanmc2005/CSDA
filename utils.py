from typing import Dict, Text, Tuple

from pyannote.audio import Audio
from pyannote.core import Timeline, Segment


def format_float_range(float_range: Text) -> Tuple[float, float]:
    float_range = [float(value) for value in float_range.split("-")]
    assert len(float_range) == 2 and float_range[0] <= float_range[1]
    return tuple(float_range)


class FileSplitter:
    def __init__(
        self,
        sample_rate: int = 16000,
        train_chunk_duration: float = 40.0,
        dev_chunk_duration: float = 20.0,
    ):
        super().__init__()
        self.train_duration = train_chunk_duration
        self.dev_duration = dev_chunk_duration
        self.audio = Audio(sample_rate=sample_rate, mono=True)

    @staticmethod
    def crop(file: Dict, subset: Text, support: Timeline) -> Dict:
        new_file = {
            "uri": f"{file['uri']}.{subset}",
            "subset": subset,
            "annotated": support,
        }
        for key in file.keys():
            if key not in new_file:
                new_file[key] = file[key]
        return new_file

    def split(self, file: Dict) -> Tuple[Dict, Dict]:
        total_duration = self.audio.get_duration(file["audio"])
        train_support, dev_support = [], []
        start_time = 0
        while start_time < total_duration:
            train_end = start_time + self.train_duration
            if train_end > total_duration:
                train_end = total_duration
            train_support.append(Segment(start_time, train_end))
            if train_end < total_duration:
                dev_end = train_end + self.dev_duration
                if dev_end > total_duration:
                    dev_end = total_duration
                dev_support.append(Segment(train_end, dev_end))
            start_time += self.train_duration + self.dev_duration
        train_file = self.crop(file, "train", Timeline(train_support))
        dev_file = self.crop(file, "development", Timeline(dev_support))
        return train_file, dev_file
