from typing import Dict, Iterator, List

from pyannote.database.protocol import SpeakerDiarizationProtocol


class SplitFileProtocol(SpeakerDiarizationProtocol):
    def __init__(self, train: List[Dict] = (), development: List[Dict] = ()):
        super().__init__()
        self._both_non_empty = len(train) > 0 and len(development) > 0
        both_empty = len(train) == 0 and len(development) == 0
        assert both_empty or self._both_non_empty
        self._train_files, self._dev_files = list(train), list(development)

    def update(self, train_file: Dict, dev_file: Dict):
        self._train_files = [train_file]
        self._dev_files = [dev_file]

    def train_iter(self) -> Iterator[Dict]:
        for file in self._train_files:
            yield file

    def development_iter(self) -> Iterator[Dict]:
        for file in self._dev_files:
            yield file
