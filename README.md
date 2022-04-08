# Continual Self-supervised Domain Adaptation for End-to-end Speaker Diarization

This is the companion repository for the paper *[Continual Self-supervised Domain Adaptation for End-to-end Speaker Diarization](/paper.pdf)* by by [Juan Manuel Coria](https://juanmc2005.github.io/), [Hervé Bredin](https://herve.niderb.fr), [Sahar Ghannay](https://saharghannay.github.io/) and [Sophie Rosset](https://perso.limsi.fr/rosset/).

> In conventional domain adaptation for speaker diarization, a large collection of annotated conversations from the target domain is required. In this work, we propose a novel continual training scheme for domain adaptation of an end-to-end speaker diarization system, which processes one conversation at a time and benefits from full self-supervision thanks to pseudo-labels. The qualities of our method allow for autonomous adaptation (e.g. of a voice assistant to a new household), while also avoiding permanent storage of possibly sensitive user conversations. We experiment extensively on the 11 domains of the DIHARD III corpus and show the effectiveness of our approach with respect to a pre-trained baseline, achieving a relative 17% performance improvement. We also find that data augmentation and a well-defined target domain are key factors to avoid divergence and to benefit from transfer.

## Install

1) Create environment:

```shell
conda create -n csda python==3.8
conda activate csda
```

2) Install the latest PyTorch version following the [official instructions](https://pytorch.org/get-started/locally/#start-locally)

4) Install dependencies:
```shell
pip install -r requirements.txt
```

## Prepare your data

- Prepare your data-loading protocol using [pyannote.database](https://github.com/pyannote/pyannote-database). For continual adaptation experiments, the files in the training subset will form the training sequence in the order they appear.
- Download noise and impulse response datasets.

## Run an experiment

- Use `run_ours.py` to reproduce the results from our training scheme
- Use `run_whole.py` to reproduce the results from non-continual systems

By default the initial model will be the one we pre-trained on AMI with true-label supervision. You can replace it using the `--model` argument.
For more information on script arguments see `run_ours.py -h` or `run_whole.py -h`

## Citation

```bibtex
Paper under review. 
```

## License

```
MIT License

Copyright (c) 2021 Université Paris-Saclay
Copyright (c) 2021 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
