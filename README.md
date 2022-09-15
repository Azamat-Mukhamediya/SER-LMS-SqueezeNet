## About

![Python 3.7.11](https://img.shields.io/badge/python-3.7.11-green.svg?style=plastic) ![PyTorch 1.9.1](https://img.shields.io/badge/pytorch-1.9.1-green.svg?style=plastic)

The implementation of the paper: "[On the Effect of Log Mel Spectrogram Parameter Tuning for Deep Learning-based Speech Emotion Recognition](https://)".

## Requirements

This code has been developed under `Python3.7.11`, `PyTorch 1.9.1` and `CUDA 11.1` on `Windows 10`.

To install required libraries:

```shell
pip install -r requirements.txt
```

## Datasets in the paper

Datasets can be downloaded or requested from EmoDB(http://emodb.bilderbar.info/), IEMOCAP(https://sail.usc.edu/iemocap/), SAVEE(http://kahlan.eps.surrey.ac.uk/savee/) webpages.

## Usage

```shell
python main.py --data emodb --data_root dataset/EMODB/wav --k 2 --seg_len 16000 --win_len 255 --hop_len 32 --n_mel 128 --batch_size 16 --epochs 1
```

## Citation

In progress
