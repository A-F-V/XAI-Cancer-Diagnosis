<!--[![CircleCI](https://circleci.com/gh/A-F-V/XAI-Cancer-Diagnosis/tree/master.svg?style=svg&circle-token=7fb0c2bb52a5fffae0961cedf3dce014293ceef6)](https://circleci.com/gh/A-F-V/XAI-Cancer-Diagnosis/tree/master)-->

[![Documentation Status](https://readthedocs.org/projects/xai-cancer-diagnosis/badge/?version=latest)](https://xai-cancer-diagnosis.readthedocs.io/en/latest/?badge=latest)

# Towards Explainable AI for Cancer Diagnosis

## Introduction

**This project explores modern deep learning methods for diagnosising cancer**. In particular, the goal is to detect the presence of cancer from a H&E staining. This may be accomplished by **1)** taking an H&E image and producing a [cell graph](https://cacm.acm.org/magazines/2017/1/211111-cell-graphs/fulltext), then **2)** using a **graph neural network** to perform the detection.

In addition, it is essential for clinical models to be explainable, otherwise they will not be used. Therefore, I add a third step: **3)** Use a post-hoc GNN explanation model (like **GCExplainer**) in order to deliver more meaningful findings to clinicians.

## Usage

This repository includes a CLI tool. Documentation for using it can be found below:

```sh
python xcd.py [OPTIONS] ACTION
```

| Action  | Description                                                                                       |
| ------- | ------------------------------------------------------------------------------------------------- |
| `setup` | Downloads, extracts and processes required datasets for training. **Requires the GCP bucket key** |
| `train` | Trains model passed in with `-m`                                                                  |

| Models      | Description                                                           |
| ----------- | --------------------------------------------------------------------- |
| `hover_net` | [HoVerNet](#bibliography) - used for **nuclei instance segmentation** |

## To Cite

## Bibliography
