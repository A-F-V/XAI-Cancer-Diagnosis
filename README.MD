# Towards Explainable AI for Cancer Diagnosis

## Introduction

**This project explores modern deep learning methods for diagnosising cancer**. In particular, the goal is to detect the presence of cancer from an H&E staining. My steps are as follows:

1. **Stain Normalization**: Implement an algorithm (**Macenko Algorithm**) for normalizing H&E stains.
2. **Nuclear Panoptic Segmentation**: Implement a deep learning model (**HoVerNet**) for identifying
   and classifying nuclei.
3. **Cell Graph Construction**: Define and create a [cell graph](https://cacm.acm.org/magazines/2017/1/211111-cell-graphs/fulltext).
4. **Cell Graph Classification**: Implement a graph neural network to classify the presence
   of cancer in graph from (3).
5. **Explainability**: Implement a post-hoc explainability algorithm (**GCExplainer**) to
   make the model in (4) more interpretable.

![Overview](./md_images/Full_Overview.png)

## Usage

This repository includes a CLI tool. Documentation for using it can be found below. Note you must have the concepts and trained models on the local machine, which are not provided in this repository due to size constraints:

```sh
python xcd.py FILE_LOCATION [ARGUMENTS]
```

| ARGUMENT              | Description                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| `--explanation_file`  | The location for explanations to be saved to. If none is provided, no explanations will be generated |
| `--hovernet_path`     | The location of the trained HoVerNet model                                                           |
| `--cell_encoder_path` | The location of the trained ResNet50 encoder model                                                   |
| `--gnn_path`          | The location of the trained GNN model                                                                |
| `--concept_path`      | The location of the concepts folder, containing the exemplary concept representation                 |

## Worked Example

![Overview](./md_images/example_pipeline.png)

<!--
## Bibliography

1. [HoVerNet](http://arxiv.org/abs/1812.06499)
   > Graham, Simon, et al. "Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." Medical Image Analysis 58 (2019): 101563.
2. [Macenko Algorithm](http://ieeexplore.ieee.org/document/5193250/)
   > Macenko, Marc, et al. "A method for normalizing histology slides for quantitative analysis." 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.
3. [PanNuke Dataset](http://arxiv.org/abs/2003.10778)
   > Gamper, Jevgenij, et al. "Pannuke dataset extension, insights and baselines." arXiv preprint arXiv:2003.10778 (2020).
4. [MoNuSeg Dataset](https://ieeexplore.ieee.org/document/7872382)
   > Kumar, Neeraj, et al. "A dataset and a technique for generalized nuclear segmentation for computational pathology." IEEE transactions on medical imaging 36.7 (2017): 1550-1560.
5. [BACH Dataset]
6. [Histographs]
7. [ResNet]
8. [FCN]
9. GINTopK
10. GCExplainer
--->
