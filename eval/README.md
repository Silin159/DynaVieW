<div align="center">

# Evaluation Benchmarks

</div>

This directory contains the evaluation pipelines for the benchmarks used in our project. Currently, we support evaluation on VinaBench and LEGO.

## Overview

The evaluation workflow generally consists of three steps:

1. Download the benchmark data.
2. Run model inference to generate predictions.
3. Run the corresponding evaluation scripts to compute metrics.

Please make sure you have correctly installed the project dependencies and configured the required model checkpoints before running the evaluation scripts.

## VinaBench

We evaluate on the VWP portion of VinaBench, which contains visual writing prompts, narratives, and annotations.

### Download Data

Download the VinaBench narratives and annotations from this [link](https://drive.google.com/file/d/1GtKTcQbcHx2RFU8FVcu6JaQxkecOJ3l7/view). After downloading the data, place it under the expected data directory used by the inference script.

### Run Inference

To run inference on VinaBench:
```
cd eval/VinaBench
bash inference.sh
```
The generated predictions will be saved to the output directory specified in inference.sh.

### Run evaluation

After inference is finished, run the evaluation script:
```
cd metrics
bash eval.sh
```
This script computes the evaluation metrics based on the generated predictions and the ground-truth annotations.

## LEGO

We also support evaluation on the [LEGO](https://github.com/BolinLai/LEGO) benchmark.

### Download Data

Download the LEGO dataset using this Hugging Face [link](https://huggingface.co/datasets/bolinlai/LEGO-Dataset). Please make sure the local data structure is consistent with the paths used in the inference script.

### Run Inference

To run inference on LEGO:
```
cd eval/LEGO
bash inference.sh
```

### Run Evaluation

For LEGO, we follow the official evaluation scripts provided by the [LEGO codebase](https://github.com/BolinLai/LEGO). Please follow the instructions in the official repository to compute the benchmark metrics.

## Expected Directory Structure

A typical evaluation directory may look like:

```
eval/
├── VinaBench/
│   ├── data/
│   ├── metrics/
│   │   └── eval.sh
│   ├── inference.sh
│   └── outputs/
│
└── LEGO/
    ├── data/
    ├── inference.sh
    └── outputs/
```

Please adjust the directory structure according to your local setup if necessary.