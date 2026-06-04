<div align="center">

# DynaVieW: Schema-Guided World Modeling for Understanding Hierarchical Visual Dynamics

</div>

<div align="center">
<b><a href="https://silin159.github.io/SilinGao/" target="_blank">Silin Gao</a><sup>1*</sup>, <a href="https://marcelluszhao.github.io/" target="_blank">Hao Zhao</a><sup>1*</sup>, <a href="https://eric11eca.github.io/" target="_blank">Zeming Chen</a><sup>1</sup>, <a href="https://smamooler.github.io/" target="_blank">Sepideh Mamooler</a><sup>1</sup>, <a href="https://www.languagesciences.cam.ac.uk/staff/antara-raaghavi-bhattacharya" target="_blank">Antara Raaghavi Bhattacharya</a><sup>1,2</sup>, <a href="https://qiyuw.github.io/" target="_blank">Qiyu Wu</a><sup>3</sup>, <a href="https://www.linkedin.com/in/hiromi-wakaki-570067286/?originalSubdomain=jp" target="_blank">Hiromi Wakaki</a><sup>3</sup>, <a href="https://www.yukimitsufuji.com/" target="_blank">Yuki Mitsufuji</a><sup>3</sup>, <a href="https://limirs.github.io/" target="_blank">Li Mi</a><sup>1,4</sup>, <a href="https://smontariol.github.io/" target="_blank">Syrielle Montariol</a><sup>1</sup>, <a href="https://atcbosselut.github.io/" target="_blank">Antoine Bosselut</a><sup>1</sup></b>

<sup>*</sup>Equal Contribution &nbsp; <sup>1</sup>EPFL &nbsp; <sup>2</sup>Harvard University &nbsp; <sup>3</sup>Sony &nbsp; <sup>4</sup>ETH Zurich

[![ArXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-B31B1B.svg?logo=arxiv&logoColor=white)](https://github.com/Silin159/DynaVieW/blob/main/DynaVieW_Camera_Ready.pdf)
</div>

## Abstract

Multimodal LLMs lack a systematic understanding of visual dynamics in complex human world activities, which requires the model to predict or simulate multiple levels of dynamic constituents, such as the general progression of actions and the associated changes of low-level details in the world. To address this challenge, we propose a dynamic visual schema-guided world model, <b>DynaVieW</b>, optimized for visual dynamic prediction and simulation. DynaVieW achieves an in-depth understanding of visual dynamics by learning <b>interleaved state-transition</b> sequences, where states cover broad visual scenes from video keyframes, and transitions capture comprehensive dynamic constituents within a <b>hierarchical schema</b>. DynaVieW jointly models transition prediction and state simulation under a mixture-of-experts architecture, with a <b>cross-expert selective attention</b> and a <b>schema token re-weighted loss</b>, to ensure effective and robust learning. DynaVieW's superior visual dynamic understanding boosts its downstream performances on both visual narrative creation and world simulation, showing improved consistency and controllability of visual generation and better instruction-following ability.

## Overview of DynaVieW

<div align="center">
<img src="figs/DynaVieW_Overview.png" width="70%" alt="overview"/>
</div>

We elevate the world modeling capabilities of interleaved vision-text LLMs via continued pre-training on more systematic human world visual dynamics.

Our proposed <b>Dyna</b>mic-<b>Vi</b>sion-ori<b>e</b>nted <b>W</b>orld model (<b>DynaVieW</b>) is pre-trained on interleaved state-transition sequences across broad domains.

Specifically, DynaVieW learns to simulate <b>visual state sequences</b> that are sourced from <b>keyframes</b> of diverse real-world videos, covering various human daily activities, robotic manipulations, art works and auto-driving recordings, etc. Meanwhile, DynaVieW learns to predict <b>transitions</b> between visual states, which are texts framed in a <b>hierarchical JSON schema</b>, to comprehensively capture both high-level progression of activities and low-level changes of visual details in a structured manner.

## Setting Up Environment

#### Docker
Please refer to `Dockerfile` (along with `requirements.txt`) for building the environment (Docker image) from scratch.
At the top of the `Dockerfile`, remember to choose the correct processor architecture that fits your platform:
```
# for arm64 or aarch64 processor architecture
FROM --platform=linux/arm64 nvidia/cuda:12.4.0-devel-ubuntu22.04

# for amd64 or x86_64 processor architecture
FROM --platform=linux/amd64 nvidia/cuda:12.4.0-devel-ubuntu22.04
```

#### Pip
To build the environment packages on an existing OS with pip:
```
# Python Version: 3.10, CUDA Version: 12
pip install --upgrade pip setuptools packaging wheel ninja pybind11
pip install torch==2.7.0 triton==3.3.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -v --no-cache-dir --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@v0.0.30
pip install -v --no-cache-dir --no-build-isolation -U git+https://github.com/flashinfer-ai/flashinfer.git@v0.2.7.post1

git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
git checkout v0.9.2
pip install -v --no-cache-dir --no-build-isolation -r requirements/build.txt
pip install -v --no-cache-dir --no-build-isolation -e .
cd ..

pip install -r requirements.txt
pip install --no-binary opencv-contrib-python --no-deps opencv-contrib-python
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## DynaVieW State-Transition Data Construction

Please refer to `data/README.md` for the construction of DynaVieW's pre-training data.

## DynaVieW Modeling and Training

Please refer to `modeling/README.md` for the pre-training of DynaVieW and fine-tuning on downstream tasks.

## DynaVieW Inference and Evaluation

[TO DO]
