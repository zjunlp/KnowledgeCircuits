<h1 align="center"> Knowledge Circuits </h1>
<h3 align="center"> Knowledge Circuits in Pretrained Transformers </h3>

<p align="center">
  <a href="https://arxiv.org/abs/2405.17969">📄arXiv</a> •
  <a href="https://www.youtube.com/watch?v=qDgCLeDs4Kg"> <img src="https://img.icons8.com/?size=100&id=19318&format=png&color=000000" height="20" width="20"> Youtube</a> • 
    <a href="https://x.com/zxlzr/status/1797261767674138924">𝕏 Blog</a>
</p>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/KnowledgeCircuits) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zjunlp/KnowledgeCircuits?color=green) 



## Table of Contents
- 🌟[Overview](#overview)
- 🔧[Installation](#installation)
- 📚[Get the circuit](#get-the-circuit)
- 🧐[Analyze Component](#analyze-component)
- 🌻[Acknowledgement](#acknowledgement)
- 🚩[Citation](#citation)

---


## 🌟Overview

This work aims to build the circuits in the pretrained language models that are responsible for the specific knowledge and analyze the behavior of these components.


## 🔧Installation

The filtered data for each kind of model is at [here](https://pan.zju.edu.cn/share/7c613d16095c504605f83eba72). Please download it and put it in the data folder.

Build the environement:
```
conda create -n knowledgecircuit python=3.10
pip install -r requirements.txt
```
❗️The code may fail under torch 2.x.x. We recommend torch 1.x.x

## 📚Get the circuit

Just run the following commond:
```
cd acdc
sh run.sh
```
Here is an example to run the circuit for the `country_capital_city` in `GPT2-Medium`.
```
MODEL_PATH=/path/to/the/model
KT=factual 
KNOWLEDGE=country_capital_city
NUM_EXAMPLES=20
MODEL_NAME=gpt2-medium

python main.py --task=knowledge \
--zero-ablation \
--threshold=0.01 \
--device=cuda:0 \
--metric=match_nll \
--indices-mode=reverse \
--first-cache-cpu=False \
--second-cache-cpu=False \
--max-num-epochs=10000 \
--specific-knowledge=$KNOWLEDGE \
--num-examples=$NUM_EXAMPLES \
--relation-reverse=False \
--knowledge-type=$KT \
--model-name=$MODEL_NAME \
--model-path=$MODEL_PATH
```

You would get the results in `acdc/factual_results/gpt2-medium`.

## 🧐Analyze component

Run the component.ipynb in notebook.

## 🌻Acknowledgement

We thank for the project of [transformer_lens](https://github.com/TransformerLensOrg/TransformerLens), [ACDC](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) and [LRE](https://lre.baulab.info/).
The code in this work is built on top of these three projects' codes.


## 🚩Citation

Please cite our repository if you use Knowledge Circuit in your work. Thanks!

```bibtex
@article{DBLP:journals/corr/abs-2405-17969,
  author       = {Yunzhi Yao and
                  Ningyu Zhang and
                  Zekun Xi and
                  Mengru Wang and
                  Ziwen Xu and
                  Shumin Deng and
                  Huajun Chen},
  title        = {Knowledge Circuits in Pretrained Transformers},
  journal      = {CoRR},
  volume       = {abs/2405.17969},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2405.17969},
  doi          = {10.48550/ARXIV.2405.17969},
  eprinttype    = {arXiv},
  eprint       = {2405.17969},
  timestamp    = {Fri, 21 Jun 2024 22:39:09 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2405-17969.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
