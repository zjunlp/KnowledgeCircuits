# Knowledge-Circuit
This work aims to build the circuits in the pretrained language models that are responsible for the specific knowledge and analyze the behavior of these components.


## Get the data and prepare the environment
The filtered data for each kind of model is at [here](https://pan.zju.edu.cn/share/7c613d16095c504605f83eba72). Please download it and put it in the data folder.

Build the environement:
```
pip install -r requirements.txt
```
❗️The code may fail under torch 2.x.x. We recommend torch 1.x.x

## Get the circuit
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
MODEL_NAME=gpt2_medium

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

## Analyze component
Run the component.ipynb in notebook.

# Acknowledgement
We thank for the project of [transformer_lens](https://github.com/TransformerLensOrg/TransformerLens), [ACDC](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) and [LRE](https://lre.baulab.info/).
The code in this work is built on top of these three projects' codes.
