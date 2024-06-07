MODEL_PATH=/path/to/your/model
KT=factual
KNOWLEDGE=country_capital_city
NUM_EXAMPLES=1
MODEL_NAME=gpt2-medium

python main.py --task=knowledge \
--zero-ablation \
--threshold=0.01 \
--device=cuda:0 \
--metric=match_nll \
--indices-mode=reverse \
--first-cache-cpu=False \
--second-cache-cpu=False \
--max-num-epochs=100000 \
--specific-knowledge=$KNOWLEDGE \
--num-examples=$NUM_EXAMPLES \
--relation-reverse=False \
--knowledge-type=$KT \
--model-name=$MODEL_NAME \
--model-path=$MODEL_PATH
