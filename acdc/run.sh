python main.py --task=knowledge \
--zero-ablation \
--threshold=0.1 \
--device=cuda:3 \
--metric=match_nll \
--indices-mode=reverse \
--first-cache-cpu=False \
--second-cache-cpu=False \
--max-num-epochs=100000
