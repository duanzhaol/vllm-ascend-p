#

python benchmarks/generate_bca_samples.py \
    --num-samples 500 --seed 42 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 1024 \
    --max-model-len 40960 --kv-cache-tokens 1453312 \
    -o benchmarks/configs/bca_workloads.yaml


python benchmarks/plot_bca_samples.py benchmarks/configs/bca_workloads.yaml