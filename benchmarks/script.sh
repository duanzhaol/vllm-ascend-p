# 生成 BCA 基准测试样本
python benchmarks/generate_bca_samples.py \
    --num-samples 200 --seed 42 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 1024 \
    --max-model-len 32768 --kv-cache-tokens 1453312 --max-concurrent-batches 4 \
    -o benchmarks/configs/qwen-pp4.yaml

python benchmarks/generate_bca_samples.py \
    --num-samples 200 --seed 42 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 1024 \
    --max-model-len 32768 --kv-cache-tokens 1351577 --max-concurrent-batches 1 \
    -o benchmarks/configs/qwen-tp4.yaml
1
# 可视化 BCA 基准测试样本
python benchmarks/plot_bca_samples.py benchmarks/configs/qwen-tp4.yaml

# 启动 vLLM 服务器
VLLM_SERVER_DEV_MODE=1 VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random VLLM_LOGGING_LEVEL=TRACE vllm serve  /tmp/models/Qwen3-8B/     --served-model-name qwen     --pipeline-parallel-size 4    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 48}' --enforce-eager --max-num-seqs 1024

# 运行 BCA 基准测试并保存结果
python benchmarks/run_benchmark_profile.py --config benchmarks/configs/bca_workloads.yaml --output-file results.csv

python benchmarks/run_benchmark_profile.py --config benchmarks/configs/qwen-pp4.yaml --output-file benchmarks/results/qwen-pp4-2.csv


#训练性能模型
python -m simulator.perf_model.training --csv benchmarks/results/qwen-tp4-300-lsh.csv --model-name qwen