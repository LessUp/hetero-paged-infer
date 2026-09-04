# P2 CUDA 流式功能 canary

这是一次**功能验证**，不是吞吐或延迟性能报告。它在干净的 `paged-serving`
`34bf02f8cf1362eea30d85b1308ca4fa9a598f32` 与 `tiny-llm`
`6df2c59b8db5ec409cf214a764e2f4eae84216d8` 上运行；硬件、驱动、CUDA、模型
SHA-256 与每个 run 的完整负载参数见 [`metadata.json`](metadata.json) 和各
`run_metadata.json`。

目的只有两个：

1. 验证 HuggingFace tokenizer 的安全增量解码在真实 CUDA HTTP 服务中会产生多个文本
   SSE chunk，而非等请求结束才发一个 chunk；
2. 验证 Poisson 到达种子同时写入 `summary.json` 与 `run_metadata.json`。

| run | 负载 | 成功 | 可见文本 chunk | 分片间隔样本 | 到达种子 |
|---|---|---:|---:|---:|---:|
| closed | c=1，smoke，1 请求，16 token 上限 | 1/1 | 16 | 15 | — |
| poisson | λ=50 req/s，smoke，1 请求，16 token 上限 | 1/1 | 16 | 15 | 20260904 |

`per_request.jsonl` 是上述 chunk 数和分片间隔的原始记录；`summary.json` 是 loadgen
权威汇总。两组均为 `warmup=0`、`n=1`，故 TTFT、TPOT、tok/s 与 req/s **不得**作为
性能数字引用，也不能同 P1 的 21-run 正式矩阵比较。P1 历史基线在相邻的
[`2026-09-04-RTX3060Laptop-paged-serving/`](../2026-09-04-RTX3060Laptop-paged-serving/)。

复现时先从仓库根目录启动真实后端：

```bash
TINY_LLM_DIR=../tiny-llm/build cargo build --locked --release --features tiny-llm
PAGED_SERVING_TINY_LLM_MAX_SEQS=8 PAGED_SERVING_TINY_LLM_DECODE_RESERVE=128 \
  ./target/release/paged-serving --serve --backend tiny-llm \
  --model-path ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokenizer ../models/tokenizer.json --max-num-seqs 8 --max-batch-size 8 --port 3003
```

再在 `benchmarks/serving/` 运行：

```bash
./run_sweep.sh --base-url http://127.0.0.1:3003 --engine paged-serving \
  --modes 'closed poisson' --concurrencies '1' --rates '50' --datasets smoke \
  --requests 1 --warmup-secs 0 --max-tokens 16 --repeats 1 \
  --model paged-serving --model-path ../../../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --backend-quant Q4_K_M --tokenizer ../../../models/tokenizer.json --cuda-archs 86 \
  --poisson-seed 20260904 --results-dir results/2026-09-04-RTX3060Laptop-paged-serving-p2-stream-canary
```
