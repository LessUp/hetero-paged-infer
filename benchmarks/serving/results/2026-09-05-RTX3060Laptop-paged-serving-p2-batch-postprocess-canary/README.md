# P2 批量末端后处理功能 canary

这是一次**功能验证**，不是吞吐或延迟性能报告。它绑定干净的
`paged-serving` `2e4e2c329966a152cd0903c3b2c30e832b43e94e` 与 `tiny-llm`
`e3dc13bdec6edc69c437a48fb6ded6b5c58e757e`；硬件、驱动、CUDA、模型 SHA-256 与
负载参数见 [`metadata.json`](metadata.json) 和
[`paged-serving_closed_c4_smoke_r1/run_metadata.json`](paged-serving_closed_c4_smoke_r1/run_metadata.json)。

此次 canary 只验证当前真实 CUDA HTTP 服务在 closed c=4 的小样本下能够完成请求：

| run | 负载 | 成功 | 输出 token 计数覆盖 | 结论范围 |
|---|---|---:|---:|---|
| closed | c=4，smoke，4 请求，16 token 上限，warmup=0，repeat=1 | 4/4 | 4/4（64 token，来自 usage） | 当前二进制可完成真实 HTTP/SSE 请求 |

正常 greedy 路径的代码事实是：每个序列的 layer forward 完成后，末层 hidden 先写入 GPU
batch buffer；循环结束后批量执行 final RMSNorm、LM head 与 argmax，再一次回传 token id。
本 canary 证明该二进制能经 HTTP 走通，不追踪每个 `tinyllm_step` 的实际 batch 形状，因而
不把 closed c=4 写成“每一步都是 4 路 fused compute”。Transformer layer forward 仍逐序列。

`per_request.jsonl` 是原始请求记录，`summary.json` 是 loadgen 权威汇总。此处 `n=1`、
`warmup=0`、发压机与服务同机；其中的 TTFT、TPOT、chunk 间隔、tok/s 与 req/s 都**不得**
作为性能数字引用，也不得与 2026-09-04 的 21-run P2 矩阵比较。该矩阵早于本次代码改动；
若要判断性能，必须在当前干净提交上重新采集完整重复矩阵。

复现时先从仓库根目录构建并启动真实后端：

```bash
TINY_LLM_DIR=../tiny-llm/build cargo build --locked --release --features tiny-llm --bin paged-serving
PAGED_SERVING_TINY_LLM_MAX_SEQS=4 \
  ./target/release/paged-serving --serve --backend tiny-llm \
  --model-path ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokenizer ../models/tokenizer.json --host 127.0.0.1 --port 3010
```

再在 `benchmarks/serving/` 运行：

```bash
./run_sweep.sh --base-url http://127.0.0.1:3010 --engine paged-serving \
  --model paged-serving --modes closed --concurrencies '4' --datasets smoke \
  --requests 4 --warmup-secs 0 --max-tokens 16 --repeats 1 \
  --model-path ../../../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --backend-quant W8A16 --tokenizer ../../../models/tokenizer.json --cuda-archs 86 \
  --results-dir results/2026-09-05-RTX3060Laptop-paged-serving-p2-batch-postprocess-canary \
  --tiny-llm-dir ../../../tiny-llm

python3 validate_results.py results/2026-09-05-RTX3060Laptop-paged-serving-p2-batch-postprocess-canary
```
