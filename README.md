# nano-vllm-lab

> 基于教学版 nano-vllm（[CZYCW/learn-nano-vllm](https://github.com/CZYCW/learn-nano-vllm)，源自 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)）的复现与对照实验。
> **我是使用者，不是引擎作者。** 本仓只包含我加的注释、实验脚本与采集到的原始数据。

在单张 RTX 3060 上把「KV Cache → 前缀缓存 → 指标口径」这条链路跑通并留证：
三组对照实验、两处缺陷定位、一份请求生命周期笔记。

## 结果

条件：Qwen3-0.6B / 8 并发 / 每 arm 重复提交 2 轮 / prompt 1088 token（前缀 1024 + 后缀 64）/
300 blocks / block 256 / RTX 3060。原始数据见 `results/04_s4_prefix_cache.jsonl`，环境见 `results/env.json`。

| arm | 前缀 | 轮次 | cached tokens | 真实 prefill 计算量 | 引擎报告 prefill | TTFT p50 |
|---|---|---|---|---|---|---|
| shared | 1024，8 条共享 | r1 冷启动 | 7168 | 1536 | 8704 | 84.2 ms |
| shared | 1024，8 条共享 | r2 重复提交 | 8192 | 512 | 8704 | **32.3 ms** |
| short | 128，共享 | r1 / r2 | 0 / 0 | 1536 / 1536 | 1536 / 1536 | 67.8 / 67.5 ms |
| random | 1024，各条不同 | r1 冷启动 | 0 | 8704 | 8704 | 401.5 ms |
| random | 1024，各条不同 | r2 重复提交 | 8192 | 512 | 8704 | 34.9 ms |

读这张表的三条：

1. **共享 1024 token 前缀、重复提交**：TTFT 84.2 → 32.3 ms（2.6×），真实 prefill 计算量 8704 → 1536（5.67×）。
2. **`short` 才是有意义的「无命中」对照**：前缀 128 < block 256，凑不满一个整块，两轮 cached 恒为 0、TTFT 几乎不动（67.8 / 67.5 ms）——说明只有整块对齐的前缀才进缓存。
3. **`random` 第 2 轮也命中了 8192 token**：该 arm 两轮用的是同一组（同种子生成的）随机前缀，所以它**不是**「无共享」对照，只有 r1 的 401.5 ms 可用。原本我以为「随机前缀组应无变化」，数据否掉了这个预期，此处按数据写。

## 口径陷阱（本仓最值得看的一条）

引擎侧 `num_tokens` 不含 `num_cached_tokens`：

- shared 组与 random 组的**引擎报告值都是 8704**；
- 只看报告值会得出「前缀缓存没生效」的相反结论；
- 按真实计算量（`computed_prefill_tokens`）两者相差 **5.67 倍**，TTFT 相差 4.8 倍（84.2 vs 401.5 ms 冷启动）。

所以「缓存有没有用」这个问题，在 nano-vllm 里不能用 `prefill_tokens` 回答，只能用
`sum(prompt - num_cached_tokens)` + 命中计数回答。

## 复现

```bash
git clone https://github.com/CZYCW/learn-nano-vllm.git && cd learn-nano-vllm
pip install -e .
export NANOVLLM_MODEL=~/huggingface/Qwen3-0.6B      # 需单卡 ≈12GB

# 三组对照（shared / short / random），每组单独起进程避免显存没释放导致 OOM
bash /path/to/nano-vllm-lab/scripts/repro.sh
```

`scripts/repro.sh` 只是把上面这条命令跑三遍并落盘，真正的实验逻辑在
`experiments/29_prefix_cache_ablation.py`（含每个 arm 的预期与实现说明）。

## 缺陷（`bugs/`）

| 文件 | 现象 | 一句话结论 | 本仓代码状态 |
|---|---|---|---|
| [bug1](bugs/bug1_block_size_dual_source.md) | `--kvcache-block-size 512` 能过参数校验，写 KV 时断言失败 | `Sequence.block_size` 硬编码 256，与 `ModelRunner` 用的 `config.kvcache_block_size` 双源 | 已修复（`llm_engine.py:82` 注入），残留类级全局状态 |
| [bug2](bugs/bug2_can_append_may_append.md) | KV 块紧张 + 并发超容量时 decode 必现 `IndexError` | `can_append`（`% block == 0`）与 `may_append`（`== 1`）谓词不一致，长度 `k*block+1` 时误判不需要新块 | 已修复 |

两处都出在教学仓旧版本，不是我的引擎实现。修复方式：bug1 在 `LLMEngine.__init__` 里用 config 覆盖 `Sequence.block_size`（早于任何 `Sequence` 构造）；bug2 把 `can_append` 的判据对齐 `may_append` 的 `== 1`。

## 边界

- 单机单卡、单模型（Qwen3-0.6B），**不外推**到其它 GPU / 模型；
- 每 arm 只跑 2 轮、每轮 1 次，**不是**收敛后的稳定值；TTFT 取 p50，n=8；
- 只测 prefill 侧（`--output-len 8`），不涉及 decode 吞吐与并发扩展性；
- `dtype` / CUDA 版本 / 上游 commit 当时没写进结果文件，`env.json` 里留空**而不是猜**；
- 数据采集于 2026-10，采集后机器驱动异常（`nvidia-smi` 无法通信），本轮未重测。

## 结构

```
experiments/29_prefix_cache_ablation.py   三组对照（shared / short / random），沿用上游仓逻辑未改
scripts/repro.sh                          复现入口
results/04_s4_prefix_cache.jsonl          原始数据（每 arm 2 轮，含 events 计数）
results/env.json                          环境快照
bugs/                                     两处缺陷：现象 → 定位 → 结论
notes/request-lifecycle.md                请求生命周期与 KV block 三态
```

许可证沿用上游 MIT（`LICENSE`）。本仓的价值在**数据与口径**，不在代码。