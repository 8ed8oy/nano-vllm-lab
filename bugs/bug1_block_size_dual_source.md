# bug1：block 粒度双源 → `store_kvcache` 断言失败

**状态：本仓代码为修复后版本（`llm_engine.py:82` 把 config 的块粒度注入 `Sequence`）**

## 现象（旧版）

`--kvcache-block-size 512` 能通过 `Config.__post_init__` 的参数校验，但写 KV 时
在 `ModelRunner.store_kvcache` 的断言处失败（`slot_mapping.numel() != N`）。

## 根因

同一个「每块多少 token」存在两个来源：

| 位置 | 取值 |
|---|---|
| `nanovllm/engine/sequence.py:34` | `block_size = 256`，类属性硬编码 |
| `nanovllm/engine/model_runner.py:72` | `self.block_size = config.kvcache_block_size` |
| `nanovllm/config.py:124` | 只 `assert kvcache_block_size % 256 == 0`，512 合法 |

`Sequence` 用 256 算 `num_blocks` / `block(i)`，`ModelRunner` 用 512 展开 `slot_mapping`，
同一 token 落到两个不同的物理位置；块表长度与实际需要的块数对不上，写 KV 时断言炸掉。

## 修复

`LLMEngine.__init__` 里、任何 `Sequence` 被构造之前，用 config 覆盖类属性：

```python
Sequence.block_size = config.kvcache_block_size   # nanovllm/engine/llm_engine.py:82
```

代码里的注释写明了理由：`Sequence` 的 `num_blocks` / `last_block_num_tokens` / `block()`
必须与 `ModelRunner` 的 `config.block_size` 完全一致，否则 `slot_mapping` 的数量和 token 数对不上。

## 这个修复的残留问题（可改进，不是 bug）

1. **类级全局状态**：`Sequence.block_size` 是类属性，同一进程里若先后建两个不同 `kvcache_block_size`
   的 engine，后者会改掉前者的粒度。
2. **校验不表达语义**：`assert kvcache_block_size % 256 == 0` 只说明「必须是 256 的倍数」，
   读代码的人看不出这条校验与 `Sequence` 有关联；它拦不住「config 与 Sequence 不相等」这件事。

加固方向（若要提交上游）：把 `block_size` 变成 `Sequence` 的实例字段，由 engine 显式传入，
删掉类级默认值；`Config` 侧把注释改成「与 Sequence 的块粒度同源」。

## 最小复现（回退修复）

删掉 `llm_engine.py:82` 那一行（或把 `Sequence.block_size` 固定回 256），再跑：

```bash
python experiments/29_prefix_cache_ablation.py --arm shared --kvcache-block-size 512
```

## 教训

参数校验通过 ≠ 配置生效。凡是「两处各存一份同一个量」的地方，要么让它只能有一处，
要么在构造时断言两处相等——不要靠注释写「与 Config 一致」。