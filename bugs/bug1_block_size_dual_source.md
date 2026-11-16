# bug1：block 粒度双源 → `store_kvcache` 断言失败

**状态：本仓代码仍可复现（2026-10 观察；此后机器驱动异常，未重测）**

## 现象

`--kvcache-block-size 512` 能通过 `Config.__post_init__` 的参数校验，但写 KV 时
在 `ModelRunner.store_kvcache` 的断言处失败。

## 定位

两个「每块多少 token」的来源同时存在：

| 位置 | 取值 |
|---|---|
| `nanovllm/engine/sequence.py:34` | `block_size = 256` 类属性硬编码 |
| `nanovllm/engine/model_runner.py:72` | `self.block_size = config.kvcache_block_size` |
| `nanovllm/config.py:124` | 只 `assert kvcache_block_size % 256 == 0`，512 合法 |

于是 `Sequence.num_blocks`（用 256）与 `slot_mapping`（用 512）对同一 token 算出不同的物理位置：
前者把 1088 token 摊成 5 个 block，后者按 512 步长写，`block_table` 长度和实际需要的 block 数对不上。

## 最小复现

```bash
python experiments/29_prefix_cache_ablation.py --arm shared --kvcache-block-size 512
```

`--kvcache-block-size` 由 `main()` 透传给 `LLM(...)`，不碰 `Sequence.block_size`。

## 结论

单一真相源缺失。`Sequence` 是数据类，把 `block_size` 当常量写死，而运行时配置另存一份；
参数校验只覆盖了「形态」（`% 256 == 0`），没覆盖「语义一致」。

## 教训

参数校验通过 ≠ 配置生效。凡是「两处各存一份同一个量」的地方，
要么让它只能有一处，要么在构造时断言两处相等——不要靠注释写「与 Config 一致」。