# 请求生命周期与 KV block 三态

## 一条请求走过的路径

```
prompt ──► LLMEngine.add_request ──► Scheduler.waiting
                                        │  按 token budget / 显存够不够
                                        ▼
                                  Scheduler.running ──► ModelRunner.run
                                        │                 ├─ prefill（首轮，整段 prompt）
                                        │                 └─ decode（逐 token）
                                        ▼
                                   Sampler ──► 流式返回 ──► finished
                                        │
                                        ▼
                             BlockManager.deallocate ──► 回 free list（惰性，不清 hash）
```

## KV block 三态（对应 `block_manager` 的三个 TEACHING 事件）

三个事件都在 `allocate()` 的前缀链循环里发出（`block_manager.py:203/218/230`）：

| 事件 | 判据 | 含义 |
|---|---|---|
| `KV_BLOCK_ALLOCATE` | 链式 hash 未命中（或 hash 碰撞，`token_ids` 不等） | 从 free list 取新块；一旦 miss，**后续块全部走新块** |
| `KV_BLOCK_REUSE` | 命中，且该块仍在 `used_block_ids` 里 | 别的序列还在用，`ref_count++` 直接共享 |
| `KV_BLOCK_REACTIVATE` | 命中，但该块在 free list 里、hash 还在 | 惰性回收的收益：KV 没被覆盖，直接重新激活 |

实测（`results/04_s4_prefix_cache.jsonl`，shared arm，8 并发，每条 prompt 1088 token = 4 整块 + 64 token 尾巴）：

- r1 冷启动：`ALLOCATE=8 / REUSE=28 / REACTIVATE=4` —— 1 条冷序列占 5 块，另 7 条各复用 4 个整块、尾巴各分配 1 块；
- r2 重复提交：`ALLOCATE` 仍是 8，`REUSE` 升到 32 —— 8 条全部整链命中，没有块需要新写。

注意「未满块不参与 cache」：`len(token_ids) != block_size` 时 hash 记 -1。这就是 `short`（前缀 128）恒 0 命中的原因。

## 惰性回收为什么是对的

`deallocate()` 只把块放回 free list，**不清** `hash_to_block_id`。
若之后有请求命中同一前缀，而这个块的 KV 还没被新数据覆盖，就能直接 `REACTIVATE`，省掉一次 prefill。
旧 hash 在块被重新分配、写入新内容时自然失效（`reset()` 把 hash 置 -1）。

代价是 `hash_to_block_id` 里会留一批指向空闲块的条目——这也是 block 数紧张时
`can_append` / `may_append` 的边界更容易被触发的原因（见 `bugs/bug2_can_append_may_append.md`）。

## 待补

- 一次带时间戳的 trace：标出 prefill / decode / block 生命周期在时间轴上的位置；
- 抢占（`preempt`）路径下三态如何变化，目前只看了空闲块耗尽时的 `IndexError`。