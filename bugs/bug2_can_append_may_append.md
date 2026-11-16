# bug2：`can_append` / `may_append` 谓词不一致 → decode `IndexError`

**状态：本仓代码为修复后版本，保留定位记录；回退一行即可复现**

## 现象

KV 块紧张（`--num-blocks` 压到刚好不够）且并发数超过容量时，decode 阶段必现 `IndexError`：
`free_block_ids` 已空，仍去取下标。

## 定位

同一个边界——「什么时候需要新 block」——在两处各写了一遍，且差一：

| 函数 | 旧判据 | 含义 |
|---|---|---|
| `can_append` | `len(seq) % block_size == 0` | 认为「块刚写满」时才需要新块 |
| `may_append` | `len(seq) % block_size == 1` | 认为「刚跨进新块第一格」时才分配新块 |

序列长度正好是 `k * block_size + 1` 时，`can_append` 返回 True（不需要新块），
`may_append` 却真的去 `free_block_ids[0]`，此时列表为空 → `IndexError`。

`can_append` 的结果直接决定 `Scheduler` 是否 `preempt` 别的序列腾块，所以它不是「少分配一次」的小事：
判错等于把预占位让给了别人。

## 修复

把 `can_append` 的判据对齐 `may_append`，并在注释里写明两处必须同源：

```python
need_new_block = 1 if (len(seq) % self.block_size == 1) else 0
```

修复版见 `nanovllm/engine/block_manager.py:283`（注释解释了为什么是 `== 1` 而不是 `== 0`）。

## 最小复现

把 `can_append` 改回 `% block_size == 0`，然后把 `--num-kvcache-blocks` 压到并发序列刚好差一块，
在 decode 阶段触发 `preempt` 路径即可。

## 教训

边界条件必须能被**一处**单测覆盖。同一语义写两遍，漂移是迟早的事——
而这类漂移不会在 happy path 上暴露，只在「资源刚好不够」时才炸。