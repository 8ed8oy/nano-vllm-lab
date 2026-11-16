"""
S4 对照实验：Prefix Caching 到底省了什么（以及为什么用错指标会看不见它）

【为什么需要它】
S3 的 harness（28_s3_param_ablation.py）统计的 `prefill_tokens` 来自
`llm_engine.step()` 的 `num_tokens = sum(len(seq))`，它 **没有扣掉 num_cached_tokens**。
于是"发两遍完全相同的 prompt"会得到一模一样的 prefill_tokens —— 看起来 prefix cache
根本没生效。本脚本同时打印两个口径，把这个陷阱直接摊开：

    engine_reported  = Σ prompt token（引擎口径，缓存命中不体现在这里）
    computed         = Σ (prompt token - num_cached_tokens)  ← 真正要算的 prefill 量

【三组对照】（默认每个 arm 跑 2 轮，用来看 REUSE 与 REACTIVATE 的差别）

    arm=shared  前缀 1024 token，所有请求共享（1024 是 256 的整数倍）
                → 预期：第 1 条 cached=0，其余 cached=1024；第 2 轮全部 cached=1024
    arm=short   前缀 128 token，所有请求共享（< block_size 256，凑不满一个整块）
                → 预期：cached 全为 0（证明"只有整块参与 cache"）
    arm=random  前缀 1024 token，每条各不相同
                → 预期：cached 全为 0（证明"链式前缀必须逐块相同"）

【跑法】在 vllm 仓库根、已 source env.sh 的 (venv) 下：

    python learn-nano-vllm/experiments/29_prefix_cache_ablation.py

    # 单跑一组（不走子进程）
    python learn-nano-vllm/experiments/29_prefix_cache_ablation.py --arm shared
    python learn-nano-vllm/experiments/29_prefix_cache_ablation.py --arm short
    python learn-nano-vllm/experiments/29_prefix_cache_ablation.py --arm random

【为什么默认走子进程】每个 arm 都要新建一个 LLM 实例（约 10GiB KV cache），
同进程里连续建三个容易因为显存没及时释放而 OOM，所以 --arm all 时每组单独起进程。

【看哪个指标】
- `cached_tokens_per_seq`：命中量的唯一真值（= 每个 seq 在 allocate() 里累加的 num_cached_tokens）
- `computed_prefill_tokens`：真正进了 forward 的 prefill token 数
- `events`：KV_BLOCK_ALLOCATE / REUSE / REACTIVATE 三态计数（挂在 block_manager 的 TEACHING 日志上，
  日志开关是 trace_teaching_mode=True + trace_components=["block_manager"] + log_level=DEBUG）
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from random import Random
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import torch  # noqa: E402

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.engine.block_manager import BlockManager  # noqa: E402

EVENTS = ("KV_BLOCK_ALLOCATE", "KV_BLOCK_REUSE", "KV_BLOCK_REACTIVATE")

# 三组对照的参数。prefix_len 是 256 的整数倍时才可能被整块缓存。
ARMS = {
    "shared": dict(shared_prefix=True, prefix_len=1024),
    "short": dict(shared_prefix=True, prefix_len=128),
    "random": dict(shared_prefix=False, prefix_len=1024),
}


class EventCounter(logging.Handler):
    """挂在 block_manager 的 logger 上，数三态事件。teaching_log 的消息形如 'TEACHING/KV_BLOCK_REUSE | ...'。"""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.counts = {e: 0 for e in EVENTS}

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        for e in EVENTS:
            if f"TEACHING/{e}" in msg:
                self.counts[e] += 1

    def snapshot(self):
        return dict(self.counts)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="S4 Prefix caching 对照实验")
    p.add_argument("--model", type=str, default=os.environ.get("NANOVLLM_MODEL"), help="默认取 $NANOVLLM_MODEL")
    p.add_argument("--arm", type=str, default="all", choices=("all", *ARMS.keys()))
    p.add_argument("--num-seqs", type=int, default=8, help="每轮并发请求数")
    p.add_argument("--suffix-len", type=int, default=64, help="每条请求独有的后缀 token 数")
    p.add_argument("--prefix-len", type=int, default=None, help="覆盖该 arm 默认的前缀长度")
    p.add_argument("--rounds", type=int, default=2, help="同一组 prompt 重复提交几轮（第 2 轮可看到 REACTIVATE）")
    p.add_argument("--output-len", type=int, default=8, help="每条请求生成 token 数（本实验只关心 prefill）")
    p.add_argument("--kvcache-block-size", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--log-level", type=str, default="DEBUG", choices=("DEBUG", "INFO", "WARNING"))
    p.add_argument("--jsonl", type=str, default=None)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--child", action="store_true", help="内部用：标记当前是子进程，只跑一个 arm")
    return p.parse_args(argv)


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def build_prompts(arm, args):
    """用伪随机 token id 直接构造 prompt，长度精确可控（不经过 chat template）。"""
    cfg = ARMS[arm]
    prefix_len = args.prefix_len if args.prefix_len is not None else cfg["prefix_len"]
    rng = Random(1234)  # 固定种子：三个 arm 的 token 分布可比
    shared = [rng.randrange(1000, 30000) for _ in range(prefix_len)]
    prompts = []
    for _ in range(args.num_seqs):
        prefix = shared if cfg["shared_prefix"] else [rng.randrange(1000, 30000) for _ in range(prefix_len)]
        suffix = [rng.randrange(1000, 30000) for _ in range(args.suffix_len)]
        prompts.append(prefix + suffix)
    return prompts, prefix_len


def run_arm(arm, args):
    result = {
        "arm": arm,
        "num_seqs": args.num_seqs,
        "suffix_len": args.suffix_len,
        "status": "ok",
        "rounds": [],
    }

    kwargs = dict(
        enforce_eager=True,
        log_level=args.log_level,
        trace_teaching_mode=True,
        trace_components=["block_manager"],  # 只留 block_manager，避免 DEBUG 日志被别的组件淹没
    )
    if args.kvcache_block_size is not None:
        kwargs["kvcache_block_size"] = args.kvcache_block_size
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs

    llm = LLM(args.model, **kwargs)
    result["num_kvcache_blocks"] = llm.model_runner.config.num_kvcache_blocks
    result["kvcache_block_size"] = llm.model_runner.config.kvcache_block_size

    # 注意：sampling_params 里 assert temperature > 1e-10，不允许 0（纯 greedy）。
    # 采样随机性对本实验无影响（只关心 prefill）。
    sampling = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len)

    # ---- 装两个观测钩子 ----
    counter = EventCounter()
    logging.getLogger("nanovllm.block_manager").addHandler(counter)

    # allocate() 里累加 num_cached_tokens，这是命中量的唯一真值来源。
    alloc_records = []
    orig_allocate = BlockManager.allocate

    def patched_allocate(self, seq):
        orig_allocate(self, seq)
        alloc_records.append(
            {
                "seq_id": seq.seq_id,
                "prompt_tokens": seq.num_prompt_tokens,
                "cached_tokens": seq.num_cached_tokens,
            }
        )

    BlockManager.allocate = patched_allocate

    # ---- warmup：用一段与实验无关的 prompt，避免首跑开销污染 TTFT ----
    if not args.no_warmup:
        warm = [[rng_tok() for _ in range(64)] for _ in range(2)]
        llm.generate(warm, sampling, use_tqdm=False)
        torch.cuda.synchronize()

    prompts, prefix_len = build_prompts(arm, args)
    result["prefix_len"] = prefix_len
    result["prompt_len"] = prefix_len + args.suffix_len

    sched = llm.scheduler
    for r in range(args.rounds):
        n0 = len(alloc_records)
        ev0 = counter.snapshot()
        for p in prompts:
            llm.add_request(p, sampling)

        ttft_ms = {}
        prefill_steps = 0
        decode_steps = 0
        engine_prefill_raw = 0  # step() 返回的正数：Σ(len(seq))，postprocess 之后，含每条新生成的 1 个 token
        prefill_time = 0.0

        t0 = perf_counter()
        while not llm.is_finished():
            t_step = perf_counter()
            outputs, num_tokens = llm.step()
            dt = perf_counter() - t_step
            if num_tokens > 0:
                prefill_steps += 1
                prefill_time += dt
                engine_prefill_raw += num_tokens
            else:
                decode_steps += 1
            now_ms = (perf_counter() - t0) * 1000
            for seq in list(sched.running) + list(sched.waiting):
                if seq.num_completion_tokens > 0 and seq.seq_id not in ttft_ms:
                    ttft_ms[seq.seq_id] = now_ms
            for seq_id, _ in outputs:
                ttft_ms.setdefault(seq_id, now_ms)
        elapsed = perf_counter() - t0
        torch.cuda.synchronize()

        recs = alloc_records[n0:]
        ev1 = counter.snapshot()
        cached = [r["cached_tokens"] for r in recs]
        engine_prefill = engine_prefill_raw - len(prompts)  # 扣掉每条第 1 个生成 token
        computed = sum(r["prompt_tokens"] - r["cached_tokens"] for r in recs)

        result["rounds"].append(
            {
                "round": r + 1,
                # 第 1 轮是冷启动（块还没进 cache）；第 2 轮起重复提交同一批 prompt，
                # 上一轮的块已回 free list 但 hash 还在 → 专门用来看 REACTIVATE。
                "label": "冷启动" if r == 0 else "重复提交",
                "cached_tokens_per_seq": cached,
                "cached_tokens_total": sum(cached),
                "computed_prefill_tokens": computed,
                "engine_prefill_tokens": engine_prefill,
                "saved_prefill_tokens": engine_prefill - computed,
                "prefill_steps": prefill_steps,
                "decode_steps": decode_steps,
                "prefill_time_s": prefill_time,
                "elapsed_s": elapsed,
                "ttft_ms_p50": pct(list(ttft_ms.values()), 0.50),
                "ttft_ms_p90": pct(list(ttft_ms.values()), 0.90),
                "events": {e: ev1[e] - ev0[e] for e in EVENTS},
            }
        )

    BlockManager.allocate = orig_allocate
    return result


_rng = Random(99)


def rng_tok():
    return _rng.randrange(1000, 30000)


def print_arm(result):
    print("=" * 100)
    print(
        f"arm={result['arm']}  prefix={result['prefix_len']} suffix={result['suffix_len']} "
        f"prompt={result['prompt_len']} seqs={result['num_seqs']} "
        f"blocks={result['num_kvcache_blocks']} block_size={result['kvcache_block_size']}"
    )
    print(
        f"{'round':>5} {'':<8} | {'cached/seq':>12} | {'computed':>9} | {'engine口径':>10} | {'省下':>6} | "
        f"{'prefill步':>8} | {'ttft_p50':>9} | {'ALLOC/REUSE/REACTIVATE':>24}"
    )
    for rd in result["rounds"]:
        c = rd["cached_tokens_per_seq"]
        cs = f"{min(c)}~{max(c)}" if c and min(c) != max(c) else (str(c[0]) if c else "-")
        ev = rd["events"]
        print(
            f"{rd['round']:>5} {rd.get('label', ''):<8} | {cs:>12} | {rd['computed_prefill_tokens']:>9} | "
            f"{rd['engine_prefill_tokens']:>10} | {rd['saved_prefill_tokens']:>6} | "
            f"{rd['prefill_steps']:>8} | {rd['ttft_ms_p50']:>7.0f}ms | "
            f"{ev['KV_BLOCK_ALLOCATE']:>6}/{ev['KV_BLOCK_REUSE']:>6}/{ev['KV_BLOCK_REACTIVATE']:>6}"
        )
    print(f"cached_tokens_per_seq 明细: {[rd['cached_tokens_per_seq'] for rd in result['rounds']]}")


def main() -> int:
    args = parse_args()
    if not args.model:
        print("错误：--model 未给出，且环境变量 NANOVLLM_MODEL 为空（先 source env.sh）", file=sys.stderr)
        return 2
    if not os.path.isdir(args.model):
        print(f"错误：模型目录不存在：{args.model}", file=sys.stderr)
        return 2

    # --arm all：每个 arm 单独起一个子进程，避免同进程里连续建 LLM 实例导致显存不释放
    if args.arm == "all" and not args.child:
        results = []
        for a in ARMS:
            cmd = [sys.executable, os.path.abspath(__file__), "--arm", a, "--child",
                   "--model", args.model, "--num-seqs", str(args.num_seqs),
                   "--suffix-len", str(args.suffix_len), "--rounds", str(args.rounds),
                   "--output-len", str(args.output_len), "--log-level", args.log_level]
            if args.prefix_len is not None:
                cmd += ["--prefix-len", str(args.prefix_len)]
            if args.jsonl:
                cmd += ["--jsonl", args.jsonl]
            if args.no_warmup:
                cmd += ["--no-warmup"]
            print(f"\n### 启动子进程：arm={a}", flush=True)
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"arm={a} 失败（退出码 {proc.returncode}）", file=sys.stderr)
                return proc.returncode
        print("\n（上面是各 arm 的独立输出；如需汇总表见 --jsonl）")
        return 0

    try:
        result = run_arm(args.arm, args)
    except Exception as e:  # noqa: BLE001
        msg = (str(e).splitlines() or [""])[0] or repr(e)
        print(f"\n[FAIL] arm={args.arm}: {type(e).__name__}: {msg}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    print_arm(result)
    if args.jsonl:
        with open(args.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"（已追加写入 {args.jsonl}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
