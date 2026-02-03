"""
生成 BCA 随机采样 YAML

生成 (B, C, A) 组合，导出为 YAML 格式，
供 run_benchmark_profile.py --config 直接消费。

采样在每请求空间 (B, c, a) 中进行:
  c = compute_per_request, a = access_per_request
  C = B * c, A = B * a

两种采样策略:

  LHS (默认, 推荐):
    边界样本 + 2D Latin Hypercube 在 (log-B, log-c) 空间采样。
    LHS 保证每个 B 分层与每个 c 分层恰好配对一次，
    从根本上解决独立采样导致 (大B, 大c) 覆盖不足的问题。

  stratified (旧策略):
    边界样本 + 三层 (decode/prefill/chunked) 体积分配 + 1D log-B 分层。

所有采样上界均纳入 KV cache 容量约束。
"""

import argparse
import math
import random
import sys

import yaml

from benchmark_profile.config import validate_params


def log_uniform_int(lo, hi, rng):
    """对数均匀采样整数，返回 [lo, hi] 范围内的值。

    在 [log(lo), log(hi)] 上均匀采样后取 exp 再取整。
    要求 lo >= 1, hi >= lo。
    """
    if lo > hi:
        return None
    if lo == hi:
        return lo
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    val = math.exp(rng.uniform(log_lo, log_hi))
    return max(lo, min(hi, round(val)))


# ---- 核心采样器 (接受固定 B) ----

def _decode_given_B(B, rng, max_len, budget, kv_budget, kv_margin=0,
                    max_concurrent=2):
    """Decode: c=1 固定, 给定 B 采样 a。"""
    c = 1
    C = B * c
    num_groups = min(max_concurrent, B)
    base, rem = divmod(B, num_groups)
    max_gs = base + (1 if rem > 0 else 0)
    if max_gs * c > budget:
        return None
    a_max = max_len - c - 3
    if kv_budget is not None:
        a_max = min(a_max, kv_budget // B - c - 2 - kv_margin)
    if a_max < 1:
        return None
    a = log_uniform_int(1, a_max, rng)
    return (B, C, B * a)


def _prefill_given_B(B, rng, max_len, budget, kv_budget, kv_margin=0,
                     max_concurrent=2):
    """Prefill: a=0 固定, 给定 B 采样 c。"""
    num_groups = min(max_concurrent, B)
    base, rem = divmod(B, num_groups)
    max_gs = base + (1 if rem > 0 else 0)
    # per_step = max_gs * c ≤ budget → c ≤ budget // max_gs
    c_max = min(budget // max_gs, max_len - 3)
    if kv_budget is not None:
        c_max = min(c_max, kv_budget // B - 2 - kv_margin)
    if c_max < 1:
        return None
    c = log_uniform_int(1, c_max, rng)
    return (B, B * c, 0)


def _chunked_given_B(B, rng, max_len, budget, kv_budget, kv_margin=0,
                     max_concurrent=2):
    """Chunked: c>=2, a>=1, 给定 B 采样 c 和 a。"""
    num_groups = min(max_concurrent, B)
    base, rem = divmod(B, num_groups)
    max_gs = base + (1 if rem > 0 else 0)
    # per_step = max_gs * c ≤ budget → c ≤ budget // max_gs
    c_max = min(budget // max_gs, max_len - 4)
    if kv_budget is not None:
        # 留 a>=1 的空间: prompt_len = a+c+2 <= kv_budget//B - kv_margin
        c_max = min(c_max, kv_budget // B - 3 - kv_margin)
    if c_max < 2:
        return None
    c = log_uniform_int(2, c_max, rng)
    a_max = max_len - c - 3
    if kv_budget is not None:
        a_max = min(a_max, kv_budget // B - c - 2 - kv_margin)
    if a_max < 1:
        return None
    a = log_uniform_int(1, a_max, rng)
    return (B, B * c, B * a)


# ---- 公共采样接口 ----

def sample_decode(rng, max_seqs, max_len, budget, kv_budget=None,
                  max_concurrent=2):
    """Decode: c=1 固定, a=log[1, a_max(B)]"""
    B = log_uniform_int(1, max_seqs, rng)
    return _decode_given_B(B, rng, max_len, budget, kv_budget,
                           max_concurrent=max_concurrent)


def sample_prefill(rng, max_seqs, max_len, budget, kv_budget=None,
                   max_concurrent=2):
    """Prefill: a=0 固定, c=log[1, c_max(B)]"""
    B = log_uniform_int(1, max_seqs, rng)
    return _prefill_given_B(B, rng, max_len, budget, kv_budget,
                            max_concurrent=max_concurrent)


def sample_chunked(rng, max_seqs, max_len, budget, kv_budget=None,
                   max_concurrent=2):
    """Chunked: c>=2, a>=1"""
    B = log_uniform_int(1, max_seqs, rng)
    return _chunked_given_B(B, rng, max_len, budget, kv_budget,
                            max_concurrent=max_concurrent)


# ---- 有效区域 log 体积估算 ----

def _estimate_layer_volumes(effective_max_seqs, max_model_len, token_budget,
                            kv_budget, kv_margin=0, max_concurrent=2,
                            n_B=500, n_c=100):
    """数值积分估算各层在 log 空间中的有效区域体积。

    Decode / Prefill 是 2D (log-B × log-a 或 log-c)。
    Chunked 是 3D (log-B × log-c × log-a)。

    Returns:
        dict {layer_name: float} — 各层的 log 空间体积
    """
    if effective_max_seqs <= 1:
        return {"decode": 1.0, "prefill": 1.0, "chunked": 1.0}

    log_B_lo = math.log(1)
    log_B_hi = math.log(effective_max_seqs)
    dlogB = (log_B_hi - log_B_lo) / n_B

    vol_decode = 0.0
    vol_prefill = 0.0
    vol_chunked = 0.0

    for i in range(n_B):
        logB = log_B_lo + (i + 0.5) * dlogB
        B = max(1, round(math.exp(logB)))

        # 计算当前 B 下的 max_group_size
        num_groups = min(max_concurrent, B)
        base, rem = divmod(B, num_groups)
        max_gs = base + (1 if rem > 0 else 0)

        # Decode: c=1, a ∈ [1, a_max]
        a_max_d = max_model_len - 4
        if kv_budget is not None:
            a_max_d = min(a_max_d, kv_budget // B - 3 - kv_margin)
        if a_max_d >= 1:
            vol_decode += math.log(a_max_d) * dlogB

        # Prefill: a=0, c ∈ [1, c_max]
        # per_step = max_gs * c ≤ token_budget → c ≤ token_budget // max_gs
        c_max_p = min(token_budget // max_gs, max_model_len - 3)
        if kv_budget is not None:
            c_max_p = min(c_max_p, kv_budget // B - 2 - kv_margin)
        if c_max_p >= 1:
            vol_prefill += math.log(c_max_p) * dlogB

        # Chunked: c ∈ [2, c_max], a ∈ [1, a_max(c)]
        c_max_ch = min(token_budget // max_gs, max_model_len - 4)
        if kv_budget is not None:
            c_max_ch = min(c_max_ch, kv_budget // B - 3 - kv_margin)
        if c_max_ch >= 2:
            log_c_lo = math.log(2)
            log_c_hi = math.log(c_max_ch)
            dlogc = (log_c_hi - log_c_lo) / n_c
            for j in range(n_c):
                logc = log_c_lo + (j + 0.5) * dlogc
                c = max(2, round(math.exp(logc)))
                a_max_ch = max_model_len - c - 3
                if kv_budget is not None:
                    a_max_ch = min(a_max_ch, kv_budget // B - c - 2 - kv_margin)
                if a_max_ch >= 1:
                    vol_chunked += math.log(a_max_ch) * dlogc * dlogB

    return {"decode": vol_decode, "prefill": vol_prefill, "chunked": vol_chunked}


# ---- B 分层抽样 ----

def _stratified_sample_layer(layer_name, sampler_given_B, target, rng,
                              effective_max_seqs, max_model_len, token_budget,
                              kv_budget, config, seen, kv_margin=0,
                              max_concurrent=2, n_strata=None):
    """在 log-B 上分层抽样，确保 B 维度覆盖均匀。

    将 [1, effective_max_seqs] 的 log 范围等分为 n_strata 段，
    每段内分配等量样本配额。若某段无法生成足够样本，配额
    顺延至后续段。
    """
    if target <= 0:
        return []

    if effective_max_seqs <= 1:
        n_strata = 1
    elif n_strata is None:
        n_strata = max(1, min(target, int(math.sqrt(target))))

    log_B_lo = math.log(1)
    log_B_hi = math.log(max(1, effective_max_seqs))
    strata_width = (log_B_hi - log_B_lo) / n_strata

    # 分配每段配额
    base = target // n_strata
    extra = target % n_strata
    allocations = [base + (1 if i < extra else 0) for i in range(n_strata)]

    samples = []
    shortfall = 0

    for i in range(n_strata):
        stratum_target = allocations[i] + shortfall
        shortfall = 0

        B_lo = max(1, round(math.exp(log_B_lo + i * strata_width)))
        B_hi = max(B_lo, round(math.exp(log_B_lo + (i + 1) * strata_width)))
        if i == n_strata - 1:
            B_hi = max(B_lo, effective_max_seqs)

        collected = 0
        attempts = 0
        max_attempts = max(stratum_target * 50, 200)

        while collected < stratum_target and attempts < max_attempts:
            attempts += 1
            B = log_uniform_int(B_lo, B_hi, rng)
            result = sampler_given_B(B, rng, max_model_len, token_budget,
                                     kv_budget, kv_margin,
                                     max_concurrent=max_concurrent)
            if result is None:
                continue
            _, C, A = result
            valid, _ = validate_params(B, C, A, config)
            if not valid:
                continue
            key = (B, C, A)
            if key in seen:
                continue
            seen.add(key)
            samples.append((layer_name, B, C, A))
            collected += 1

        shortfall = stratum_target - collected

    if shortfall > 0:
        print(f"Warning: {layer_name} layer short by {shortfall} samples "
              f"after stratified sampling", file=sys.stderr)

    return samples


# ---- Latin Hypercube Sampling ----

def _latin_hypercube(n, ndim, rng):
    """生成 n 个 Latin Hypercube 样本，每维度 [0,1]。

    将 [0,1] 分成 n 个等宽分层，每维度的每个分层恰好出现一次。
    保证各维度的边际分布均匀覆盖，同时避免样本聚集。

    Args:
        n: 样本数。
        ndim: 维度数。
        rng: random.Random 实例。

    Returns:
        list of n tuples, each of length ndim, values in [0,1].
    """
    perms = []
    for _ in range(ndim):
        p = list(range(n))
        rng.shuffle(p)
        perms.append(p)
    samples = []
    for i in range(n):
        point = tuple((perms[d][i] + rng.random()) / n for d in range(ndim))
        samples.append(point)
    return samples


def _log_quantile(u, lo, hi):
    """将 u ∈ [0,1] 映射到 [lo, hi] 的对数均匀分位数 (整数)。"""
    if lo >= hi:
        return lo
    val = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    return max(lo, min(hi, round(val)))


def _compute_max_gs(B, max_concurrent):
    """计算给定 B 下的 max_group_size (divmod 调度)。"""
    num_groups = min(max_concurrent, B)
    base, rem = divmod(B, num_groups)
    return base + (1 if rem > 0 else 0)


def _generate_lhs_samples(target, rng, effective_max_seqs, max_model_len,
                           token_budget, kv_budget, config, seen,
                           kv_margin=0, max_concurrent=2,
                           prefill_fraction=0.15):
    """通过 2D LHS 在 (log-B, log-c) 空间生成 BCA 样本。

    在 (log-B, log-c) 平面上做 Latin Hypercube 采样，保证
    每个 B 分层与每个 c 分层恰好配对一次。对于每个 (B, c) 点，
    按约束 clip c，然后 log-uniform 随机采样 a。

    与旧的三层独立采样相比，LHS 保证 (大B, 大c) 等关键区域
    一定被覆盖，从根本上消除因随机性导致 step_time 分布偏斜的问题。

    Args:
        target: 目标样本数。
        rng: random.Random 实例。
        effective_max_seqs: B 的最大值。
        max_model_len: 已减去 margin 的模型长度上限。
        token_budget: per-step token budget。
        kv_budget: KV cache 总容量 (tokens)，可为 None。
        config: validate_params 所需配置字典。
        seen: 已有的 (B, C, A) key 集合，用于去重。
        kv_margin: prompt_len 距理论上限的余量。
        max_concurrent: 并发批次数。
        prefill_fraction: a=0 (纯 prefill) 样本的目标比例。

    Returns:
        list of (layer_name, B, C, A) tuples.
    """
    # c 的全局上限 (B=1 时的理论最大值)
    c_max_global = min(token_budget, max_model_len - 3)
    if kv_budget is not None:
        c_max_global = min(c_max_global, kv_budget - 2 - kv_margin)
    c_max_global = max(1, c_max_global)

    # 超采样以补偿约束裁剪 + 去重导致的丢弃
    oversample = int(target * 1.5) + 30
    lhs_points = _latin_hypercube(oversample, 2, rng)

    samples = []
    for u_b, u_c in lhs_points:
        if len(samples) >= target:
            break

        # Dim 0 → B (log-uniform [1, max_seqs])
        B = _log_quantile(u_b, 1, effective_max_seqs)

        # Dim 1 → c (先映射到全局范围，再 clip 到 B 约束)
        c = _log_quantile(u_c, 1, c_max_global)
        max_gs = _compute_max_gs(B, max_concurrent)
        c_max_B = min(token_budget // max_gs, max_model_len - 3)
        if kv_budget is not None:
            c_max_B = min(c_max_B, kv_budget // B - 2 - kv_margin)
        if c_max_B < 1:
            continue
        c = max(1, min(c, c_max_B))
        C = B * c

        # a: prefill_fraction 概率取 a=0, 其余 log-uniform
        a_max = max_model_len - c - 3
        if kv_budget is not None:
            a_max = min(a_max, kv_budget // B - c - 2 - kv_margin)

        if a_max < 1 or rng.random() < prefill_fraction:
            a = 0
        else:
            a = log_uniform_int(1, a_max, rng)
        A = B * a

        # 去重 + 验证
        key = (B, C, A)
        if key in seen:
            continue
        valid, _ = validate_params(B, C, A, config)
        if not valid:
            continue
        seen.add(key)

        # 按 c, a 值分类 layer
        if c == 1 and a > 0:
            layer = "decode"
        elif a == 0:
            layer = "prefill"
        else:
            layer = "chunked"
        samples.append((layer, B, C, A))

    if len(samples) < target:
        print(f"Warning: LHS generated {len(samples)}/{target} samples "
              f"after constraint filtering", file=sys.stderr)

    return samples


# ---- 边界样本 ----

def _log_steps(lo, hi, n):
    """在 [lo, hi] 范围内生成 n 个对数等间距整数 (含两端)。"""
    if lo > hi or n < 1:
        return []
    if lo == hi or n == 1:
        return [lo]
    pts = set()
    for i in range(n):
        t = i / (n - 1)
        val = round(math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo))))
        pts.add(max(lo, min(hi, val)))
    return sorted(pts)


def generate_boundary_samples(effective_max_seqs, token_budget, max_model_len,
                              config, max_concurrent=2, edge_steps=8):
    """生成边界样本: 角点 + 沿各轴的边扫描。

    边界样本确保 RF 模型在参数空间边界有训练数据，避免外推失准。

    生成策略:
      角点: B, c, a 各取 {min, max} 的有效组合
      边扫: 固定两个维度在极值，第三个维度 log 等间距扫描

    Returns:
        list of (layer_name, B, C, A) tuples, set of seen keys
    """
    max_B = effective_max_seqs
    max_c = min(token_budget, max_model_len - 3)
    max_a = max_model_len - 4  # c=1 时的 a 上限

    samples = []
    seen = set()

    def _try_add(B, c, a):
        C = B * c
        A = B * a
        key = (B, C, A)
        if key in seen:
            return
        valid, _ = validate_params(B, C, A, config)
        if not valid:
            return
        seen.add(key)
        samples.append(("boundary", B, C, A))

    def _max_gs(B):
        """计算给定 B 下的 max_group_size（复制 server divmod 逻辑）。"""
        ng = min(max_concurrent, B)
        base, rem = divmod(B, ng)
        return base + (1 if rem > 0 else 0)

    # --- 角点: B × c × a 各取端值的合法组合 ---
    B_corners = [1, max_B]
    c_corners = [1, max_c]
    a_corners = [0, 1, max_a]
    for B in B_corners:
        for c in c_corners:
            # c 受 B 约束: per_step = max_gs * c ≤ budget
            actual_c = min(c, token_budget // _max_gs(B), max_model_len - 3)
            if actual_c < 1:
                continue
            for a in a_corners:
                # a 受 c 约束: a + c + 2 < max_model_len
                actual_a = min(a, max_model_len - actual_c - 3)
                if actual_a < 0:
                    actual_a = 0
                _try_add(B, actual_c, actual_a)

    # --- 边扫: 沿 B 轴 (固定 c, a 在端值) ---
    B_steps = _log_steps(1, max_B, edge_steps)
    for B in B_steps:
        _try_add(B, 1, 0)                                     # decode 最简
        a_at_c1 = min(max_a, max_model_len - 1 - 3)
        _try_add(B, 1, a_at_c1)                               # decode 最大 context
        c_at_B = min(token_budget // _max_gs(B), max_model_len - 3)
        if c_at_B >= 1:
            _try_add(B, c_at_B, 0)                             # prefill 最大 c

    # --- 边扫: 沿 c 轴 (固定 B=1, a 在端值) ---
    c_steps = _log_steps(1, max_c, edge_steps)
    for c in c_steps:
        _try_add(1, c, 0)                                      # B=1, prefill
        a_at_c = min(max_model_len - c - 3, max_a)
        if a_at_c >= 1:
            _try_add(1, c, a_at_c)                              # B=1, 最大 a

    # --- 边扫: 沿 a 轴 (固定 B=1, c=1) ---
    a_steps = _log_steps(1, max_a, edge_steps)
    for a in a_steps:
        _try_add(1, 1, a)                                       # B=1, decode 扫 context

    # --- 边扫: 沿 a 轴 (固定 B=max_B, c=1) ---
    for a in a_steps:
        _try_add(max_B, 1, a)                                   # max_B, decode 扫 context

    return samples, seen


# ---- 主采样函数 ----

def generate_random_bca_samples(num_samples, token_budget, max_num_seqs,
                                max_model_len, kv_cache_tokens=None,
                                max_concurrent_batches=2, seed=42,
                                prompt_len_margin=50, strategy="lhs"):
    """主采样函数: 边界样本 + 内部随机采样。

    Args:
        kv_cache_tokens: KV cache 总容量 (tokens)。若提供，则约束
            B * prompt_len <= kv_cache_tokens。
            可通过 num_gpu_blocks * block_size 计算得到。
        max_concurrent_batches: profile_step 的并发批次数，
            用于 divmod 分发计算 per-step budget 约束。
        prompt_len_margin: 每请求长度距理论上限的余量。
            采样上界为 max_model_len - prompt_len_margin，
            避免触碰极端边界。
        strategy: 采样策略:
            "lhs" - 边界 + 2D Latin Hypercube (推荐, 保证 B×c 网格覆盖)
            "stratified" - 边界 + 三层体积分配 + B 分层 (旧策略)

    Returns:
        list of (layer_name, B, C, A) tuples
    """
    rng = random.Random(seed)

    config = {
        "token_budget": token_budget,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "max_concurrent_batches": max_concurrent_batches,
    }
    if kv_cache_tokens is not None:
        config["num_gpu_blocks"] = kv_cache_tokens
        config["block_size"] = 1

    # B 的实际上界（total 语义：B 就是总请求数，不需要除以 max_concurrent）
    effective_max_seqs = max_num_seqs
    # KV cache 总容量（total 语义：不需要除以 max_concurrent）
    kv_budget = kv_cache_tokens
    # 采样用的模型长度上限 (留 margin 余量)
    sampling_max_len = max_model_len - prompt_len_margin

    # 1. 边界样本 (确保模型在参数空间边界有训练数据)
    boundary_samples, seen = generate_boundary_samples(
        effective_max_seqs, token_budget, sampling_max_len, config,
        max_concurrent=max_concurrent_batches)

    remaining = max(0, num_samples - len(boundary_samples))

    if strategy == "lhs":
        # ---- LHS 策略: 2D Latin Hypercube in (log-B, log-c) ----
        print(f"  strategy: LHS (2D: log-B x log-c)")
        lhs_samples = _generate_lhs_samples(
            remaining, rng, effective_max_seqs, sampling_max_len,
            token_budget, kv_budget, config, seen,
            kv_margin=prompt_len_margin,
            max_concurrent=max_concurrent_batches)
        all_samples = list(boundary_samples) + lhs_samples
    else:
        # ---- 旧策略: 三层体积分配 + B 分层随机采样 ----
        volumes = _estimate_layer_volumes(
            effective_max_seqs, sampling_max_len, token_budget, kv_budget,
            kv_margin=prompt_len_margin,
            max_concurrent=max_concurrent_batches)

        layer_spec = [
            ("decode", _decode_given_B),
            ("prefill", _prefill_given_B),
            ("chunked", _chunked_given_B),
        ]

        # Chunked 的 3D 体积远大于 2D 层，缩减权重以平衡覆盖密度
        weights = dict(volumes)
        weights["chunked"] = weights.get("chunked", 0) / 2

        # Largest-remainder 分配法
        total_weight = sum(weights.values())
        if total_weight > 0:
            raw = {name: remaining * weights[name] / total_weight
                   for name in weights}
        else:
            raw = {name: remaining / len(layer_spec) for name in weights}

        allocations = {name: max(1, int(v)) for name, v in raw.items()}
        alloc_remainder = remaining - sum(allocations.values())
        frac_order = sorted(raw.keys(),
                            key=lambda n: -(raw[n] - int(raw[n])))
        for name in frac_order:
            if alloc_remainder <= 0:
                break
            allocations[name] += 1
            alloc_remainder -= 1

        print(f"  layer volumes: "
              f"{ {k: f'{v:.1f}' for k, v in volumes.items()} }")
        print(f"  layer allocations: {allocations} "
              f"(boundary={len(boundary_samples)})")

        all_samples = list(boundary_samples)
        for layer_name, sampler_given_B in layer_spec:
            target = allocations.get(layer_name, 0)
            layer_samples = _stratified_sample_layer(
                layer_name, sampler_given_B, target, rng,
                effective_max_seqs, sampling_max_len, token_budget,
                kv_budget, config, seen, kv_margin=prompt_len_margin,
                max_concurrent=max_concurrent_batches)
            all_samples.extend(layer_samples)

    return all_samples


# ---- 输出 ----

def dump_yaml(samples, token_budget, max_num_seqs, max_model_len, seed,
              num_samples, output_path, max_concurrent_batches=2,
              kv_cache_tokens=None):
    """将采样结果导出为 YAML 文件。"""
    # 按层分组
    layers = {}
    for layer_name, B, C, A in samples:
        layers.setdefault(layer_name, []).append((B, C, A))

    # 构建 test_cases 列表
    test_cases = []
    for layer_name, B, C, A in samples:
        test_cases.append({
            "batch_size": B,
            "compute_tokens": C,
            "access_tokens": A,
            "layer": layer_name,
        })

    # 构建完整 YAML 结构
    doc = {
        "token_budget": token_budget,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "max_concurrent_batches": max_concurrent_batches,
    }
    if kv_cache_tokens is not None:
        doc["kv_cache_tokens"] = kv_cache_tokens
    doc["test_cases"] = test_cases

    # 构建 header 注释
    layer_counts = {}
    for layer_name, _, _, _ in samples:
        layer_counts[layer_name] = layer_counts.get(layer_name, 0) + 1
    layer_summary = ", ".join(f"{k}={v}" for k, v in layer_counts.items())

    header_lines = [
        f"# Auto-generated BCA samples for RF training",
        f"# seed: {seed}, num_samples: {num_samples}, actual: {len(samples)}",
        f"# constraints: token_budget={token_budget}, max_num_seqs={max_num_seqs}, "
        f"max_model_len={max_model_len}, max_concurrent_batches={max_concurrent_batches}, "
        f"kv_cache_tokens={kv_cache_tokens}",
        f"# layer distribution: {layer_summary}",
        "",
    ]
    header = "\n".join(header_lines)

    yaml_body = yaml.dump(doc, default_flow_style=False, sort_keys=False,
                          allow_unicode=True)

    with open(output_path, "w") as f:
        f.write(header)
        f.write(yaml_body)

    return output_path


def print_stats(samples):
    """打印采样统计信息。"""
    if not samples:
        print("No samples generated.", file=sys.stderr)
        return

    Bs = [s[1] for s in samples]
    Cs = [s[2] for s in samples]
    As = [s[3] for s in samples]

    # per-request 值
    cs = [C // B for _, B, C, A in samples]
    a_s = [A // B if B > 0 else 0 for _, B, C, A in samples]

    def stats(name, vals):
        return (f"  {name:20s}: min={min(vals):8d}, max={max(vals):8d}, "
                f"mean={sum(vals)/len(vals):10.1f}, n={len(vals)}")

    print("\n--- BCA Sample Statistics ---")
    print(f"Total samples: {len(samples)}")

    # 层分布
    layer_counts = {}
    for layer_name, _, _, _ in samples:
        layer_counts[layer_name] = layer_counts.get(layer_name, 0) + 1
    for layer, cnt in layer_counts.items():
        print(f"  {layer:20s}: {cnt} samples ({100*cnt/len(samples):.1f}%)")

    print("\nTotal values (B, C, A):")
    print(stats("B (batch_size)", Bs))
    print(stats("C (compute_tokens)", Cs))
    print(stats("A (access_tokens)", As))

    print("\nPer-request values (c, a):")
    print(stats("c (compute/req)", cs))
    print(stats("a (access/req)", a_s))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate BCA random samples as YAML for benchmark profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example usage:
  python generate_bca_samples.py --num-samples 500 --seed 42 -o bca_workloads.yaml
  python run_benchmark_profile.py --config bca_workloads.yaml --output-file results.csv
""",
    )
    parser.add_argument(
        "--num-samples", type=int, default=500,
        help="Number of BCA samples to generate (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=2048,
        dest="token_budget",
        help="Token budget / max_num_batched_tokens (default: 2048)",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=128,
        help="Maximum number of sequences (default: 128)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="Maximum model length (default: 8192)",
    )
    parser.add_argument(
        "--kv-cache-tokens", type=int, default=None,
        help="Total KV cache capacity in tokens (= num_gpu_blocks * block_size). "
             "Constrains B * prompt_len <= kv_cache_tokens. "
             "If not set, KV cache capacity check is skipped.",
    )
    parser.add_argument(
        "--max-concurrent-batches", type=int, default=2,
        dest="max_concurrent_batches",
        help="Number of concurrent pipeline batches (typically = PP size). "
             "Used for divmod distribution to compute per-step budget. (default: 2)",
    )
    parser.add_argument(
        "--strategy", choices=["lhs", "stratified"], default="lhs",
        help="Sampling strategy: 'lhs' (Latin Hypercube, recommended) "
             "or 'stratified' (legacy 3-layer volume-based). Default: lhs.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="bca_workloads.yaml",
        help="Output YAML file path (default: bca_workloads.yaml)",
    )

    args = parser.parse_args()

    print(f"Generating {args.num_samples} BCA samples...")
    print(f"  token_budget={args.token_budget}, max_num_seqs={args.max_num_seqs}, "
          f"max_model_len={args.max_model_len}, kv_cache_tokens={args.kv_cache_tokens}, "
          f"max_concurrent_batches={args.max_concurrent_batches}")
    print(f"  seed={args.seed}")

    samples = generate_random_bca_samples(
        num_samples=args.num_samples,
        token_budget=args.token_budget,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        kv_cache_tokens=args.kv_cache_tokens,
        max_concurrent_batches=args.max_concurrent_batches,
        seed=args.seed,
        strategy=args.strategy,
    )

    print_stats(samples)

    output_path = dump_yaml(
        samples=samples,
        token_budget=args.token_budget,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
        num_samples=args.num_samples,
        output_path=args.output,
        max_concurrent_batches=args.max_concurrent_batches,
        kv_cache_tokens=args.kv_cache_tokens,
    )
    print(f"Written {len(samples)} test cases to {output_path}")


if __name__ == "__main__":
    main()
