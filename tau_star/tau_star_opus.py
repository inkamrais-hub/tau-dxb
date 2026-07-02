"""τ* 求解器 — 通用版，适配 τ-softplus 和 τ-opus

核心公式: τ* = Cov(s, log σ(s)) / Var(log σ(s))

支持任意 σ 函数:
  softplus, sigmoid, exp, relu, gelu, silu, tanh_shift

Pipeline:
  1. σ(s) 变换
  2. 统计量计算: cov_ratio [, skew, kurt]
  3. 回归 (可选 OLS)
  4. 裁剪 [tau_min, tau_max]

用法:
  from tau_star_opus import tau_star_opus

  # 单 head
  tau, stats = tau_star_opus(scores, sigma='softplus')       # 闭式

  # 批量 per-head
  est = TauStarOpus(sigma='softplus')
  taus, stats_list = est.batch(scores)  # [B, H, Lq, Lk] → [H]

  # 自定义 sigma
  est = TauStarOpus(sigma='sigmoid', tau_min=0.5, tau_max=10.0)
  tau, stats = est(scores)

  # 逐层
  tau, stats = est.per_layer(scores, layer_idx=3)

预置便捷函数:
  tau_star_opus(scores, sigma='softplus')       — 闭式 (推荐)
  tau_star_opus_regressed(scores, sigma='softplus') — OLS 回归
"""
import math
import torch
import torch.nn.functional as F

EPS = 1e-8
CLAMP_MIN = 1e-8

# ═══════════════════════════════════════════════════════════════
# σ 函数库 (与 stau_opus.py 保持一致)
# ═══════════════════════════════════════════════════════════════

def _sigma_softplus(x):
    return F.softplus(x).clamp(min=CLAMP_MIN)

def _sigma_relu(x):
    return F.relu(x).clamp(min=CLAMP_MIN)

def _sigma_silu(x):
    return F.silu(x).clamp(min=CLAMP_MIN)

def _sigma_gelu(x):
    return F.gelu(x).clamp(min=CLAMP_MIN)

def _sigma_exp(x):
    return torch.exp(x.clamp(max=20)).clamp(min=CLAMP_MIN)

def _sigma_tanh_shift(x):
    return (torch.tanh(x) + 1).clamp(min=CLAMP_MIN)

def _sigma_sigmoid(x):
    return torch.sigmoid(x).clamp(min=CLAMP_MIN)

SIGMA_REGISTRY = {
    "softplus": _sigma_softplus,
    "relu": _sigma_relu,
    "silu": _sigma_silu,
    "gelu": _sigma_gelu,
    "exp": _sigma_exp,
    "tanh_shift": _sigma_tanh_shift,
    "sigmoid": _sigma_sigmoid,
}

# ═══════════════════════════════════════════════════════════════
# 预置 OLS 系数 (per-sigma, 需 recalibrate 校准)
# ═══════════════════════════════════════════════════════════════

# softplus: 沿用 tau_star.py 的校准系数 (Qwen3-0.6B, R²=0.79)
SOFTPLUS_REGRESSION_CFG = {
    'coef_cov': 0.912,
    'coef_skew': -3.902,
    'coef_kurt': -0.235,
    'coef_bias': 1.932,
    'tau_min': 1.05,
    'tau_max': 20.0,
}

# clamp (legacy): 旧版系数
CLAMP_REGRESSION_CFG = {
    'coef_cov': 0.89,
    'coef_skew': -3.10,
    'coef_kurt': -2.41,
    'coef_bias': 1.0,
    'tau_min': 1.05,
    'tau_max': 20.0,
}

# 其他 σ 函数默认用闭式 (纯 Cov/Var)，无回归系数
# sigmoid/exp/relu/gelu/silu 的回归系数需要 recalibrate() 校准

# 逐层系数 (softplus, Qwen3-0.6B 28L×16H, R²=0.90)
PER_LAYER_CFG = {
    0:  {'bias': 5.456,  'coef_cov': 0.473,  'coef_skew': 12.997, 'coef_kurt': -2.464},
    1:  {'bias': 0.645,  'coef_cov': 0.949,  'coef_skew': 0.655,  'coef_kurt': -1.943},
    2:  {'bias': 1.505,  'coef_cov': 0.940,  'coef_skew': -5.031, 'coef_kurt': 2.996},
    3:  {'bias': 1.053,  'coef_cov': 1.162,  'coef_skew': -8.022, 'coef_kurt': -8.263},
    4:  {'bias': 0.198,  'coef_cov': 1.162,  'coef_skew': -3.143, 'coef_kurt': -6.605},
    5:  {'bias': 1.760,  'coef_cov': 0.749,  'coef_skew': -0.134, 'coef_kurt': -2.501},
    6:  {'bias': 1.998,  'coef_cov': 0.909,  'coef_skew': -4.648, 'coef_kurt': -13.065},
    7:  {'bias': 1.322,  'coef_cov': 0.966,  'coef_skew': -7.693, 'coef_kurt': 4.994},
    8:  {'bias': 1.043,  'coef_cov': 1.183,  'coef_skew': -3.196, 'coef_kurt': 3.918},
    9:  {'bias': 0.342,  'coef_cov': 1.072,  'coef_skew': 0.914,  'coef_kurt': 1.995},
    10: {'bias': 1.013,  'coef_cov': 0.893,  'coef_skew': -4.302, 'coef_kurt': 0.586},
    11: {'bias': 0.663,  'coef_cov': 1.173,  'coef_skew': -6.883, 'coef_kurt': -2.799},
    12: {'bias': 0.755,  'coef_cov': 1.014,  'coef_skew': -1.558, 'coef_kurt': -1.147},
    13: {'bias': 1.404,  'coef_cov': 0.948,  'coef_skew': -3.457, 'coef_kurt': -0.232},
    14: {'bias': 0.543,  'coef_cov': 1.050,  'coef_skew': 0.704,  'coef_kurt': -3.320},
    15: {'bias': 0.483,  'coef_cov': 0.915,  'coef_skew': 2.452,  'coef_kurt': -3.131},
    16: {'bias': 0.010,  'coef_cov': 1.099,  'coef_skew': 1.683,  'coef_kurt': -4.648},
    17: {'bias': -0.588, 'coef_cov': 0.996,  'coef_skew': 3.932,  'coef_kurt': -3.408},
    18: {'bias': 2.365,  'coef_cov': 0.917,  'coef_skew': -6.555, 'coef_kurt': 0.741},
    19: {'bias': 5.556,  'coef_cov': 0.488,  'coef_skew': -8.972, 'coef_kurt': 7.514},
    20: {'bias': 0.655,  'coef_cov': 1.073,  'coef_skew': -1.412, 'coef_kurt': -2.847},
    21: {'bias': 0.174,  'coef_cov': 1.173,  'coef_skew': -1.576, 'coef_kurt': -4.119},
    22: {'bias': 2.769,  'coef_cov': 1.132,  'coef_skew': -17.963,'coef_kurt': 15.877},
    23: {'bias': 0.440,  'coef_cov': 1.138,  'coef_skew': -2.160, 'coef_kurt': 5.947},
    24: {'bias': 0.507,  'coef_cov': 0.899,  'coef_skew': 10.181, 'coef_kurt': -18.926},
    25: {'bias': 3.488,  'coef_cov': 0.746,  'coef_skew': -1.662, 'coef_kurt': -4.278},
    26: {'bias': 1.304,  'coef_cov': 1.151,  'coef_skew': -8.136, 'coef_kurt': 1.288},
    27: {'bias': -2.017, 'coef_cov': 1.426,  'coef_skew': 8.090,  'coef_kurt': -4.507},
}

PER_CLUSTER_CFG = {
    'shallow': {'bias': 2.430, 'coef_cov': 0.798, 'coef_skew': -1.826, 'coef_kurt': 2.210, 'layers': 'L0-L8'},
    'middle':  {'bias': 0.130, 'coef_cov': 1.115, 'coef_skew': -1.697, 'coef_kurt': -2.144, 'layers': 'L9-L17'},
    'deep':    {'bias': 2.729, 'coef_cov': 0.916, 'coef_skew': -7.465, 'coef_kurt': 2.263, 'layers': 'L18-L27'},
}


# ═══════════════════════════════════════════════════════════════
# TauStarOpus — 通用 τ* 求解器
# ═══════════════════════════════════════════════════════════════

class TauStarOpus:
    """通用 τ* 估计器，适配任意 σ 函数。

    Args:
        sigma: str — σ 函数名: 'softplus'|'sigmoid'|'exp'|'relu'|'gelu'|'silu'|'tanh_shift'
               或 callable — 自定义 σ 函数
        stats: list — 统计量: ['cov_ratio'] | ['cov_ratio', 'skew', 'kurt']
        coef: dict or None — OLS 系数, None 表示纯 Cov/Var 闭式
        tau_min: float — 裁剪下限
        tau_max: float — 裁剪上限
        cov_on: str — Cov 的计算方式: 's' (Cov(s, log σ)) | 'phi' (Cov(σ, log σ))
        eps: float — 数值稳定常数

    Pre-built:
        TauStarOpus.softplus_closed()  — softplus + 纯 Cov/Var (推荐)
        TauStarOpus.softplus_regressed() — softplus + OLS 回归
        TauStarOpus.sigmoid_closed()  — sigmoid + 纯 Cov/Var
        TauStarOpus.exp_closed()      — exp + 纯 Cov/Var
        TauStarOpus.relu_closed()     — relu + 纯 Cov/Var
        TauStarOpus.gelu_closed()     — gelu + 纯 Cov/Var
        TauStarOpus.silu_closed()     — silu + 纯 Cov/Var
    """

    def __init__(self, sigma='softplus', stats=None, coef=None,
                 tau_min=0.5, tau_max=20.0, cov_on='s', eps=EPS):
        if stats is None:
            stats = ['cov_ratio']
        valid_stats = {'cov_ratio', 'skew', 'kurt'}
        for s in stats:
            if s not in valid_stats:
                raise ValueError(f"Unknown stat: {s}. Options: {valid_stats}")
        if cov_on not in ('s', 'phi'):
            raise ValueError(f"cov_on must be 's' or 'phi', got {cov_on}")

        # 解析 sigma 函数
        if callable(sigma):
            self.sigma_name = 'custom'
            self._sigma_fn = sigma
        elif sigma in SIGMA_REGISTRY:
            self.sigma_name = sigma
            self._sigma_fn = SIGMA_REGISTRY[sigma]
        else:
            raise ValueError(f"Unknown sigma: {sigma}. Options: {list(SIGMA_REGISTRY.keys())}")

        self.stats = list(stats)
        self.coef = coef
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.cov_on = cov_on
        self.eps = eps

    def __repr__(self):
        coef_str = 'OLS' if self.coef else 'closed'
        return (f"TauStarOpus(sigma='{self.sigma_name}', stats={self.stats}, "
                f"regression={coef_str}, cov_on='{self.cov_on}', "
                f"tau=[{self.tau_min}, {self.tau_max}])")

    def __call__(self, scores):
        """估计单个 head 的 τ*.

        Args:
            scores: [Lq, Lk] pre-softmax attention scores

        Returns:
            tau: float
            stats: dict
        """
        return self._estimate(scores)

    # ── 预置构造器 ──────────────────────────────────────────

    @classmethod
    def softplus_closed(cls):
        """softplus + 纯 Cov/Var (CANONICAL, R² 最高)."""
        return cls(sigma='softplus', stats=['cov_ratio'], coef=None)

    @classmethod
    def softplus_regressed(cls):
        """softplus + cov_ratio + skew + kurt (OLS, R²=0.79)."""
        return cls(sigma='softplus', stats=['cov_ratio', 'skew', 'kurt'],
                   coef=SOFTPLUS_REGRESSION_CFG.copy())

    @classmethod
    def sigmoid_closed(cls):
        """sigmoid + 纯 Cov/Var."""
        return cls(sigma='sigmoid', stats=['cov_ratio'], coef=None,
                   tau_min=0.3, tau_max=10.0)

    @classmethod
    def exp_closed(cls):
        """exp + 纯 Cov/Var."""
        return cls(sigma='exp', stats=['cov_ratio'], coef=None,
                   tau_min=0.1, tau_max=5.0)

    @classmethod
    def relu_closed(cls):
        """relu + 纯 Cov/Var."""
        return cls(sigma='relu', stats=['cov_ratio'], coef=None)

    @classmethod
    def gelu_closed(cls):
        """gelu + 纯 Cov/Var."""
        return cls(sigma='gelu', stats=['cov_ratio'], coef=None)

    @classmethod
    def silu_closed(cls):
        """silu + 纯 Cov/Var."""
        return cls(sigma='silu', stats=['cov_ratio'], coef=None)

    @classmethod
    def tanh_shift_closed(cls):
        """tanh_shift + 纯 Cov/Var."""
        return cls(sigma='tanh_shift', stats=['cov_ratio'], coef=None,
                   tau_min=0.3, tau_max=10.0)

    # ── 核心估计 ──────────────────────────────────────────

    def _estimate(self, scores):
        """单 head τ* 估计."""
        valid = scores > -1e4
        s = scores[valid].float()
        if s.numel() < 10:
            return self.tau_min, {'cov_ratio': 0.0, 'n_valid': s.numel()}

        phi = self._sigma_fn(s)
        log_phi = phi.log()

        if self.cov_on == 'phi':
            cov = ((phi - phi.mean()) * (log_phi - log_phi.mean())).mean()
        else:
            cov = ((s - s.mean()) * (log_phi - log_phi.mean())).mean()

        var_log = log_phi.var().clamp(min=self.eps)
        cov_ratio = (cov / var_log).item()

        stats = {'cov_ratio': round(cov_ratio, 4), 'n_valid': int(s.numel())}

        if self.coef is None:
            tau_val = cov_ratio
        else:
            c = self.coef
            tau_val = c.get('coef_bias', 1.0) + c['coef_cov'] * cov_ratio
            if 'skew' in self.stats:
                std_s = s.std().clamp(min=self.eps)
                z = (s - s.mean()) / std_s
                skew = (z ** 3).mean().item()
                tau_val += c['coef_skew'] * math.tanh(skew / 5.0)
                stats['skew'] = round(skew, 4)
            if 'kurt' in self.stats:
                if 'skew' not in self.stats:
                    std_s = s.std().clamp(min=self.eps)
                    z = (s - s.mean()) / std_s
                kurt = (z ** 4).mean().item() - 3.0
                tau_val += c['coef_kurt'] * math.tanh(kurt / 10.0)
                stats['kurt'] = round(kurt, 4)

        tau_val = max(self.tau_min, min(self.tau_max, tau_val))
        stats['tau'] = round(tau_val, 4)
        return tau_val, stats

    # ── 批量估计 ──────────────────────────────────────────

    def batch(self, scores):
        """批量估计 [B, H, Lq, Lk] → [H] tensor.

        Args:
            scores: [B, H, Lq, Lk] pre-softmax attention scores

        Returns:
            taus: [H] tensor
            stats_list: list of per-head stats dicts
        """
        H = scores.shape[1]
        taus = torch.zeros(H, dtype=torch.float32, device=scores.device)
        stats_list = []
        for h in range(H):
            tau_h, stats_h = self._estimate(scores[0, h])
            taus[h] = tau_h
            stats_list.append(stats_h)
        return taus, stats_list

    def per_head_map(self, scores):
        """批量估计 [B, H, Lq, Lk] → dict {layer_idx: [H] tensor}.

        返回 dict 格式，兼容 run_gpt2_taustar.py 的 tau_star_map。
        """
        H = scores.shape[1]
        taus = torch.zeros(H, dtype=torch.float32, device=scores.device)
        stats_list = []
        for h in range(H):
            tau_h, stats_h = self._estimate(scores[0, h])
            taus[h] = tau_h
            stats_list.append(stats_h)
        return taus, stats_list

    # ── 逐层估计 ──────────────────────────────────────────

    def per_layer(self, scores, layer_idx):
        """逐层 τ* 估计 (使用 PER_LAYER_CFG 系数, 仅 softplus).

        Args:
            scores: [Lq, Lk] pre-softmax attention scores
            layer_idx: int

        Returns:
            tau: float, stats: dict
        """
        c = PER_LAYER_CFG.get(layer_idx)
        if c is None:
            n_layers = len(PER_LAYER_CFG)
            third = max(PER_LAYER_CFG.keys()) // 3 if n_layers > 0 else 9
            if layer_idx < third:
                c = PER_CLUSTER_CFG['shallow']
            elif layer_idx < 2 * third:
                c = PER_CLUSTER_CFG['middle']
            else:
                c = PER_CLUSTER_CFG['deep']
        old_coef = self.coef
        self.coef = c
        tau, stats = self._estimate(scores)
        self.coef = old_coef
        return tau, stats

    def per_cluster(self, scores, layer_idx):
        """三簇 τ* 估计 (shallow/middle/deep)."""
        n_layers = len(PER_LAYER_CFG)
        third = max(PER_LAYER_CFG.keys()) // 3 if n_layers > 0 else 9
        if layer_idx < third:
            c = PER_CLUSTER_CFG['shallow']
        elif layer_idx < 2 * third:
            c = PER_CLUSTER_CFG['middle']
        else:
            c = PER_CLUSTER_CFG['deep']
        old_coef = self.coef
        self.coef = c
        tau, stats = self._estimate(scores)
        self.coef = old_coef
        return tau, stats

    # ── 重标定 ──────────────────────────────────────────

    def recalibrate(self, pairs):
        """OLS 拟合新系数 (用于新 σ 函数).

        Args:
            pairs: list of (scores_matrix, tau_ground_truth)
                scores: [Lq, Lk] pre-softmax attention scores
                gt_tau: float — verified τ* (e.g. from grid-search)

        Returns:
            coef: dict — 新系数
            r2: float — OLS R²
        """
        rows = []
        targets = []

        for scores, gt_tau in pairs:
            valid = scores > -1e4
            s = scores[valid].float()
            if s.numel() < 10:
                continue

            phi = self._sigma_fn(s)
            log_phi = phi.log()

            if self.cov_on == 'phi':
                cov = ((phi - phi.mean()) * (log_phi - log_phi.mean())).mean()
            else:
                cov = ((s - s.mean()) * (log_phi - log_phi.mean())).mean()
            var_log = log_phi.var().clamp(min=self.eps)
            cov_ratio = (cov / var_log).item()

            features = [cov_ratio]
            if 'skew' in self.stats:
                std_s = s.std().clamp(min=self.eps)
                z = (s - s.mean()) / std_s
                skew = (z ** 3).mean().item()
                features.append(math.tanh(skew / 5.0))
            if 'kurt' in self.stats:
                if 'skew' not in self.stats:
                    std_s = s.std().clamp(min=self.eps)
                    z = (s - s.mean()) / std_s
                kurt = (z ** 4).mean().item() - 3.0
                features.append(math.tanh(kurt / 10.0))

            rows.append(features)
            targets.append(gt_tau)

        if len(rows) < 10:
            return (self.coef or {}).copy(), 0.0

        X = torch.tensor(rows)
        y = torch.tensor(targets)
        coeffs = torch.linalg.lstsq(X, y).solution
        r2 = 1.0 - ((y - X @ coeffs) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2 = r2.item()

        new_coef = {}
        idx = 0
        new_coef['coef_cov'] = round(coeffs[idx].item(), 4)
        idx += 1
        if 'skew' in self.stats:
            new_coef['coef_skew'] = round(coeffs[idx].item(), 4)
            idx += 1
        if 'kurt' in self.stats:
            new_coef['coef_kurt'] = round(coeffs[idx].item(), 4)
            idx += 1
        new_coef['tau_min'] = self.tau_min
        new_coef['tau_max'] = self.tau_max

        self.coef = new_coef
        return new_coef, round(r2, 4)


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

# 全局默认估计器 (lazy init)
_ESTIMATORS = {}

def _get_estimator(sigma, regressed=False):
    key = (sigma, regressed)
    if key not in _ESTIMATORS:
        if regressed and sigma == 'softplus':
            _ESTIMATORS[key] = TauStarOpus.softplus_regressed()
        elif sigma == 'softplus':
            _ESTIMATORS[key] = TauStarOpus.softplus_closed()
        elif sigma == 'sigmoid':
            _ESTIMATORS[key] = TauStarOpus.sigmoid_closed()
        elif sigma == 'exp':
            _ESTIMATORS[key] = TauStarOpus.exp_closed()
        elif sigma == 'relu':
            _ESTIMATORS[key] = TauStarOpus.relu_closed()
        elif sigma == 'gelu':
            _ESTIMATORS[key] = TauStarOpus.gelu_closed()
        elif sigma == 'silu':
            _ESTIMATORS[key] = TauStarOpus.silu_closed()
        elif sigma == 'tanh_shift':
            _ESTIMATORS[key] = TauStarOpus.tanh_shift_closed()
        else:
            _ESTIMATORS[key] = TauStarOpus(sigma=sigma, coef=SOFTPLUS_REGRESSION_CFG.copy() if regressed else None)
    return _ESTIMATORS[key]


def tau_star_opus(scores, sigma='softplus'):
    """τ* 闭式估计 — 通用版.

    Args:
        scores: [Lq, Lk] pre-softmax attention scores
        sigma: str — σ 函数名

    Returns:
        tau: float, stats: dict
    """
    return _get_estimator(sigma, regressed=False)(scores)


def tau_star_opus_regressed(scores, sigma='softplus'):
    """τ* OLS 回归估计 — 通用版 (仅 softplus 有校准系数).

    Args:
        scores: [Lq, Lk] pre-softmax attention scores
        sigma: str — σ 函数名

    Returns:
        tau: float, stats: dict
    """
    return _get_estimator(sigma, regressed=True)(scores)


def tau_star_opus_batch(scores, sigma='softplus', regressed=False):
    """批量 τ* 估计 [B, H, Lq, Lk] → [H] tensor.

    Args:
        scores: [B, H, Lq, Lk]
        sigma: str
        regressed: bool

    Returns:
        taus: [H] tensor, stats_list: list
    """
    return _get_estimator(sigma, regressed).batch(scores)


def tau_star_opus_perlayer(scores, layer_idx, sigma='softplus'):
    """逐层 τ* 估计."""
    return _get_estimator(sigma, regressed=False).per_layer(scores, layer_idx)


# ═══════════════════════════════════════════════════════════════
# 向后兼容 — 与 tau_star.py 接口兼容
# ═══════════════════════════════════════════════════════════════

# tau_star() 别名，兼容 run_gpt2_taustar.py 的 import
def tau_star(scores):
    """[兼容] 等价于 tau_star_opus(scores, sigma='softplus')."""
    return _get_estimator('softplus', regressed=False)(scores)


def tau_star_sp_closed(scores):
    """[兼容] softplus + 闭式."""
    return _get_estimator('softplus', regressed=False)(scores)


def tau_star_regressed(scores):
    """[兼容] softplus + OLS."""
    return _get_estimator('softplus', regressed=True)(scores)


# ═══════════════════════════════════════════════════════════════
# 快速诊断
# ═══════════════════════════════════════════════════════════════

def diagnose_scores(scores, sigma='softplus'):
    """诊断 scores 分布，打印所有 σ 函数的 τ* 估计.

    Args:
        scores: [Lq, Lk] pre-softmax attention scores
        sigma: str — 主要 σ 函数 (打印详细信息)
    """
    valid = scores > -1e4
    s = scores[valid].float()

    print(f"  scores 分布: n={s.numel()}, mean={s.mean().item():.3f}, "
          f"std={s.std().item():.3f}, min={s.min().item():.3f}, max={s.max().item():.3f}")
    print(f"  {'σ':>14s} {'τ*':>8s} {'cov_ratio':>10s} {'备注':>20s}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*20}")

    results = {}
    for name in SIGMA_REGISTRY:
        est = TauStarOpus(sigma=name)
        tau_val, stats = est(scores)
        note = ""
        if name == sigma:
            note = "← 目标 σ"
        if name == 'softplus':
            note += " (基准)"
        print(f"  {name:>14s} {tau_val:>8.3f} {stats['cov_ratio']:>10.4f} {note:>20s}")
        results[name] = tau_val

    return results


# ═══════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("TauStarOpus 自测")
    print("=" * 60)

    torch.manual_seed(42)

    # 生成模拟 scores
    n = 64
    for dist_name, scores in [
        ("高斯 N(0,1)", torch.randn(n)),
        ("高斯 N(0,3)", torch.randn(n) * 3),
        ("有峰值", torch.randn(n) * 0.5 + torch.tensor([8.0] + [0.0] * (n - 1))),
        ("均匀", torch.rand(n) * 6 - 3),
    ]:
        print(f"\n--- {dist_name} ---")
        diagnose_scores(scores.unsqueeze(0))

    # 测试批量
    print(f"\n--- 批量测试 [2, 4, 16, 16] ---")
    batch_scores = torch.randn(2, 4, 16, 16)
    for sigma_name in ['softplus', 'sigmoid', 'exp']:
        est = TauStarOpus(sigma=sigma_name)
        taus, _ = est.batch(batch_scores)
        print(f"  {sigma_name:>12s}: taus={taus.tolist()}")

    print("\nDone.")