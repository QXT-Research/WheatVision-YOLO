# -*- coding: utf-8 -*-
"""YOLO-CAST 自定义模块，严格按论文式(1)-(14)实现。

C2f-CILA  : Backbone 特征提取（式 1-7）
GS-BiFPN  : 轻量化双向多尺度融合 Neck（式 8-11），GSConv + 归一化加权融合
P2-TSCHead: stride-4 高分辨率分支 + 任务解耦上下文头（式 12-14）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.head import Detect


# ---------------------------------------------------------------- GSConv (式 8)
class GSConv(nn.Module):
    """GSConv：1x1 逐点卷积 + 3x3 深度卷积 + 通道重排（半通道拼接后 Shuffle）。"""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, s)
        self.cv2 = DWConv(c_, c_, k, s)
        self.out_channels = c2

    def forward(self, x):
        u = self.cv1(x)
        v = self.cv2(u)
        y = torch.cat([u, v], 1)  # (B, c2, H, W)
        b, c, h, w = y.shape
        y = y.view(b, 2, c // 2, h, w).transpose(1, 2).contiguous().view(b, c, h, w)  # Shuffle
        return y


# ---------------------------------------------------------------- CILA (式 2-7)
class CILA(nn.Module):
    """Context-Enhanced and Illumination-modulated Local-Aggregation 单元。

    输入为跨阶段聚合得到的共享表示 Z（式 1），输出与 Z 同形状。
    """

    def __init__(self, c):
        super().__init__()
        self.c = c
        # 局部纹理分支 + 多尺度上下文分支（式 2）
        self.loc = DWConv(c, c, 3)
        self.d5 = DWConv(c, c, 5)
        self.d7a = DWConv(c, c, (7, 1))
        self.d7b = DWConv(c, c, (1, 7))
        self.d11a = DWConv(c, c, (11, 1))
        self.d11b = DWConv(c, c, (1, 11))
        # 尺度自适应选择（式 3）
        r = max(c // 4, 16)
        self.sa_fc1 = nn.Linear(3 * c, r)
        self.sa_fc2 = nn.Linear(r, 3)
        # 颜色统计门控（式 4）
        self.cg_fc1 = nn.Linear(2 * c, r)
        self.cg_fc2 = nn.Linear(r, c)
        # 方向位置约束（式 5，坐标注意力形式）
        self.dir_conv = Conv(c, 2 * c, 1)
        self.dir_h = nn.Conv2d(c, c, 1)
        self.dir_w = nn.Conv2d(c, c, 1)
        # 光照门控（式 6）
        self.ill_conv = nn.Conv2d(3, 1, 7, padding=3, bias=True)
        # 输出融合（式 7）
        self.proj = Conv(c, c, 1)
        self.lam = nn.Parameter(torch.zeros(2))       # λ_loc, λ_ctx → Softmax
        self.gamma = nn.Parameter(torch.tensor(1e-2))  # 近零初始化的残差缩放

    def forward(self, z):
        b, c, h, w = z.shape
        # 式(2)
        f_loc = self.loc(z)
        f5 = self.d5(z)
        f7 = self.d7b(self.d7a(z))
        f11 = self.d11b(self.d11a(z))
        # 式(3) 尺度描述子 + Softmax 自适应加权
        s = F.adaptive_avg_pool2d(torch.cat([f5, f7, f11], 1), 1).flatten(1)
        a = torch.softmax(self.sa_fc2(F.silu(self.sa_fc1(s))), dim=1)
        f_ctx = (a[:, 0:1, None, None] * f5
                 + a[:, 1:2, None, None] * f7
                 + a[:, 2:3, None, None] * f11)
        # 式(4) 通道均值/标准差 → 颜色统计门控
        mu = z.mean(dim=(2, 3))
        nu = z.var(dim=(2, 3), unbiased=False).clamp_min(1e-6).sqrt()
        a_c = torch.sigmoid(self.cg_fc2(F.silu(self.cg_fc1(torch.cat([mu, nu], 1)))))
        a_c = a_c.unsqueeze(-1).unsqueeze(-1)
        # 式(5) 方向位置约束（坐标注意力：沿空间维拼接，1x1 卷积不跨空间位置，H≠W 安全）
        h_desc = z.mean(dim=3, keepdim=True)                     # B,C,H,1
        w_desc = z.mean(dim=2, keepdim=True).transpose(2, 3)     # B,C,W,1
        q = self.dir_conv(torch.cat([h_desc, w_desc], dim=2))    # B,2C,H+W,1
        qh = q[:, :c, :h]                                        # B,C,H,1
        qw = q[:, c:, h:]                                        # B,C,W,1
        a_h = torch.sigmoid(self.dir_h(qh))                      # B,C,H,1
        a_w = torch.sigmoid(self.dir_w(qw.transpose(2, 3)))      # B,C,1,W
        # 式(6) 光照门控
        m = z.mean(dim=1, keepdim=True)
        d_loc = (m - F.avg_pool2d(m, 3, stride=1, padding=1)).abs()
        g_ill = torch.sigmoid(self.ill_conv(torch.cat([m, z.amax(dim=1, keepdim=True), d_loc], 1)))
        # 式(7) 残差融合
        lam = torch.softmax(self.lam, dim=0)
        fused = lam[0] * f_loc + lam[1] * (f_ctx * a_c * a_h * a_w)
        return z + self.gamma * self.proj(fused * (1.0 + g_ill))


# ---------------------------------------------------------------- C2f-CILA (式 1)
class C2f_CILA(nn.Module):
    """C2f 骨架 + CILA 单元：跨阶段聚合得共享表示 Z（式 1），再经 CILA 输出。"""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut=True) for _ in range(n))
        self.cila = CILA(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        z = self.cv2(torch.cat(y, 1))
        return self.cila(z)


# ---------------------------------------------------------------- GSFuse (式 9-11)
class GSFuse(nn.Module):
    """BiFPN 风格融合节点：非负归一化可学习权重 + GSConv 精炼。

    aligns 中每个元素指定对齐方式：
        'none'  —— 分辨率已对齐，直通
        'up2'   —— 最近邻上采样 2 倍（来自更深层）
        'down2' —— 步长 2 卷积下采样（来自更浅层）
    所有输入通道数必须等于 c2（由 YAML 中的前置 1x1 Conv 统一）。
    """

    def __init__(self, c_in_list, c2, aligns=('none', 'up2')):
        super().__init__()
        self.n = len(aligns)
        assert len(c_in_list) == self.n, f"GSFuse: {len(c_in_list)} inputs vs {self.n} align modes"
        self.ups = nn.ModuleList(
            nn.Upsample(scale_factor=2, mode='nearest') if a == 'up2' else nn.Identity() for a in aligns
        )
        self.downs = nn.ModuleList(
            Conv(c2, c2, 3, 2) if a == 'down2' else nn.Identity() for a in aligns
        )
        self.w = nn.Parameter(torch.ones(self.n))
        self.refine = GSConv(c2, c2)

    def forward(self, x):
        w = F.relu(self.w)
        w = w / (w.sum() + 1e-4)  # 式(9)
        out = sum(wi * d(u(xi)) for wi, xi, u, d in zip(w, x, self.ups, self.downs))
        return self.refine(out)   # Ψ_o 精炼（式 10）


# ---------------------------------------------------------------- P2-TSCHead (式 12-14)
class P2TSCDetect(Detect):
    """任务解耦上下文检测头：P2 高分辨率分支 + 分类/回归异构上下文注入。

    前向时把 4 层（P2/P3/P4/P5）特征先 1x1 投影到统一宽度 W（式 12），
    分类支拼接相邻深层语义（式 13），回归支拼接相邻浅层几何细节（式 14），
    再分别送入 Detect 的 cv3（分类）与 cv2（回归）。
    """

    def __init__(self, nc, w, reg_max=16, end2end=False, ch=()):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=(w,) * 4)
        self.q = nn.ModuleList(Conv(c, w, 1) for c in ch)              # 式(12)
        self.phi_cls = nn.ModuleList(Conv(2 * w, w, 1) for _ in range(3))  # P2,P3,P4 分类上下文
        self.phi_reg = nn.ModuleList(Conv(2 * w, w, 1) for _ in range(3))  # P3,P4,P5 回归上下文
        self.down_reg = nn.ModuleList(Conv(w, w, 3, 2) for _ in range(3))

    def _tsc_feats(self, x):
        """按式(13)(14)构造任务特异特征，返回 (分类特征列表, 回归特征列表)。"""
        q = [self.q[i](x[i]) for i in range(4)]
        f_cls, f_reg = [], []
        for l in range(4):
            # 分类：拼接相邻深层语义
            if l < 3:
                c_in = torch.cat([q[l], F.interpolate(q[l + 1], scale_factor=2, mode='nearest')], 1)
                f_cls.append(self.phi_cls[l](c_in))
            else:
                f_cls.append(q[3])
            # 回归：拼接相邻浅层细节
            if l == 0:
                f_reg.append(q[0])
            else:
                r_in = torch.cat([q[l], self.down_reg[l - 1](q[l - 1])], 1)
                f_reg.append(self.phi_reg[l - 1](r_in))
        return f_cls, f_reg

    def forward(self, x):
        f_cls, f_reg = self._tsc_feats(x)
        bs = f_cls[0].shape[0]
        preds = {
            "boxes": torch.cat(
                [self.cv2[i](f_reg[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1),
            "scores": torch.cat(
                [self.cv3[i](f_cls[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1),
            "feats": f_cls,
        }
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)
