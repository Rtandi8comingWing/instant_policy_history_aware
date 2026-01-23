# Instant Policy 源码实现指南 (重构版)

本目录包含 Instant Policy 论文的完整 Python 实现，**严格按照论文设计**：

**"Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)**

## 架构概述

与标准 Diffusion Policy 的关键区别：

| 方面 | 标准 Diffusion Policy | Instant Policy (本实现) |
|------|----------------------|-------------------------|
| 表示 | 池化特征 → Transformer | **图结构贯穿始终** |
| 注意力 | 标准自注意力 | **边特征参与注意力 (Eq.3)** |
| 动作表示 | 扁平动作向量 | **幽灵夹爪节点** |
| 时序建模 | 全连接 | **时序边 (t→t+1)** |
| 输出 | 噪声预测 | **几何流向量** |

## 项目结构

```
ip_src/
├── models/
│   ├── graph_diffusion.py     # 主模型 (兼容 .pyi 接口)
│   ├── graph_transformer.py   # 边特征注意力 + 图 Transformer
│   ├── diffusion.py           # 流匹配 (非 DDPM)
│   └── encoders.py            # PointNet++ (冻结)
├── data/
│   ├── graph_builder.py       # 图构建 (核心修改)
│   ├── pseudo_demo.py         # 伪示范生成
│   └── dataset.py             # PyTorch Dataset
├── utils/
│   ├── transforms.py          # SE(3) 变换
│   └── sampling.py            # sample_to_cond_demo
└── training/
    ├── trainer.py             # PyTorch Lightning
    └── losses.py              # 流匹配损失
```

## 核心修改详解

### 1. 图构建 (`graph_builder.py`)

#### NeRF-like 正弦位置编码

```python
def sinusoidal_positional_encoding(coords, num_frequencies=10):
    """
    e = [sin(2^k π p), cos(2^k π p), ...] for k = 0, 1, ..., L-1
    """
    freq_bands = 2.0 ** torch.arange(num_frequencies)
    scaled_coords = coords.unsqueeze(-1) * freq_bands * π
    return torch.cat([scaled_coords.sin(), scaled_coords.cos()], dim=-1)
```

#### 边特征 (相对位置编码)

```python
def compute_edge_features(src_pos, dst_pos, edge_index):
    """
    Δp = p_dst - p_src
    edge_attr = sinusoidal_encoding(Δp) + distance
    """
    delta_p = dst_pos[dst_idx] - src_pos[src_idx]
    return sinusoidal_positional_encoding(delta_p)
```

#### 时序边 (红色线)

```python
# 同一演示内 t → t+1 的 Gripper 节点连接
for wp_idx in range(num_waypoints - 1):
    curr_gripper = demo_gripper_indices[wp_idx]
    next_gripper = demo_gripper_indices[wp_idx + 1]
    temporal_edges.append((curr_gripper[i], next_gripper[i]))
```

**注意**: 不同演示之间**不进行全连接**！

#### 幽灵夹爪节点 (Ghost Gripper)

```python
# 动作表示为未来关键点假设
ghost_positions: [horizon, 6, 3]  # 6个夹爪节点 × 3D位置

# 连接到当前 live gripper
EDGE_TYPE_LIVE_GHOST = ("gripper", "live_to_ghost", "ghost")
EDGE_TYPE_GHOST_GHOST = ("ghost", "ghost_to_ghost", "ghost")  # 时序
```

### 2. 图 Transformer (`graph_transformer.py`)

#### 边特征注意力 (Equation 3)

```python
class EdgeAwareAttention(MessagePassing):
    """
    Attn(Q, K + W_edge·E) · (V + W_edge·E)
    """
    def message(self, q_i, k_j, v_j, edge_feat, ...):
        k_with_edge = k_j + edge_feat  # K + W_5·E
        v_with_edge = v_j + edge_feat  # V + W_5·E
        attn = (q_i * k_with_edge).sum() * scale
        return softmax(attn) * v_with_edge
```

#### ActionDecoder ψ(·) - 图 Transformer

```python
class ActionDecoder(nn.Module):
    """
    输入: 完整 HeteroData 图 (包含 ghost 节点)
    过程: 消息传递更新 ghost 节点特征
    输出: 几何流向量 [∇p_trans, ∇p_rot] (6D)
    """
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        out_dict = self.transformer(x_dict, edge_index_dict, edge_attr_dict)
        ghost_features = out_dict["ghost"]
        
        flow = self.flow_head(ghost_features)  # [num_ghost, 6]
        gripper = self.gripper_head(ghost_features)  # [num_ghost, 1]
        
        return flow, gripper
```

### 3. 流匹配 (`diffusion.py`)

```python
class FlowMatchingScheduler:
    """
    Rectified Flow: x_t = (1-t)·x_0 + t·x_1
    Velocity: v = x_1 - x_0
    """
    def step(self, velocity, x_t, t, dt):
        return x_t - dt * velocity
```

输出是 **3D 位置流向量**，而非 9D 扁平噪声。

## 使用方式

```python
from ip_src import GraphDiffusion, sample_to_cond_demo

# 加载模型 (接口与原始 .pyi 兼容)
model = GraphDiffusion.load_from_checkpoint('./checkpoints/model.pt', device='cuda')
model.set_num_demos(2)
model.set_num_diffusion_steps(4)

# 预测动作
actions, grips = model.predict_actions(full_sample)
# actions: [horizon, 4, 4] - 相对变换矩阵
# grips: [horizon, 1] - 夹爪命令
```

## 与论文的对应关系

| 论文内容 | 实现位置 |
|---------|---------|
| Section 3.2 Graph Representation | `graph_builder.py` |
| Appendix A Position Encoding | `sinusoidal_positional_encoding()` |
| Equation 3 Attention | `EdgeAwareAttention` |
| Figure 2 Temporal Edges | `EDGE_TYPE_TEMPORAL` |
| Section 3.2 Action Nodes | Ghost gripper nodes |
| Appendix C Networks | `LocalGraphEncoder`, `ContextAggregator`, `ActionDecoder` |

## 重要约束

1. **几何编码器必须冻结** - 论文 Appendix H 明确指出
2. **异构图设计** - 不同节点/边类型使用独立权重
3. **时序边** - 仅在同一演示内连接，不跨演示全连接
4. **流匹配输出** - 3D 几何流向量，而非扁平噪声

## 引用

```bibtex
@inproceedings{vosylius2025instant,
  title={Instant Policy: In-Context Imitation Learning via Graph Diffusion},
  author={Vosylius, Vitalis and Johns, Edward},
  booktitle={Proceedings of ICLR},
  year={2025}
}
```
