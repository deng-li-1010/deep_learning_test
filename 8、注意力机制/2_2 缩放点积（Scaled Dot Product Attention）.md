



让我先查看 `test.py` 文件中 `DotProductAttention` 类的完整实现。



toolName: view_files

status: success


filePath: /Users/bytedance/python_project/JupyterProject/test.py

# DotProductAttention 类解析

这是一个**缩放点积注意力（Scaled Dot-Product Attention）**的实现，是 Transformer 架构中最核心的组件之一。

---

## 整体结构

```python
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        # 初始化 dropout 层
    def forward(self, queries, keys, values, valid_lens=None):
        # 计算注意力并返回加权后的 value
```

---

## 关键参数说明

| 参数 | 维度 | 含义 |
|------|------|------|
| `queries` | `(batch_size, q_len, d)` | 查询向量，每个查询维度为 d |
| `keys` | `(batch_size, k_len, d)` | 键向量，每个键维度为 d |
| `values` | `(batch_size, v_len, d_v)` | 值向量，通常 k_len = v_len |
| `valid_lens` | 1D 或 2D | 指定每个序列的有效长度，用于掩蔽 padding |

---

## Forward 方法核心逻辑

### 第一步：计算注意力分数

```python
d = queries.shape[-1]  # 获取查询向量的维度
scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
```

**计算过程**：
- `keys.transpose(1,2)`：将 keys 的形状从 `(B, k_len, d)` 转为 `(B, d, k_len)`
- `torch.bmm`：批量矩阵乘法，计算每个 query 与所有 key 的点积
- 输出 `scores` 的形状为 `(batch_size, q_len, k_len)`

**为什么要除以 `√d`？**

这是缩放因子的作用：
- 当维度 `d` 很大时，点积的结果会很大，导致 softmax 的梯度趋近于 0（梯度消失）
- 除以 `√d` 可以将点积的方差归一化到 1，保持数值稳定性

### 第二步：应用 Masked Softmax

```python
self.attention_weights = masked_softmax(scores, valid_lens)
```

将分数转换为注意力权重（概率分布），同时处理 padding 部分：
- 如果 `valid_lens` 不为 None，padding 位置的分数会被设为 `-1e6`，softmax 后趋近于 0
- 这样模型就不会关注 padding 的无效信息

### 第三步：计算输出

```python
return torch.bmm(self.dropout(self.attention_weights), values)
```

- 将注意力权重与 values 进行加权求和
- 通过 dropout 正则化，防止过拟合

---

## Masked Softmax 辅助函数解析

```python
def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 1D valid_lens: 每个样本一个长度值
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 2D valid_lens: 每个样本的每个 query 都有一个长度值
            valid_lens = valid_lens.reshape(-1)
        # 使用 sequence_mask 将无效位置设为 -1e6
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

**设计意图**：支持两种 `valid_lens` 格式：
- **1D**：`(batch_size,)` - 每个样本有一个统一的有效长度
- **2D**：`(batch_size, q_len)` - 每个 query 可以有不同的有效长度

---

## 注意力机制原理图示

```
Queries (B, q_len, d)  ×  Keys^T (B, d, k_len)
         ↓ (点积 + 缩放)
     Scores (B, q_len, k_len)
         ↓ (Masked Softmax)
 Attention Weights (B, q_len, k_len)
         ↓ (加权求和)
    Output (B, q_len, d_v)
```

---

## 输入输出示例

```python
# 输入
batch_size, q_len, k_len, d = 2, 3, 4, 8
queries = torch.randn(batch_size, q_len, d)
keys = torch.randn(batch_size, k_len, d)
values = torch.randn(batch_size, k_len, d)
valid_lens = torch.tensor([3, 2])  # 第一个样本有效长度3，第二个样本有效长度2

# 初始化并前向传播
attention = DotProductAttention(dropout=0.1)
output = attention(queries, keys, values, valid_lens)

# 输出形状
print(output.shape)  # (2, 3, 8) - 与 queries 形状一致
print(attention.attention_weights.shape)  # (2, 3, 4) - 注意力权重矩阵
```

---

## 总结

| 特性 | 说明 |
|------|------|
| **缩放因子** | `1/√d` 防止梯度消失 |
| **Masking** | 支持动态序列长度，忽略 padding |
| **Dropout** | 正则化提高泛化能力 |
| **批处理** | 使用 `bmm` 支持批量计算 |

这是标准 Transformer 注意力机制的简洁实现，被广泛应用于 NLP、CV 等领域的序列建模任务中。
        