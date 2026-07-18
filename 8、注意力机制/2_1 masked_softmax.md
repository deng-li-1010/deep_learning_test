



# masked_softmax 函数详细分析

## 函数定义回顾

```python
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

---

## 结合具体例子分析

假设我们有以下输入（延续之前的注意力机制示例）：

```python
import torch
from d2l import torch as d2l

# 模拟注意力分数矩阵 scores
# shape: (batch_size=2, q_len=3, k_len=4)
X = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0],   # query 0 与所有 key 的分数
     [5.0, 6.0, 7.0, 8.0],   # query 1 与所有 key 的分数  
     [9.0, 10.0, 11.0, 12.0]],# query 2 与所有 key 的分数
    [[13.0, 14.0, 15.0, 16.0], # 第二个样本
     [17.0, 18.0, 19.0, 20.0],
     [21.0, 22.0, 23.0, 24.0]]
])

# valid_lens: 第一个样本有效长度3，第二个样本有效长度2
valid_lens = torch.tensor([3, 2])
```

---

## 执行流程分步解析

### 第一步：判断 valid_lens 是否为 None

```python
if valid_lens is None:
    return nn.functional.softmax(X, dim=-1)
```

- 如果 `valid_lens` 为 None，直接对最后一个维度做 softmax
- 本例中 `valid_lens = [3, 2]`，不为 None，进入 else 分支

### 第二步：获取原始形状并处理 valid_lens

```python
shape = X.shape  # shape = (2, 3, 4)

if valid_lens.dim() == 1:
    # valid_lens 是 1D 张量: [3, 2]
    valid_lens = torch.repeat_interleave(valid_lens, shape[1])
```

**关键操作：`repeat_interleave`**

| 输入 | 操作 | 输出 |
|------|------|------|
| `valid_lens = [3, 2]` | 每个元素重复 `shape[1]=3` 次 | `[3, 3, 3, 2, 2, 2]` |

**为什么要重复？**

原始输入 `X` 的形状是 `(2, 3, 4)`，其中：
- batch_size = 2，每个样本有 3 个 query
- 如果每个样本只有一个统一的有效长度，需要将其扩展为每个 query 对应一个长度值
- 扩展后 valid_lens 的长度 = `batch_size × q_len = 2 × 3 = 6`

### 第三步：展平并应用 mask

```python
# 将 X 展平为 (batch_size * q_len, k_len) = (6, 4)
X_flattened = X.reshape(-1, shape[-1])

# 应用 sequence_mask
# 将超过 valid_len 的位置设为 -1e6
X_masked = d2l.sequence_mask(X_flattened, valid_lens, value=-1e6)
```

**展平后的 X 和 mask 效果：**

| 行（query） | 原始值 | valid_len | Mask 后的值 |
|-------------|--------|-----------|-------------|
| 样本0, q0 | [1, 2, 3, 4] | 3 | [1, 2, 3, **-1e6**] |
| 样本0, q1 | [5, 6, 7, 8] | 3 | [5, 6, 7, **-1e6**] |
| 样本0, q2 | [9, 10, 11, 12] | 3 | [9, 10, 11, **-1e6**] |
| 样本1, q0 | [13, 14, 15, 16] | 2 | [13, 14, **-1e6**, **-1e6**] |
| 样本1, q1 | [17, 18, 19, 20] | 2 | [17, 18, **-1e6**, **-1e6**] |
| 样本1, q2 | [21, 22, 23, 24] | 2 | [21, 22, **-1e6**, **-1e6**] |

### 第四步：恢复形状并计算 softmax

```python
# 恢复为原始形状 (2, 3, 4)
X = X_masked.reshape(shape)

# 对最后一个维度做 softmax
return nn.functional.softmax(X, dim=-1)
```

**softmax 计算原理：**

对于 `[1, 2, 3, -1e6]`：
- exp(1) ≈ 2.718, exp(2) ≈ 7.389, exp(3) ≈ 20.085, exp(-1e6) ≈ 0
- sum ≈ 30.192
- softmax ≈ [0.09, 0.245, 0.665, **0**]

padding 位置的权重被压制到几乎为 0。

---

## 两种 valid_lens 格式对比

### 格式一：1D valid_lens（每个样本一个长度）

```python
valid_lens = torch.tensor([3, 2])  # shape: (2,)
# 扩展后: [3, 3, 3, 2, 2, 2]
```

**适用场景**：所有 query 共享相同的有效长度约束

### 格式二：2D valid_lens（每个 query 独立长度）

```python
valid_lens = torch.tensor([
    [3, 2, 1],  # 样本0：q0看前3个，q1看前2个，q2看前1个
    [2, 3, 4]   # 样本1：q0看前2个，q1看前3个，q2看前4个
])  # shape: (2, 3)
# 展平后: [3, 2, 1, 2, 3, 4]
```

**适用场景**：不同 query 需要关注的 key 范围不同（如因果注意力）

---

## 完整执行流程图

```
输入: X(2,3,4), valid_lens[3,2]
       │
       ▼
┌─────────────────────────────┐
│ valid_lens.dim() == 1 ?     │──Yes──▶ repeat_interleave([3,2], 3)
└─────────────────────────────┘              │
       │No                                   ▼
       ▼                              valid_lens = [3,3,3,2,2,2]
  reshape(-1)
       │
       └──────┬──────────────────────────────┘
              ▼
    X.reshape(-1, 4) → (6, 4)
              │
              ▼
    d2l.sequence_mask(X, valid_lens, -1e6)
              │
              ▼
    X.reshape(2, 3, 4) → 恢复形状
              │
              ▼
    softmax(X, dim=-1) → 输出注意力权重
```

---

## d2l.sequence_mask 函数说明

虽然代码中调用的是 d2l 库的函数，但它的核心逻辑等价于：

```python
def sequence_mask(X, valid_len, value=0):
    """将 X 中超过 valid_len 的位置设为 value"""
    max_len = X.size(1)
    mask = torch.arange(max_len, dtype=torch.float32, 
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
```

**工作原理**：
1. 创建一个从 0 到 max_len-1 的序列 `[0, 1, 2, 3]`
2. 与每个 valid_len 比较，生成布尔 mask
3. 将 mask 为 False 的位置设为指定 value（此处为 `-1e6`）

---

## 总结

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 判断 valid_lens | 处理无需 mask 的情况 |
| 2 | repeat_interleave/reshape | 将 valid_lens 适配到展平后的 X |
| 3 | reshape X | 将多维输入展平为 (N, k_len) |
| 4 | sequence_mask | 将 padding 位置设为极小值 |
| 5 | softmax | 计算注意力权重（padding 位置趋近于 0） |

这个函数的设计巧妙之处在于：
- **通用性强**：支持 1D 和 2D 两种 valid_lens 格式
- **效率高**：通过展平操作统一处理，避免复杂的条件分支
- **数值稳定**：使用 `-1e6` 确保 softmax 后 padding 权重几乎为 0
        