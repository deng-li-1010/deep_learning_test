# Bahdanau注意力（动手学深度学习10.4节）超通俗完整讲解
## 一、先搞懂：Bahdanau注意力解决了什么痛点？
### 1. 原始Seq2Seq（无注意力）的致命缺陷
机器翻译原始编码器-解码器架构：
1. **编码器（RNN/GRU/LSTM）**：把整段源文本（英文）所有单词压缩成**唯一固定长度向量c**（只用最后一步隐状态）；
2. **解码器**：每一步生成目标单词（法语）时，全程都用同一个c。

**问题**：
- 短句还好，长句子时，唯一的c会丢失大量细节；
- 生成每个目标词时，平等看待所有源单词，分不清哪些源词和当前翻译词强相关、哪些无关。
  举例：英文`I drink orange juice`翻译成法语，生成`jus（果汁）`时，模型本该重点关注`orange juice`，但原始模型强行同等看待`I / drink / orange / juice`，翻译容易出错。

### 2. Bahdanau注意力核心思想（2014，NLP第一个实用注意力）
**动态上下文向量**：解码器每生成一个新词，**单独计算一个专属上下文$c_{t'}$**，只加权聚合和当前翻译强相关的源单词特征，无关单词权重压低。
- 对齐思想：自动学习目标词和源词的对应关系（翻译对齐）；
- 打分方式：**加性注意力（Additive Attention）**，Bahdanau专属打分函数。

## 二、核心公式完整拆解（式10.4.1）
$$
c_{t'} = \sum_{t=1}^T \alpha(s_{t'-1}, h_t) h_t
$$
### 逐个变量通俗解释
1. $t'$：解码器当前时间步（正在生成第$t'$个目标词）
2. $t'-1$：解码器上一步（生成前一个词后的隐状态）
3. $s_{t'-1}$：**查询Query**，解码器上一步顶层隐状态，代表“我现在要翻译什么，我需要找源句里哪些词”
4. $T$：源文本总词数（输入序列长度）
5. $h_t$：编码器第$t$步输出隐状态，**同时是Key键、Value值**
    - Key：用来和Query打分，衡量相关性；
    - Value：真正拿来加权求和的特征；
6. $\alpha(s_{t'-1},h_t)$：**注意力权重**，0~1之间，所有权重总和=1，由**加性注意力打分函数**算出：
   $\alpha$越大 → 第$t$个源单词对当前翻译越重要；
7. $c_{t'}$：**当前步专属上下文向量**，所有源隐状态$h_t$按权重加权求和，送入解码器RNN参与生成下一个词。

### 数据流一句话总结
拿解码器上一步状态当“查询”，去和每一个源单词特征打分，得到权重，用权重把源单词特征加权融合，得到本次翻译专用上下文。

## 三、模型整体架构（通俗版）
### 1. 编码器（不变，和原始Seq2Seq一致）
输入源文本 → 嵌入层 → 多层GRU/LSTM → 输出**每一步全部隐状态$h_1,h_2...h_T$**（Bahdanau必须保存全部，原始模型只留最后一个）。

### 2. 带Bahdanau注意力的解码器（核心改动）
每一步解码循环流程：
1. 取出解码器上一步顶层隐状态$s_{t'-1}$，升维变成Query；
2. Query和编码器全部隐状态（Key/Value）送入加性注意力层；
3. 注意力层输出加权求和后的上下文$c_{t'}$；
4. 当前目标词嵌入向量 和 $c_{t'}$ 在特征维度拼接；
5. 拼接后的向量送入GRU，更新解码器隐状态；
6. GRU输出经过全连接层，预测下一个目标单词。

关键区别：原始解码器输入只有词嵌入；Bahdanau解码器输入=词嵌入+动态上下文。

## 四、PyTorch代码逐行深度解析（仅讲解pytorch版本）
### 4.1 基础父类 AttentionDecoder
```python
class AttentionDecoder(d2l.Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
    @property
    def attention_weights(self):
        raise NotImplementedError
```
1. **作用**：统一带注意力解码器的接口规范；
2. `attention_weights`属性：强制子类实现，用于提取、可视化每一步注意力权重热力图；
3. `raise NotImplementedError`：子类不重写调用会报错，保证代码规范。

### 4.2 核心类 Seq2SeqAttentionDecoder（Bahdanau解码器主体）
完整代码分段拆解，参数、中间张量shape、输入输出全说明
```python
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        # 1. 加性注意力层（Bahdanau专用打分）
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        # 2. 目标词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 3. GRU循环层：输入维度=词嵌入+上下文向量维度
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        # 4. 输出全连接层，映射到目标词表
        self.dense = nn.Linear(num_hiddens, vocab_size)
```
#### __init__初始化参数详解
| 参数 | 含义 | 示例入参 |
|------|------|----------|
| vocab_size | 目标语言词表总大小 | 机器翻译法语词表1800 |
| embed_size | 词嵌入向量维度 | 32 |
| num_hiddens | GRU隐状态维度、注意力中间向量维度 | 32 |
| num_layers | GRU堆叠层数 | 2 |
| dropout | 丢弃概率，防止过拟合 | 0.1 |

#### 内部组件说明
1. `self.attention = d2l.AdditiveAttention`
    - Bahdanau核心：加性注意力；
    - 四个入参：query维度、key维度、value维度、dropout；
    - 本模型中query/key/value维度统一=num_hiddens；
    - 输出：加权后的上下文context，同时保存注意力权重。
2. `nn.Embedding`：将目标单词数字id转为稠密向量。
3. `nn.GRU`：输入维度`embed_size + num_hiddens`——因为每一步输入是**词嵌入拼接上下文向量**。
4. `nn.Linear`：GRU输出隐状态映射到词表维度，预测每个单词概率。

---
#### 方法1：init_state 初始化解码器状态
```python
def init_state(self, enc_outputs, enc_valid_lens, *args):
    outputs, hidden_state = enc_outputs
    return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)
```
##### 输入入参
1. `enc_outputs`：编码器输出元组`(outputs, hidden_state)`
    - `outputs` shape：`(num_steps, batch_size, num_hiddens)`，编码器每一步全部隐状态；
    - `hidden_state` shape：`(num_layers, batch_size, num_hiddens)`，编码器最终隐状态，用来初始化解码器GRU；
2. `enc_valid_lens`：批量内每条源句子真实有效长度，屏蔽padding填充词，注意力不给填充分配权重。

##### 中间处理：permute(1,0,2)
交换维度：`(num_steps, batch, hidden) → (batch, num_steps, hidden)`
目的：后续注意力计算标准输入shape（batch优先）。

##### 返回值（解码器state三元组）
```
(enc_outputs, hidden_state, enc_valid_lens)
```
1. enc_outputs：`(batch, src_len, num_hiddens)` 全部源隐状态（Key/Value）；
2. hidden_state：编码器顶层隐状态，作为解码器GRU初始状态；
3. enc_valid_lens：有效长度，过滤padding。

##### 示例入参出参
输入编码器输出：`outputs=(7,4,16), hidden=(2,4,16)`
permute后：`outputs=(4,7,16)`
返回state：`[(4,7,16), (2,4,16), None]`（测试时无padding填None）

---
#### 方法2：forward 前向传播（核心解码逻辑）
```python
def forward(self, X, state):
    enc_outputs, hidden_state, enc_valid_lens = state
    # 词嵌入+维度调换
    X = self.embedding(X).permute(1, 0, 2)
    outputs, self._attention_weights = [], []
    # 逐词循环解码（目标序列逐时间步）
    for x in X:
        # 构造Query：解码器上一步顶层隐状态扩展维度
        query = torch.unsqueeze(hidden_state[-1], dim=1)
        # 注意力计算上下文
        context = self.attention(
            query, enc_outputs, enc_outputs, enc_valid_lens)
        # 词向量升维，和上下文拼接
        x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
        # GRU前向
        out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
        outputs.append(out)
        self._attention_weights.append(self.attention.attention_weights)
    # 全部解码完成，拼接输出+全连接预测
    outputs = self.dense(torch.cat(outputs, dim=0))
    return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                      enc_valid_lens]
```
##### 输入
1. X：目标输入序列，shape `(batch_size, num_steps)`，单词id；
2. state：`init_state`返回的三元组。

##### 逐行拆解中间数据流转（附shape示例：batch=4，src_len=7，embed=8，hidden=16）
1. `X = self.embedding(X).permute(1,0,2)`
    - 嵌入后：`(4,7,8)`；permute后：`(7,4,8)`（时间步放第一维，方便循环逐词取）
2. 循环`for x in X`：x代表单步目标词向量，shape `(4,8)`
3. `query = torch.unsqueeze(hidden_state[-1], dim=1)`
    - `hidden_state[-1]`：解码器最后一层隐状态 `(4,16)`；
    - unsqueeze新增维度1，变成`(4,1,16)`，符合注意力Query标准shape`(batch, query_num, dim)`。
4. `context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)`
   注意力输入格式：`attention(query, key, value, valid_len)`
    - query：`(4,1,16)`；key/value：`(4,7,16)`；
    - 输出context：`(4,1,16)`，当前步专属加权上下文向量。
5. `x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)`
    - x升维：`(4,1,8)`；context `(4,1,16)`；
    - 最后一维拼接：`(4,1, 8+16=24)`，这就是GRU单步输入。
6. `out, hidden_state = self.rnn(x.permute(1,0,2), hidden_state)`
   permute调换维度：`(1,4,24)`（GRU默认time_major）；
    - out：单步输出`(1,4,16)`；
    - hidden_state：更新后的多层隐状态`(2,4,16)`。
7. 保存输出与权重：`outputs`存每一步GRU输出；`_attention_weights`存每一步α权重，用于可视化。
8. 循环结束，拼接所有时间步输出：`torch.cat(outputs, dim=0) → (7,4,16)`
9. `self.dense()`映射词表：`(7,4,10)`（测试词表大小10）
10. 返回调换维度：`permute(1,0,2) → (4,7,10)`（batch放第一维，标准输出格式）

##### 返回值
1. outputs：`(batch, tgt_len, vocab_size)` 每个词的预测分数；
2. 新state三元组：保留编码器全部隐状态、更新后的解码器隐状态、有效长度。

---
#### 方法3：attention_weights 属性
```python
@property
def attention_weights(self):
    return self._attention_weights
```
- 装饰器@property：像访问属性一样调用，不用加括号；
- 返回列表：列表长度=目标序列长度，每个元素是单步注意力权重矩阵`(batch,1,src_len)`；
- 用途：训练完成后取出，绘制热力图，直观看到每个目标词对齐了哪些源单词。

### 4.3 模型测试代码（入参出参完整示例）
```python
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)  # 输入：4条样本，源序列长度7
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
```
#### 输出shape解释
```
(torch.Size([4, 7, 10]), 3, torch.Size([4, 7, 16]), 2, torch.Size([4, 16]))
```
1. `[4,7,10]`：模型输出，batch=4，目标长度7，词表10；
2. `3`：state是三元组；
3. `[4,7,16]`：编码器全部隐状态`(batch,src_len,hidden)`；
4. `2`：GRU两层；
5. `[4,16]`：解码器单层隐状态shape。

## 五、训练与预测流程
### 1. 训练逻辑
1. 加载机器翻译英法数据集，构造迭代器；
2. 实例化编码器+Bahdanau注意力解码器，封装为EncoderDecoder完整模型；
3. 交叉熵损失、Adam优化器，迭代250轮；
4. 相比无注意力Seq2Seq，训练更慢：每一步解码都要遍历全部源序列做注意力打分，计算量更大。

### 2. 预测+BLEU评估
1. 输入英文句子，模型逐词生成法语；
2. 提取`attention_weights`，拼接成四维张量`(1,1,tgt_len,src_len)`；
3. 绘制热力图：X轴源单词位置，Y轴目标单词位置，色块深浅代表注意力权重大小；
4. 现象：能清晰看到对齐关系，比如生成`va`时，注意力集中在源词`go`。

## 六、章节小结（提炼核心考点）
1. **原始Seq2Seq缺陷**：全局唯一上下文向量，长句信息丢失；
2. **Bahdanau核心创新**：每一步解码生成动态专属上下文$c_{t'}$，基于加性注意力加权源隐状态；
3. **QKV分配**：Query=解码器上一步隐状态，Key=Value=编码器全部隐状态；
4. **解码器改动关键点**：GRU输入=目标词嵌入 + 注意力上下文拼接；
5. **可解释性**：注意力权重热力图可视化源-目标词对齐关系；
6. **打分函数**：固定使用加性注意力，区别于后续Transformer缩放点积注意力。

## 七、课后练习通俗解读
1. **LSTM替换GRU**：只需将代码中nn.GRU改为nn.LSTM，同时适配LSTM双层隐状态（h,c），逻辑不变，精度略有提升，训练速度更慢；
2. **替换为缩放点积注意力**：
    - 优点：矩阵乘法并行度更高，训练速度更快；
    - 缺点：要求query/key维度严格相等，加性注意力无维度限制；
    - 改动：将`d2l.AdditiveAttention`替换为`d2l.DotProductAttention`，其余代码几乎不用修改。