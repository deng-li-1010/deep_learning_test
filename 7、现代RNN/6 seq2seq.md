# 《动手学深度学习》9.7 序列到序列Seq2Seq 通俗完整讲解（仅PyTorch代码详解）
## 一、Seq2Seq 是干嘛的？通俗理解
日常很多任务：**输入一段不定长文本，输出另一段不定长文本**，比如机器翻译、对话机器人、文本摘要。
传统RNN只能输入输出等长序列，解决不了翻译（英文短句长、法语长句短）这种输入输出长度不一样的场景。
Seq2Seq 核心架构：**编码器Encoder + 解码器Decoder**，两个RNN（这里用GRU）配合工作：
1. **编码器**：吞掉整段输入句子，把全部信息压缩成一个固定长度的向量（上下文向量c）；
2. **解码器**：拿着上下文向量c，一个词一个词逐词生成目标句子，直到生成结束符`<eos>`停止。

### 特殊标记（必懂）
- `<bos>`：Begin of Sentence，句子起始符，解码器生成的第一个输入；
- `<eos>`：End of Sentence，句子结束符，预测出这个词就停止生成；
- `<pad>`：填充符，短句子末尾补`<pad>`，统一批次内句子长度，方便批量计算；训练时要屏蔽填充符损失。

## 二、核心数学公式逐行拆解（变量全解释）
### 1. 编码器隐状态更新公式（9.7.1）
$$h_t = f(x_t, h_{t-1})$$
- $t$：输入序列第$t$个词元；
- $x_t$：第$t$个词的词嵌入向量；
- $h_{t-1}$：上一个时间步的隐藏状态；
- $f$：GRU/LSTM循环变换函数；
- $h_t$：当前时间步隐藏状态，存储前$t$个单词的语义信息。

### 2. 上下文向量公式（9.7.2）
$$c = q(h_1,...,h_T)$$
- $T$：输入句子总长度；
- $q$：聚合函数，本文最简单实现：直接取最后一步隐状态 $c=h_T$；
- $c$：上下文向量，整句输入的全部信息压缩载体，传给解码器。

### 3. 解码器隐状态更新公式（9.7.3）
$$s_{t'} = g(y_{t'-1}, c, s_{t'-1})$$
- $t'$：输出序列第$t'$个词元；
- $y_{t'-1}$：上一步生成的单词向量；
- $c$：编码器传来的全局上下文向量；
- $s_{t'-1}$：解码器上一步隐藏状态；
- $g$：解码器GRU变换函数；
- $s_{t'}$：解码器当前隐状态，融合原文信息+已生成译文信息。

### 4. BLEU评估指标公式（9.7.4）
$$\text{BLEU} = \exp\left(\min\left(0,1-\frac{\text{len}_\text{label}}{\text{len}_\text{pred}}\right)\right) \cdot \prod_{n=1}^k p_n^{\frac{1}{2^n}}$$
1. 第一部分：短句惩罚因子BP
   $\exp\left(\min\left(0,1-\frac{\text{len}_\text{label}}{\text{len}_\text{pred}}\right)\right)$
    - $\text{len}_\text{label}$：真实翻译句子单词数；
    - $\text{len}_\text{pred}$：模型生成句子单词数；
    - 作用：防止模型只输出很短、正确率虚高的短句；生成句子越短，惩罚越大，分数越低。
2. 第二部分：n元语法精度累乘
    - $k$：最大匹配n元组，代码默认k=2（1-gram+2-gram）；
    - $p_n$：n元精度，=预测与标签匹配的n元词组数量 ÷ 预测里总n元词组数量；
    - $\frac{1}{2^n}$：权重，n越大指数越小，长词组匹配会大幅拉高分数，鼓励通顺长句。
3. 取值范围：0~1，越接近1翻译质量越好。

## 三、编码器 Seq2SeqEncoder（PyTorch代码逐行详解）
### 完整代码
```python
#@save
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state
```
### 1. __init__ 构造函数参数详解
| 参数 | 含义 |
| ---- | ---- |
| vocab_size | 源语言词表总大小（单词编号最大值+1） |
| embed_size | 词嵌入向量维度，每个单词转embed_size维浮点数向量 |
| num_hiddens | GRU隐藏层神经元数量，隐状态向量维度 |
| num_layers | GRU堆叠层数（多层循环网络，提取深层语义） |
| dropout | 循环层dropout概率，防止过拟合 |

- `self.embedding = nn.Embedding(vocab_size, embed_size)`：词嵌入层，输入单词数字索引，输出稠密语义向量；
- `self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)`：多层GRU，输入维度=词嵌入维度，输出隐藏层维度num_hiddens。

### 2. forward 前向传播：数据变换全过程
输入`X`：批次输入，**形状(batch_size, num_steps)**
- batch_size：一批句子数量；num_steps：统一填充后的句子最大长度。
#### 步骤1：词嵌入
`X = self.embedding(X)`
输出形状：`(batch_size, num_steps, embed_size)`
数字单词索引 → 语义向量。

#### 步骤2：维度交换 permute(1,0,2)
`X = X.permute(1, 0, 2)`
形状变为：`(num_steps, batch_size, embed_size)`
原因：PyTorch GRU默认**第一维是时间步**（time_major），每一步同时处理整批句子同一个位置单词。

#### 步骤3：GRU前向计算
`output, state = self.rnn(X)`
1. `output`：每一个时间步所有层输出，形状`(num_steps, batch_size, num_hiddens)`；
2. `state`：**最后一个时间步多层隐状态**，形状`(num_layers, batch_size, num_hiddens)`，这就是传给解码器的上下文载体。

#### 返回值
`return output, state`
- output：编码器全部时间步隐状态（本章没用，下一章注意力机制才用）；
- state：最终隐状态，用于初始化解码器隐状态。

### 编码器实例演示（入参出参）
```python
# 实例化：词表10个词，嵌入8维，隐藏层16，2层GRU，无dropout
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
# 输入：batch=4个句子，每个句子固定7个词
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
print(output.shape) # torch.Size([7, 4, 16]) (时间步，批次，隐藏维度)
print(state.shape)  # torch.Size([2, 4, 16]) (层数，批次，隐藏维度)
```

## 四、解码器 Seq2SeqDecoder（PyTorch逐行详解）
### 完整代码
```python
#@save
class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state
```
### 1. __init__ 参数
- vocab_size：目标语言词表大小；embed_size：目标词嵌入维度；
- GRU输入维度 = `embed_size + num_hiddens`：每个词向量拼接全局上下文向量；
- `self.dense`：全连接输出层，把GRU隐藏状态映射到目标词表维度，输出每个单词预测概率。

### 2. init_state 初始化方法
```python
def init_state(self, enc_outputs, *args):
    return enc_outputs[1]
```
接收编码器返回的`(output, state)`，直接取出编码器最终隐状态`state`，作为解码器初始隐状态。
要求：编码器、解码器GRU层数、隐藏单元数必须完全一致，否则维度不匹配。

### 3. forward 前向数据流转（核心拼接逻辑）
输入`X`：解码器输入序列（训练时是带`<bos>`的标签序列），形状`(batch_size, num_steps)`
输入`state`：编码器传来的初始隐状态。
#### 步骤1：目标词嵌入
`X = self.embedding(X).permute(1, 0, 2)`
形状：`(num_steps, batch_size, embed_size)`，统一时间步在前。

#### 步骤2：提取全局上下文向量并广播
`context = state[-1].repeat(X.shape[0], 1, 1)`
- `state[-1]`：编码器最后一层隐状态，即全局上下文c，形状`(batch_size, num_hiddens)`；
- repeat复制时间步维度，变成`(num_steps, batch_size, num_hiddens)`；
  作用：每个时间步解码器都能看到整句原文信息。

#### 步骤3：拼接单词向量+上下文
`X_and_context = torch.cat((X, context), 2)`
第三维拼接，单步向量维度变为`embed_size + num_hiddens`，匹配GRU输入维度。

#### 步骤4：GRU计算解码隐状态
`output, state = self.rnn(X_and_context, state)`
output：每个时间步解码隐状态 `(num_steps, batch_size, num_hiddens)`。

#### 步骤5：全连接预测单词，换回batch优先维度
`output = self.dense(output).permute(1, 0, 2)`
最终输出形状：`(batch_size, num_steps, vocab_size)`，每个位置是词表所有单词原始得分（logits），后续softmax得到概率。

### 解码器实例演示
```python
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
print(output.shape) # torch.Size([4, 7, 10]) (批次，时间步，目标词表大小)
print(state.shape)  # torch.Size([2, 4, 16])
```

## 五、掩码函数 sequence_mask + 带遮蔽损失 MaskedSoftmaxCELoss
### 1. sequence_mask 掩码函数（屏蔽填充<pad>）
#### 代码
```python
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
```
#### 参数&逻辑
- X：待掩码张量，形状`(batch, num_steps)`或`(batch, num_steps, C)`；
- valid_len：每个句子**有效真实长度**（不含pad），一维向量`(batch,)`；
- mask生成逻辑：
    1. 生成0~maxlen-1序列；
    2. 对每个句子，时间步 < 有效长度 = True（保留），≥有效长度 = False（掩码置0）；
- 示例：
```python
X = torch.tensor([[1,2,3],[4,5,6]])
sequence_mask(X, torch.tensor([1,2]))
# 输出：[[1,0,0],[4,5,0]]，第二个句子第3位填充被屏蔽
```

### 2. MaskedSoftmaxCELoss 带遮蔽交叉熵损失
#### 代码
```python
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
```
#### 输入形状
- pred：解码器输出 `(batch, num_steps, vocab_size)`；
- label：真实目标句子 `(batch, num_steps)`；
- valid_len：每个句子有效长度 `(batch,)`。
#### 流程拆解
1. 生成权重矩阵weights，有效位置1，填充位置0；
2. `pred.permute(0,2,1)`：CrossEntropyLoss要求预测维度在第二维，转换为`(batch, vocab_size, num_steps)`；
3. 计算每个词元单独损失（reduction=none不自动求和）；
4. 损失 × 掩码权重，填充位损失清零，按句子求均值，返回每个句子损失。

#### 测试示例
```python
loss = MaskedSoftmaxCELoss()
# 3个句子，最长4词，有效长度分别4、2、0
loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0]))
# 输出 tensor([2.3026, 1.1513, 0.0000])
# 第三句有效长度0，全部掩码，损失为0；第二句只有一半有效词，损失减半
```

## 六、训练函数 train_seq2seq（强制教学Teacher Forcing）
### 核心概念：强制教学
训练阶段，**不用模型上一步预测的词输入解码器**，而是直接拿真实标签`<bos>+标签前n-1个词`作为输入。
优点：训练收敛更快、梯度稳定；缺点：训练/预测输入分布不一致（暴露偏差）。

### 关键代码片段讲解
```python
bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
dec_input = torch.cat([bos, Y[:, :-1]], 1)
```
- Y：目标标签句子（包含`<eos>`）；
- Y[:, :-1]：去掉最后一列`<eos>`；
- 拼接`<bos>`在开头，得到解码器输入：`<bos> y1 y2 ... yT-1`。

完整训练流程：
1. 遍历数据集，取出源句X、源有效长度X_valid_len、目标Y、目标有效长度Y_valid_len；
2. 构造解码器输入dec_input（强制教学）；
3. 编码器编码X，解码器解码得到预测Y_hat；
4. 计算掩码损失，反向传播；
5. 梯度裁剪d2l.grad_clipping(net, 1)，防止RNN梯度爆炸；
6. 累加总损失与总词元数，每10轮打印损失曲线。

## 七、预测函数 predict_seq2seq（推理阶段）
训练和预测最大区别：**预测时解码器输入是上一步自己预测的词，不是真实标签**。
### 推理步骤
1. 输入源句子分词，末尾加`<eos>`，截断/填充到num_steps；
2. 编码器一次性编码整句，拿到隐状态初始化dec_state；
3. 解码器初始输入`<bos>`；
4. 循环逐词生成：
    - 解码器输入当前词，输出词表概率；
    - argmax取概率最大词作为下一轮输入；
    - 若预测出`<eos>`，立刻停止生成；
5. 把词索引转回单词，返回译文。

## 八、BLEU评估函数讲解
```python
def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```
1. 短句惩罚：先计算BP因子；
2. 遍历1~k元词组：
    - 统计标签内所有n元词组出现次数；
    - 遍历预测句n元词组，匹配则计数+1，标签字典次数-1（避免重复匹配）；
    - 计算pn，加权乘入总分；
3. 返回0~1之间分数，越高翻译越好。

### 实例
标签：A B C D E F；预测：A B B C D
- p1=4/5，p2=3/4，p3=1/3，p4=0；k=2时只算1、2元组。

## 九、整体流程总结
1. **数据预处理**：句子分词、数字编码、添加`<bos>/<eos>/<pad>`、统一长度；
2. **模型搭建**：Encoder(Embedding+多层GRU) + Decoder(Embedding+GRU+全连接)；
3. **训练**：强制教学输入解码器，掩码交叉熵忽略填充损失，梯度裁剪；
4. **推理**：自回归逐词生成，遇到`<eos>`终止；
5. **评估**：BLEU分数衡量译文与标准译文重合度。

## 十、原始Seq2Seq缺陷（本章遗留问题，下一章注意力解决）
1. **信息瓶颈**：所有原文信息压缩进唯一固定长度c，长句子信息丢失严重；
2. **无对齐能力**：解码器每一步只能看全局向量，无法重点关注原文对应单词；
3. **训练推理不一致**：强制教学导致暴露偏差，长文本生成质量下滑。
   后续9.8注意力机制会解决该瓶颈。