# 从零实现RNN 逐行超细拆解（PyTorch，分模块、按执行顺序讲）
先明确整体任务：拿《时间机器》英文文本，**字符级循环神经网络**，输入一串字符，预测下一个字符，实现文本续写。
所有代码拆成 6 大阶段：数据加载 → one-hot编码 → RNN权重/隐状态初始化 → RNN前向计算 → 预测函数 → 训练循环（梯度裁剪+参数更新）。

## 前置导入
```python
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```
- `torch`：张量、运算、GPU
- `nn`：损失函数、网络基类
- `F.one_hot`：独热编码
- `d2l`：封装好的文本数据集、绘图、累加器、计时器等工具

---

# 第一块：加载时序数据（最前面执行）
```python
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```
### 参数含义
1. `batch_size=32`：一次并行处理32条独立文本片段
2. `num_steps=35`：每条片段固定长度35个字符（时间步T=35）
3. `train_iter`：迭代器，每次取出一组 `(X,Y)`
    - X：输入序列，shape `(32, 35)` 32行batch，每行35个字符索引
    - Y：标签序列，是X整体向右偏移一位（输入t时刻字符，预测t+1字符）
4. `vocab`：词表对象
    - `vocab.token_to_idx['a']`：字符转数字索引
    - `vocab.idx_to_token[5]`：数字转回字符
    - `len(vocab)`：全文出现过多少种不同字符（本数据集28）

### 两种迭代器区别（重点）
```python
# 随机采样版：每次随机截取文本片段，片段之间无上下文
train_random_iter, vocab_random_iter = d2l.load_data_time_machine(
    batch_size, num_steps, use_random_iter=True)
```
1. 默认顺序分区：文本从头到尾连续切，第1个batch第0条结尾 = 第2个batch第0条开头，隐状态可以跨batch传递
2. 随机采样：每次随机抽一段，batch之间完全无关，每轮都要重置隐状态

---

# 第二块：One-hot 编码维度拆解（最难理解的维度转换）
```python
X = torch.arange(10).reshape((2, 5))  # 模拟小batch：batch=2，时间步=5
F.one_hot(X.T, 28).shape
```
### 分步拆解
1. `X = (batch, num_steps) = (2,5)`
```
X = [[0,1,2,3,4],
     [5,6,7,8,9]]
```
2. `X.T` 转置，交换两个维度 → `(num_steps, batch) = (5,2)`
```
X.T = [[0,5],
       [1,6],
       [2,7],
       [3,8],
       [4,9]]
```
3. `F.one_hot(张量, num_classes=28)`
   每个数字变成长度28的0/1向量，数字几，第几位就是1
   输出shape：`(num_steps, batch, vocab_size) = (5,2,28)`

### 为什么必须转置？
RNN逻辑是**按时间一步步循环**：
第一层维度是时间步，循环时可以直接 `for 每个时间步输入 in inputs`，逐个计算每一刻的隐状态H。
如果不转置，第一层是batch，循环逻辑会非常别扭。

### 举例单时间步输入
取 t=0 时刻：`inputs[0]` shape `(2,28)`，对应batch内两条样本的字符向量，正好可以做矩阵乘法。

---

# 第三块：定义RNN参数函数 get_params
```python
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 1. 隐层权重
    W_xh = normal((num_inputs, num_hiddens))  # 输入→隐藏
    W_hh = normal((num_hiddens, num_hiddens))# 上一时刻隐藏→当前隐藏
    b_h = torch.zeros(num_hiddens, device=device) # 隐藏偏置

    # 2. 输出层权重
    W_hq = normal((num_hiddens, num_outputs)) # 隐藏→输出预测字符
    b_q = torch.zeros(num_outputs, device=device) # 输出偏置

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True) # 所有参数开启梯度，可训练
    return params
```
## 逐段解释
1. `num_inputs = vocab_size`：输入是one-hot向量，长度等于字符种类
2. `num_outputs = vocab_size`：输出预测下一个字符，也是28分类
3. `normal()` 辅助函数：生成小随机数初始化权重（不能全0，否则无法区分特征）
4. 5组权重含义（RNN核心公式配套）
   $$H_t = tanh(X_t W_{xh} + H_{t-1} W_{hh} + b_h)$$
   $$Y_t = H_t W_{hq} + b_q$$
- $W_{xh}$：当前输入向量映射到隐藏维度
- $W_{hh}$：上一步记忆H映射到当前记忆
- $b_h$：隐藏层偏置
- $W_{hq}$：隐藏记忆映射到字符分类得分
- $b_q$：输出层偏置

5. `requires_grad_(True)`：告诉torch反向传播时要计算这个张量梯度，用于更新权重

---

# 第四块：初始化隐状态 init_rnn_state
```python
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```
## 讲解
1. 训练刚开始，没有任何历史字符，所以初始记忆H全部置0
2. H形状：`(batch_size, num_hiddens)`
   每个batch里每条样本独立拥有一组隐藏记忆
3. 为什么包进元组 `(H,)`？
   为统一接口：后面LSTM有两个状态（细胞态+隐状态），也要返回元组，这里RNN只有一个状态，也保持元组格式，代码复用。

---

# 第五块：单轮RNN前向计算 rnn() 核心循环
```python
def rnn(inputs, state, params):
    # inputs: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state # 取出隐状态
    outputs = [] # 保存每个时间步预测得分

    # 逐时间步循环
    for X in inputs:
        # 计算当前隐状态
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 由隐状态预测字符
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    # 把所有时间步拼接成一个大矩阵
    return torch.cat(outputs, dim=0), (H,)
```
## 逐行拆解循环内部
1. `for X in inputs`：遍历每一个时间步 t
    - X 单步输入 shape `(batch, vocab_size)`
2. `torch.mm(X, W_xh)`：当前输入映射隐藏
3. `torch.mm(H, W_hh)`：上一步记忆映射隐藏
4. 两项相加 + 偏置，过tanh激活，限制值域 [-1,1]，缓解梯度爆炸
5. 用新H算出当前时刻字符得分Y，存入列表
6. 循环结束后 `torch.cat(outputs, dim=0)`
   outputs列表每个元素是`(batch, vocab_size)`，拼接后：
   `(num_steps * batch, vocab_size)`
   方便后面和展平的标签一次性算交叉熵损失

返回两个东西：
1. 全部时刻预测得分
2. 最后一步隐状态（保留上下文，给下一个batch用）

---

# 第六块：封装模型类 RNNModelScratch
把编码、初始化状态、前向逻辑打包成类，方便调用
```python
class RNNModelScratch: 
    """从零实现RNN模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        # 保存超参、权重生成函数、初始化状态函数、前向计算函数
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # X原始输入 (batch, num_steps)
        # 1.转置 + one-hot 编码
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # 2.调用前面写好的rnn前向函数
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        # 对外接口，生成初始0隐状态
        return self.init_state(batch_size, self.num_hiddens, device)
```
## 关键：`__call__` 魔法函数
实例化net后，可以直接 `net(X, state)` 调用，等价调用这里的逻辑：
1. 自动完成转置+one-hot编码，不用外部手动处理维度
2. 调用我们手写的rnn循环做前向传播

## 测试维度代码逐行看懂
```python
num_hiddens = 512 # 隐藏层神经元个数
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
X = torch.arange(10).reshape((2, 5)) # batch=2, T=5
state = net.begin_state(X.shape[0], d2l.try_gpu()) # 初始化H
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```
执行流程：
1. 创建网络，随机初始化5组权重
2. 构造测试输入X
3. `begin_state` 创建全0隐状态 (2,512)
4. `net(X,state)` 进入`__call__`：转置+one-hot → 调用rnn循环
5. 返回所有时刻输出Y、最后时刻隐状态new_state

输出结果含义：
- Y.shape = (10,28)：2batch ×5时间步=10条预测，28个字符分类
- new_state[0].shape=(2,512)：最后一步记忆，每条样本512维隐藏向量

---

# 第七块：文本预测函数 predict_ch8（续写文字）
```python
def predict_ch8(prefix, num_preds, net, vocab, device):
    """prefix前缀，往后生成num_preds个字符"""
    state = net.begin_state(batch_size=1, device=device) # 单条样本生成
    outputs = [vocab[prefix[0]]] # 先把第一个字符转索引存起来
    # 生成输入：取上一步最后一个预测字符，包装成 (1,1) 输入格式
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    
    # 阶段1：预热前缀（只更新记忆，不生成新字符）
    for y in prefix[1:]:
        _, state = net(get_input(), state) # 前向传播更新H，丢弃输出
        outputs.append(vocab[y]) # 把原有前缀字符存入结果
    
    # 阶段2：循环生成新字符
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # argmax取得分最高字符索引
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    
    # 索引列表转回字符串
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```
## 两段核心逻辑：预热 + 生成
### 1）预热阶段（必须有）
比如前缀是 `time traveller`，先把这串字符全部喂进网络：
网络逐字符更新隐状态H，让H记住这段上下文，**不能直接跳过前缀生成**，否则没有上下文记忆，生成乱码。
只更新state，输出不使用，只是缓存输入字符。

### 2）生成阶段
1. 用上一步预测的字符作为下一轮输入
2. net前向得到所有字符得分
3. `argmax(dim=1)` 贪心选概率最大字符
4. 存入outputs，循环直到生成指定数量字符

### 为什么`batch_size=1`？
生成文本时只有一条前缀，不需要并行batch。

---

# 第八块：梯度裁剪 grad_clipping（解决梯度爆炸）
```python
def grad_clipping(net, theta):
    """裁剪梯度上限theta"""
    # 区分两种模型：手写RNN类 / torch官方nn.Module
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 计算所有参数梯度的全局L2范数
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    # 如果范数超过阈值，等比例缩小所有梯度
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```
## 通俗原理
RNN长序列反向传播时，矩阵反复相乘，梯度会指数爆炸，参数跳来跳去不收敛。
1. 把所有权重梯度拼成一个大向量，算长度norm
2. 如果长度超过θ（一般设1），全部梯度乘以缩放系数，保证梯度总长度≤θ
3. **只缩小大小，不改变梯度方向**，不破坏优化趋势

## 关键限制
只能解决梯度爆炸；不能解决梯度消失（长距离依赖记不住）。

---

# 第九块：单轮训练函数 train_epoch_ch8（一个epoch完整流程）
```python
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 累加器：[总损失和, 总字符数量]
    for X, Y in train_iter:
        # ========== 1. 判断是否重置隐状态 ==========
        if state is None or use_random_iter:
            # 第一轮 / 随机采样：每次batch重新初始化0状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 顺序分区：复用前一个batch的state，但截断梯度
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        
        # ========== 2. 处理标签维度 ==========
        y = Y.T.reshape(-1) # Y转置展平，和模型输出shape对齐
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state) # 前向传播得到预测得分
        
        # ========== 3. 计算损失 ==========
        l = loss(y_hat, y.long()).mean()
        
        # ========== 4. 反向传播 + 梯度裁剪 + 参数更新 ==========
        if isinstance(updater, torch.optim.Optimizer):
            # torch官方优化器分支
            updater.zero_grad() # 清空上一轮梯度
            l.backward() # 反向传播计算梯度
            grad_clipping(net, 1) # 限制梯度爆炸
            updater.step() # 更新权重
        else:
            # d2l手写SGD分支
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        
        # 累加总损失、总字符数
        metric.add(l * y.numel(), y.numel())
    # 困惑度 exp(平均损失) + 每秒处理字符数
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```
## 逐段重点拆解
### 1. state.detach_() 截断梯度（顺序分区特有）
顺序采样时，下一个batch承接上一个batch的隐状态。
如果不detach，反向传播会一直追溯到几千步之前，计算图无限变长、显存爆炸。
`detach_()`：切断state梯度流向之前的计算图，只在当前batch内反向传播。

### 2. 标签维度对齐
模型输出 `y_hat = (T*B, V)`
原始标签Y `(B,T)` → Y.T转置`(T,B)` → reshape(-1)展平`(T*B,)`，刚好匹配输入交叉熵格式。

### 3. 训练三步固定流程
zero_grad() → loss.backward() → grad_clipping → step()

### 4. 困惑度 Perplexity
平均loss是交叉熵，exp后就是困惑度：
- PPL=1：模型100%猜对所有字符
- PPL越大，文本越不通顺

---

# 第十块：总训练入口 train_ch8
循环多个epoch，每隔10轮打印生成文本、绘图
```python
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 选择优化器
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # 手写SGD，传学习率
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    # 封装生成函数，固定生成50个字符
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    
    # 循环训练轮次
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        # 每10轮打印生成样例
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    # 训练结束输出最终指标
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

---

# 最后启动训练代码执行顺序
```python
num_epochs, lr = 500, 1
net = RNNModelScratch(len(vocab), 512, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```
执行完整链路：
1. 创建网络，初始化全部权重
2. 循环500个epoch
3. 每个epoch遍历全部文本batch：前向→loss→反向→裁剪梯度→更新参数
4. 每10轮用当前网络续写文本，打印观察效果
5. 训练完成输出最终困惑度，测试两段前缀生成文本

## 顺序分区 vs 随机采样训练区别总结
1. **顺序分区（默认）**
    - batch之间上下文连续，隐状态传递，学习长依赖更好
    - 必须detach截断梯度防止计算图无限延伸
    - 困惑度更低，生成文字连贯
2. **随机采样**
    - 每batch重置state，无跨batch记忆
    - 梯度计算简单，显存占用稳定
    - 文本碎片化，连贯性差，PPL更高