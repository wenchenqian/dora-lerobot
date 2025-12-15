LoRA 训练目前是只能支持每一层 Attention 层和 MoE  共享专家的 lora 权重是吧，目前还没有MoE 层路由专家的 lora 权重

支持了routed_experts的lora
因为我觉得routed_experts的参数量占绝大部分，微调这个更合理些

![img_14.png](img_14.png)
![img_15.png](img_15.png)
![img_16.png](img_16.png)

这份配置文件描述的是一个 **基于 DeepSeek 架构的超大规模 MoE（Mixture-of-Experts）语言模型**，用于 **指令微调（SFT）或继续预训练**，并启用了 **LoRA（Low-Rank Adaptation）** 参数高效微调技术。下面我将从 **模型整体结构、Tokenizer、Embedding、位置编码、Transformer Block（含 MLA + MoE）、LoRA 配置** 等方面逐层深度解析。

---

## 一、整体概览

- **模型类型**：DeepSeek 风格的 **61 层 MoE 模型**
- **隐藏层维度**：`7168`（注意：不是 7680，区别于前一份配置）
- **注意力机制**：**MLA（Multi-head Latent Attention）** —— DeepSeek V3 的核心创新
- **MoE 配置**：256 专家、Top-8、每层 MoE（前 3 层 dense）
- **上下文长度**：通过 YaRN 扩展至 **163,840 tokens**
- **LoRA 微调**：全组件注入（Q/KV/Proj/FFN/Experts/Output）

### 为什么需要 YaRN？—— RoPE 的外推问题

**RoPE 的核心思想是：****通过旋转矩阵将位置信息注入 attention 的 Q/K 向量中****，其频率由 **`θ = base^(2d/D)` 决定（`base=10000` 是默认值）。

* **训练时****：模型见过的位置 **`m ∈ [0, L_train)`（如 L\_train = 4096）
* **推理时****：若 **`m ≥ L_train`（如 m=50000），RoPE 的旋转角度会变得**极其高频、振荡剧烈****，导致：**
  * **attention 分布混乱**
  * **模型“看不懂”长文本**
  * **性能急剧下降（称为 ****外推失效，extrapolation failure****）**

    rope_scaling_type: yarn
    rope_scaling_factor: 40
    rope_scaling_original_max_position_embeddings: 4096
    max_size: 163840

---

## 二、Tokenizer 配置

```yaml
tokenizer:
  type: 'PretrainedFromHF'
  path: /data/.../deepseek_R1/
  vocab_size: 129280
  padded_size: 129280
  divided_by: 1
  use_fast: True
```

- 使用 Hugging Face 格式的 **DeepSeek R1 tokenizer**
- 词表大小 **129,280**（比前一份配置的 153k 小，可能是早期版本）
- `use_fast: True`：启用 Rust 加速的 tokenizer（如 `tokenizers` 库）
- `divided_by: 1`：不对 vocab size 做 padding 对齐（如对齐到 128/256 等）

---

## 三、Embedding & Position Encoding

### 1. Embedding

- `hidden_size: 7168`：词向量维度
- `init_std: 0.5`：较大的初始化标准差（通常为 0.02，此处可能针对特殊初始化策略）

### 2. Position Encoding（RoPE + YaRN）

```yaml
position:
  type: 'rope'
  max_size: 163840
  rope_base: 10000.0
  rope_scaling_type: yarn
  rope_scaling_factor: 40
  rope_scaling_original_max_position_embeddings: 4096
  rope_scaling_mscale_all_dim: 1.0
```

- 原始训练上下文：4096
- 通过 **YaRN（Yet another RoPE extensioN）** 扩展到 **163,840**（40×）
- `rope_base=10000`：标准 RoPE 基数
- `mscale=1.0`：无额外幅度缩放（通常设为 >1 以补偿外推损失，此处保守）

---

## 四、Transformer Blocks（核心创新：MLA + MoE）

### 1. 基础结构

- `num_layers: 61`：超深网络
- `layernorm: RMSNorm (ε=1e-6)`：标准配置

  **RMSNorm = 去掉均值中心化的 LayerNorm，只做幅度归一化**
- `sandwich_norm: False`：不使用 sandwich LN（即 LN 不夹在 attention/FFN 之间）

**SwiGLU**(**x**)**=**Swish**(**x**W**1****)**⊙**(**x**W**3****)**⋅**W**2****

**Swish**(**z**)**=**z**⋅**σ**(**z**)**，其中 σ**σ** 是 sigmoid 函数

**无 bias + SwiGLU + RMSNorm**

---

### 2. Attention：MLA（Multi-head Latent Attention）

这是 **DeepSeek V3 的标志性技术**，不同于 GQA/MQA。

```yaml
attention:
  attention_type: "MLA"
  q_lora_rank: 1536
  kv_lora_rank: 512
  qk_nope_head_dim: 128   # Non-Positional part
  qk_rope_head_dim: 64    # RoPE part
  v_head_dim: 128
  head_num: 128
  qk_layernorm: True
```

#### MLA 核心思想：

将 Q/K 分解为两个子空间：

- **NOPE（Non-Positional）**：`128-dim`，捕获语义信息
- **ROPE**：`64-dim`，仅用于位置编码

并通过 **低秩投影** 压缩 Q/K 矩阵：

- Q 投影：原始 `7168 → 1536（LoRA） → 128+64=192`
- KV 投影：`7168 → 512（LoRA） → 128+64`

> 💡 这种设计 **大幅降低 QKV 计算量和显存**，同时保持长上下文性能。

#### 其他细节：

- `qk_layernorm: True`：在 Q/K 上加 LN，提升稳定性（DeepSeek 特有）
- `kv_group_size: 1`：即 **MQA（Multi-Head Attention with shared KV）**，128 个 Q 头共享 1 组 KV
- `qkv_bias: False`：无偏置项（节省参数）
- `padded_base_length: 128`：可能用于 kernel 优化对齐

---

### 3. FFN & MoE（Mixture of Experts）

```yaml
ffn:
  hidden_size: 18432  # ~2.57× hidden_size
  activation: 'swiglu'

moe:
  layers_replaced_as_dense: [0, 1, 2]  # 前3层不用MoE
  num_experts: 256
  topk: 8
  routed_expert_hidden_size: 2048
  shared_expert_hidden_size: 2048
  n_shared_experts: 1
  aux_loss_coeff: 0.0001
  load_balancing_type: noaux_tc
  sigmoid_gating: True
  router_type: 'learnable'
  routed_scaling_factor: 2.5
  norm_topk_prob: True
  topk_group: 4
  router_enable_expert_bias: True
```

#### MoE 架构解析：

- **总专家数**：256
- **激活数**：Top-8（每个 token 选 8 个专家）
- **专家结构**：
  - **Routed Experts**：256 个，每专家 FFN 中间维度 = 2048
  - **Shared Expert**：1 个，所有 token 都经过（类似 GLaM）
- **前 3 层为 Dense**：避免浅层过拟合或路由不稳定

#### 路由器（Router）高级特性：

- `sigmoid_gating: True`：使用 sigmoid 而非 softmax 打分（更稀疏）
- `router_enable_expert_bias: True`：每个专家有可学习偏置，提升负载均衡
- `topk_group: 4`：将 256 专家分 4 组，每组选 Top-2（总 Top-8），减少通信
- `routed_scaling_factor: 2.5`：对 routed experts 输出放大 2.5 倍（补偿 shared expert 影响）
- `load_balancing_type: noaux_tc`：使用无辅助损失的负载均衡（可能基于 HCCL/NCCL 优化）
- `aux_loss_coeff: 0.0001`：极小的辅助损失权重（几乎不用）


**不再使用 Softmax，而是对每个专家独立应用 Sigmoid****：**

gi=σ(si)=11+exp⁡(−si)**g**i****=**σ**(**s**i****)**=**1**+**exp**(**−**s**i****)**1**

---

## 五、Output Head

```yaml
head:
  shared_embedding: False
```

- **不共享 embedding 与 output weight**（即 `lm_head` 独立训练）
- 在 SFT 或 MoE 模型中常见，因 embedding 可能无需更新

---

## 六、LoRA 微调配置（全量适配）

```yaml
lora:
  lora_r: 16
  lora_alpha: 32
  lora_target_modules: 
    - linear_qb 
    - linear_kvb 
    - linear_proj 
    - experts 
    - mlp.linear_fc1 
    - mlp.linear_fc2 
    - output_layer
  lora_register_forward_hook: word_embeddings input_layernorm
```

### 1. LoRA 参数

- `lora_r=16`：秩
- `lora_alpha=32` → 缩放因子 = 32/16 = **2**（标准设置）

### 2. 注入模块（非常全面！）


| 模块                 | 说明                                      |
| -------------------- | ----------------------------------------- |
| `linear_qb`          | MLA 中的 Q 投影（含 LoRA 分解）           |
| `linear_kvb`         | MLA 中的 KV 投影                          |
| `linear_proj`        | Attention 输出投影（O 矩阵）              |
| `experts`            | **所有 MoE 专家**（包括 routed + shared） |
| `mlp.linear_fc1/fc2` | Dense 层中的 FFN（前 3 层）               |
| `output_layer`       | 语言模型头（lm_head）                     |

3. 特殊 Hook

- `lora_register_forward_hook: word_embeddings input_layernorm`表示对 **word embeddings 和 input layernorm** 也注册了前向钩子，可能用于：
  - 冻结 embedding 但记录梯度
  - 添加适配层（如 prefix-tuning）


### `LoraParallelLinearMoE`：适配张量并行线性层


**Megatron 的线性层有两种并行模式：**

* **`RowParallelLinear`****：输出维度切分 → LoRA 的 ****A 矩阵行切分****，****B 矩阵完整****；**
* **`ColumnParallelLinear`****：输入维度切分 → LoRA 的 ****A 矩阵完整****，****B 矩阵列切分****。**

所有非 MoE 的 TP 线性层（如 attention QKV、MLP 中间层等）

### `LoraGroupGemmExperts`：适配 MoE 专家层


* **双投影 LoRA****：**
  * `lora_A / lora_B`：用于 **up projection****（weight1）；**
  * `lora_A2 / lora_B2`：用于 **down projection****（weight2）；**
* **分组 GEMM 优化****：**
  * **使用 **`npu_gmm`（华为 NPU 高效分组矩阵乘）；
  * **通过 **`group_list` 指定每个专家的 token 数量。



#### 1. **替换 LoRA 并行层实现**

* **关键！** 将 PEFT 原生的 `LoraParallelLinear` 替换为你自定义的 `LoraParallelLinearMoE`；
* **确保 LoRA 与 Megatron 的 **`Row/ColumnParallelLinear` 兼容



* **在模型构建完成后，广播 LoRA 参数到 TP/PP 组****；**
* **确保所有 rank 的 LoRA 初始权重一致（避免训练发散）。**


构建原始 Megatron 模型
↓
[ C: 注入 LoRA 配置 + 替换 tp_layer.LoraParallelLinear ]
↓
[ C: 调用 get_peft_model(model, lora_config) ]
↓
[ B: PEFT 遍历模型 → 对每个 target_module 调用 create_new_module ]
↓
[ B: create_new_module 识别层类型（ColumnParallelLinear / GroupGemmExperts）]
↓
┌───────────────────────┐
│ 根据类型选择 LoRA 类型 │
└───────────────────────┘
↓
若是 Row/ColumnParallelLinear → 返回 LoraParallelLinearMoE（A）
若是 GroupGemmExperts         → 返回 LoraGroupGemmExperts（A）
↓
[ B: 调用 replace_module → 将原层替换为 A 中的 LoRA 层 ]
↓
[ C: 设置参数属性、注册 hook、扩展 wrapper 白名单 ]
↓
[ C: broadcast_params() → 同步 LoRA 初始权重 ]
↓
[ 训练开始：forward 调用 A 的 forward 方法 ]
↓
[ 保存 checkpoint 时：C 的 unwrap_model_wrapper 穿透 PeftModel 获取 base model ]

---

## 七、与前一份配置的对比


| 项目             | 本配置                       | 前一份配置        |
| ---------------- | ---------------------------- | ----------------- |
| `hidden_size`    | 7168                         | 7680              |
| `vocab_size`     | 129280                       | 153376            |
| `attention_type` | MLA                          | MLA（相同）       |
| `MoE`            | 256E8T                       | 256E8T（相同）    |
| `LoRA targets`   | **+ experts + output_layer** | 无 experts/output |
| **训练阶段**     | 可能是**继续预训练 or SFT**  | 明确为 SFT        |

> 本配置的 LoRA 更激进 —— **连 MoE 专家和 lm_head 都微调**，适合需要强任务适配的场景（如金融、机器人指令）。

---

## 八、潜在问题与建议

1. **LoRA 覆盖 experts**：

   - MoE 专家参数量巨大（256 × 2048 × 7168 ≈ 37B），即使 LoRA（r=16）也会新增 **~120M 参数**
   - 需确保显存足够，或考虑 **仅微调 shared expert**
2. **`init_std=0.5` for embedding**：

   - 过大可能导致训练初期不稳定，建议监控 loss 是否爆炸
3. **`mscale=1.0`**：

   - YaRN 推荐 `mscale > 1`（如 1.2~1.5）以补偿外推损失，可尝试调高
4. **NPU 兼容性**：

   - MLA + MoE + LoRA 的组合对 kernel 要求高，需确认 NPU 是否支持自定义算子

---
