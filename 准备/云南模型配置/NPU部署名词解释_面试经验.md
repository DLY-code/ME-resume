# NPU部署相关名词解释 & 面试经验整理

> 来源：项目部署调试过程中的总结，涉及 Orange Pi AI Pro + 昇腾NPU 的 ReID 模型部署

---

## 对话一：ONNX Runtime、ACL、dtype、shape 名词解释

### ONNX Runtime

**是什么：** 微软开发的模型推理框架。你训练好的模型（比如用PyTorch训练的ReID模型）可以导出成 `.onnx` 格式，然后用 ONNX Runtime 在各种硬件上运行。

**为什么要它：** PyTorch 模型（`.pth`）依赖 PyTorch 才能跑，部署时太重。ONNX 是一种"通用格式"，就像把文档保存成 PDF，不管对方用什么系统都能打开。

**项目里的作用：** `reid_inference.py` 用 ONNX Runtime 在 CPU/GPU 上跑 `.onnx` 模型，这是在开发服务器上验证用的。

---

### ACL（Ascend Computing Language）

**是什么：** 华为昇腾 NPU 的推理接口，全称 Ascend Computing Language。就像 NVIDIA GPU 用 CUDA，华为 NPU 用 ACL。

**为什么要它：** Orange Pi AI Pro 上有昇腾 310B1 NPU，想利用 NPU 加速就必须用 ACL 接口。`.om` 模型格式就是专门给 ACL 用的。

**项目里的作用：** `ais_bench` 是封装了 ACL 的 Python 工具包，`InferSession` 就是 ACL 推理会话。从 `.onnx` 用 `atc` 工具转成 `.om`，再用 ACL 跑。

**关键坑（亲身踩过的）：** ONNX Runtime 和 ACL 的输出格式不一样，原版代码没处理这个差异，导致准确度差。

---

### dtype（数据类型）

**是什么：** Data Type 的缩写，即数字的存储格式。常见的：

| dtype | 全称 | 精度 | 内存占用 |
|-------|------|------|---------|
| `float32` | 32位浮点 | 高 | 4字节/个数 |
| `float16` | 16位浮点 | 低一半 | 2字节/个数 |
| `int8` | 8位整数 | 更低 | 1字节/个数 |

**为什么重要：** 模型特征向量里的每个数字都有 dtype。用 float16 计算余弦相似度，数值精度损失，导致相似度结果不准。

**项目里的坑：** ACL 推理后输出可能是 float16（NPU 为了加速会降精度），但计算相似度需要 float32。修复版加了：
```python
if output.dtype != np.float32:
    output = output.astype(np.float32)
```

---

### shape（张量形状）

**是什么：** 多维数组的维度描述。比如一张 256×256 的 RGB 图像，shape 是 `(256, 256, 3)`，表示高×宽×通道数。

**常见形状含义：**

| shape | 含义 |
|-------|------|
| `(256, 256, 3)` | 单张图像，H×W×C |
| `(1, 3, 256, 256)` | batch=1 的图像，B×C×H×W（PyTorch格式） |
| `(128,)` | 128维特征向量 |
| `(1, 128)` | batch=1 的特征向量 |
| `(1, 128, 1, 1)` | 多余维度，OM模型有时会输出这种 |

**项目里的坑：** ONNX Runtime 输出形状是干净的 `(1, 128)`，取 `[0]` 得到 `(128,)` 的特征向量，正常。ACL 输出可能是 `(1, 128, 1, 1)`，直接 `[0][0]` 取到的是 `(1, 1)` 而不是 `(128,)`（第一个`[0]`取batch第0个→`(128,1,1)`，第二个`[0]`取第0个→`(1,1)`），计算就出错了。修复版用 `.squeeze()` 移除多余的大小为1的维度，统一变成 `(128,)`。

---

### 面试话术（参考）

> 在 Orange Pi AI Pro 上部署 ReID 模型时，发现用 NPU 推理的匹配准确度明显低于用 ONNX Runtime 的结果。排查后发现两个问题：一是昇腾 ACL 推理输出的 dtype 是 float16，而计算余弦相似度需要 float32，精度损失导致特征向量不准；二是 OM 模型输出的 tensor shape 有多余维度，没有正确展平就直接使用。修复方法是在 ACL 输出后增加 dtype 转换和 squeeze 操作，对齐两个推理后端的输出格式。

---

## 对话二：CANN 是什么

### CANN 不只是推理框架

**CANN**（Compute Architecture for Neural Networks）是华为昇腾的**整套AI计算软件栈**，包含多个组件：

```
CANN 软件栈
├── ATC          # 模型转换工具（onnx → om）
├── ACL          # 编程接口（代码用来调用NPU）
├── ais_bench    # 推理测试工具（InferSession 就来自这里）
├── MindSpore    # 华为自己的训练框架（类似 PyTorch）
└── 驱动 / 运行时  # 底层硬件驱动
```

---

### 类比 NVIDIA 理解

| 华为 | NVIDIA | 作用 |
|------|--------|------|
| CANN | CUDA 生态 | 整套AI计算平台 |
| ACL | CUDA Runtime | 编程接口，调用硬件 |
| ATC | TensorRT | 模型优化/转换工具 |
| `.om` 模型 | `.engine` / TensorRT模型 | 部署用的优化模型格式 |
| MindSpore | PyTorch | 训练框架 |

---

### 项目中的调用链

```
你的代码
  └── ais_bench.InferSession   ← Python封装
        └── ACL               ← 编程接口
              └── CANN运行时   ← 底层
                    └── 昇腾310B1 NPU  ← 硬件
```

---

### 面试话术（参考）

> 项目部署在 Orange Pi AI Pro 上，使用华为 CANN 软件栈。模型转换用 CANN 中的 ATC 工具将 ONNX 转成 OM 格式，推理时通过 ACL 接口调用 NPU 执行。

---

*记录时间：2026-04-11*
