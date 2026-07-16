# Python与PyTorch基础

## 一、Python基本数据类型

Python变量不需要提前声明类型，常见数据类型包括：

| 类型 | 作用 | 示例 |
|---|---|---|
| `int` | 表示整数 | `count = 10` |
| `float` | 表示浮点数 | `rate = 0.01` |
| `str` | 表示文本 | `name = "model"` |
| `bool` | 表示真假状态 | `is_train = True` |
| `None` | 表示暂时没有值 | `result = None` |

可以使用`type()`查看变量类型：

```python
value = 10
print(type(value))
```

## 二、Python常用容器

### 1. 列表

列表用于保存一组有顺序、可以修改的数据。

```python
numbers = [1, 2, 3]
numbers.append(4)
```

### 2. 元组

元组与列表类似，但创建后不能修改，适合表示固定结构。

```python
shape = (16, 2)
```

### 3. 集合

集合中的元素不重复，适合去重和成员判断。

```python
items = {1, 2, 2, 3}
```

### 4. 字典

字典使用“键—值”形式保存数据。

```python
config = {
    "learning_rate": 0.01,
    "batch_size": 32
}
```

### 5. 容器对比

| 类型 | 是否有序 | 是否可修改 | 是否允许重复 | 访问方式 |
|---|---:|---:|---:|---|
| `list` | 是 | 是 | 是 | 索引 |
| `tuple` | 是 | 否 | 是 | 索引 |
| `set` | 不强调顺序 | 是 | 否 | 成员判断 |
| `dict` | 是 | 是 | 键不能重复 | 键 |

## 三、条件判断与循环

### 1. 条件判断

```python
if score >= 60:
    result = "通过"
else:
    result = "未通过"
```

### 2. `for`循环

```python
for item in items:
    print(item)
```

### 3. `enumerate`

`enumerate`可以同时获得元素的位置和内容。

```python
for index, item in enumerate(items):
    print(index, item)
```

## 四、函数

函数用于封装可以重复使用的逻辑。

```python
def calculate_sum(a, b):
    result = a + b
    return result
```

阅读函数时应重点关注：

1. 输入参数是什么
2. 函数内部进行了什么处理
3. 返回值是什么
4. 函数在哪里被调用

## 五、类与对象

类用于组织属性和方法，对象是类的具体实例。

```python
class ModelConfig:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def show(self):
        print(self.learning_rate)
```

其中：

- `class`用于定义类
- `__init__`用于初始化对象
- `self`表示当前对象
- 属性用于保存数据
- 方法用于定义对象的行为

## 六、Tensor基本概念

Tensor是PyTorch中保存数据、模型参数和计算结果的基本对象，可以理解为支持批量计算和自动求导的多维数组。

| Tensor形式 | 维度数 | Shape示例 |
|---|---:|---|
| 标量 | 0 | `()` |
| 向量 | 1 | `(D,)` |
| 矩阵 | 2 | `(B, D)` |
| 三维Tensor | 3 | `(B, L, D)` |

常用字母含义：

- `B`：Batch Size
- `L`：序列长度
- `D`：特征维度

## 七、Tensor的重要属性

```python
import torch

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

print(x.shape)
print(x.ndim)
print(x.dtype)
print(x.device)
```

| 属性 | 含义 |
|---|---|
| `shape` | 各个维度的大小 |
| `ndim` | Tensor的维度数量 |
| `dtype` | 数据类型 |
| `device` | Tensor所在设备 |
| `requires_grad` | 是否记录梯度 |

## 八、Tensor索引与切片

```python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(x[0])
print(x[0, 1])
print(x[:, 1])
```

含义：

- `x[0]`：读取第一行
- `x[0, 1]`：读取第一行第二列
- `x[:, 1]`：读取所有行的第二列

## 九、Tensor形状变换

```python
x.reshape(new_shape)
x.transpose(0, 1)
x.unsqueeze(0)
x.squeeze(0)
```

| 操作 | 作用 |
|---|---|
| `reshape` | 重新组织Tensor形状 |
| `transpose` | 交换两个维度 |
| `unsqueeze` | 增加一个长度为1的维度 |
| `squeeze` | 删除长度为1的维度 |

进行形状变换时，应保证元素总数不发生变化。

## 十、Tensor运算

### 1. 逐元素运算

```python
a + b
a - b
a * b
a / b
```

### 2. 矩阵乘法

```python
result = a @ b
```

`*`表示对应位置元素相乘，`@`表示矩阵乘法。

### 3. 聚合运算

```python
x.sum()
x.mean()
x.max()
x.min()
```

### 4. 拼接

```python
result = torch.cat([a, b], dim=0)
```

## 十一、Tensor操作检查原则

进行Tensor计算前，应重点检查：

1. Shape是否符合预期
2. 数据类型是否一致
3. Tensor是否位于同一设备
4. 模型输出与标签Shape是否匹配
5. 是否错误删除了Batch维或特征维

# Conda 与 PyTorch 环境配置

## 一、为什么使用独立环境

不同项目可能需要不同版本的：

- Python
- PyTorch
- NumPy
- CUDA 运行时
- 第三方依赖

如果全部安装到 `base` 环境中，容易出现版本冲突。

推荐原则：

```text
一个项目
→ 一个独立 Conda 环境
→ 固定 Python 版本
→ 固定依赖版本
→ 保存环境与运行记录
```

---

## 二、Anaconda 与 Miniconda

Anaconda 和 Miniconda 都包含 Conda，都可以创建和管理独立环境。

### Miniconda

特点：

- 只包含 Conda、Python 和少量基础工具
- 安装体积较小
- 适合服务器从零配置
- 后续按需安装依赖

### Anaconda

特点：

- 包含 Conda
- 预装较多数据分析和科学计算软件
- 体积较大
- 适合本地已有完整环境的情况

如果本机已经安装并能够正常使用 Anaconda，就不需要重复安装 Miniconda，可以直接使用 Anaconda 中的 Conda 创建课程环境。

在实验室服务器上，如果没有个人 Conda 环境，则可以按照课程步骤在个人目录安装 Miniconda。

---

## 三、创建独立 Conda 环境

创建课程环境：

```bash
conda create -n lesson03 python=3.10 -y
```

激活环境：

```bash
conda activate lesson03
```

检查当前环境：

```bash
conda env list
which python
python --version
python -m pip --version
```

应确认：

- 当前激活环境为 `lesson03`
- Python 路径属于 `lesson03`
- Python 版本为 3.10
- pip 与 Python 来自同一个环境

安装普通 Python 包时，推荐使用：

```bash
python -m pip install <包名>
```

这种方式可以明确使用当前 Python 对应的 pip，降低软件包安装到错误环境的风险。

---

## 四、Miniconda 服务器安装原则

服务器安装 Miniconda 前，应检查系统架构：

```bash
uname -m
```

Linux x86_64 服务器应使用对应安装包。

安装原则：

- 安装到个人 Home 目录
- 不使用 `sudo`
- 不覆盖其他用户目录
- 不在公共目录安装个人环境
- 不复制或同步整个 `miniconda3` 目录

初始化完成后检查：

```bash
conda --version
conda env list
```

---

## 五、Conda 镜像与 pip 镜像

Conda 和 pip 使用不同的包管理体系。

### Conda 镜像

用于下载：

- Conda 环境
- Python
- Conda 软件包

### PyPI 镜像

用于通过 pip 安装 Python 软件包。

课程使用清华 TUNA 镜像。

pip 镜像地址：

```text
https://pypi.tuna.tsinghua.edu.cn/simple
```

单次命令临时指定镜像：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>
```

该写法只影响当前命令，不修改全局 pip 配置。

镜像配置原则：

- 修改前先检查并备份
- 不混用多个国内镜像
- 不使用 HTTP 镜像
- 不关闭 SSL 验证
- 不随意添加 `trusted-host`

---

## 六、PyTorch、GPU 与 CUDA

### 1. 检查服务器 GPU

```bash
nvidia-smi
```

该命令可以查看：

- GPU 型号
- 显存容量
- 显存占用
- GPU 利用率
- 当前 GPU 进程
- NVIDIA 驱动版本

### 2. 检查 PyTorch

```python
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

### 3. 两类 CUDA 信息

`nvidia-smi` 中显示的 CUDA Version 表示 NVIDIA 驱动能够支持的 CUDA 版本上限。

PyTorch 实际使用的 CUDA 运行时应通过：

```python
torch.version.cuda
```

进行检查。

因此：

```text
nvidia-smi 能看到 GPU
```

不等于：

```text
当前 PyTorch 一定可以使用 GPU
```

还需要检查 PyTorch 安装版本、CUDA 运行时和设备可用性。

---

## 七、本地 CPU 环境

暂时无法连接服务器时，可以先完成本地 CPU 环境。

基本流程：

```bash
conda create -n lesson03 python=3.10 -y
conda activate lesson03
python -m pip install torch torchvision torchaudio
```

检查：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

本地没有 NVIDIA GPU 时：

```text
torch.cuda.is_available() = False
```

属于正常结果。

不需要为了让结果变成 `True` 而安装：

- CUDA
- cuDNN
- 模拟 GPU 工具

本地 CPU 环境可用于：

- 学习代码
- 数据处理
- 模型结构检查
- 小规模训练
- 基础调试

---

## 八、Jupyter 内核注册

在独立环境中安装：

```bash
python -m pip install ipykernel
```

注册内核：

```bash
python -m ipykernel install \
  --user \
  --name lesson03 \
  --display-name "Python 3.10 (lesson03)"
```

查看内核：

```bash
jupyter kernelspec list
```

注册后，在 Jupyter Notebook 中可以选择：

```text
Python 3.10 (lesson03)
```

注册新内核只是增加一个可选项，不会替换已有的 Jupyter 内核。

---

## 九、环境验证

完整验证可以包括：

```bash
which python
python --version
python -m pip --version
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"
```

张量计算验证：

```bash
python -c "import torch; x=torch.tensor([1.0,2.0,3.0]); print(x.sum().item())"
```

预期结果：

```text
6.0
```

---

## 十、常见问题排查

### 1. Conda 找不到

检查：

```bash
which conda
conda --version
```

服务器中还可以检查：

```bash
~/miniconda3/bin/conda
```

### 2. 软件包装到错误环境

检查：

```bash
which python
python -m pip --version
```

重点确认 Python 和 pip 是否来自同一个环境。

### 3. PyTorch 无法导入

检查：

```bash
python -m pip show torch
```

再保留完整导入报错：

```bash
python -c "import torch"
```

### 4. PyTorch 看不到 GPU

依次检查：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"
```

需要确认：

- 系统是否识别 GPU
- PyTorch 是否为 GPU 版本
- CUDA 运行时是否匹配
- 是否误装 CPU 版 PyTorch

### 5. 镜像中找不到软件包

先检查镜像是否同步该版本。

必要时可以在单条安装命令中临时指定其他可用来源，但不要随意修改全局配置。

---

## 十一、实验复现记录

环境记录至少应包含：

- Conda 环境名称
- Python 版本
- PyTorch 版本
- NumPy 版本
- CUDA 运行时
- GPU 型号
- 安装命令
- 验证命令
- 完整报错
- 问题解决过程

环境配置不仅是“安装成功”，还要保证：

```text
可检查
可复现
可排错
不包含敏感信息
```