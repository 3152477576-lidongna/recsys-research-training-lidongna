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