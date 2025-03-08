# 动态温度Goodness增强前向-前向学习 (DTG-FF)

本仓库实现了论文"Dynamic Temperature Goodness for Enhanced Forward-Forward Learning"中描述的动态温度Goodness (DTG)机制。DTG-FF系统地解决了原始前向-前向算法的基本限制，同时保持其优雅性和生物学合理性。

## 概述

前向-前向(FF)算法由Geoffrey Hinton提出，通过对正负数据使用两次前向传递来替代反向传播。然而，FF中固定的判别标准限制了其在复杂数据集上的性能。

DTG-FF通过以下方式增强FF算法：

1. **动态温度机制**：基于特征清晰度自适应调整判别标准
2. **特征驱动的优化框架**：结合温度调整、历史稳定化和自适应边界
3. **自适应阈值**：为每一层学习最佳判别边界

## 核心组件

### DTGLayer

实现动态温度goodness计算的核心构建模块。每一层：
- 计算特征统计量以确定清晰度
- 基于特征分布动态调整温度
- 使用带温度缩放的归一化L2距离计算goodness
- 通过softplus激活维持自适应阈值

### FF_DTG_Model

完整的模型架构：
- 处理正负样本
- 使用动态边界和自适应阈值计算损失
- 提供每层准确率指标
- 通过特征拼接支持分类任务

### FF_DTG_Config

管理配置的类：
- 数据集设置（MNIST/CIFAR-10）
- 模型架构（层数，隐藏维度）
- 训练参数（学习率，权重衰减）

## 性能

DTG-FF达到：
- MNIST上98.7%的准确率
- CIFAR-10上60.1%的准确率（比原始FF提高11.11%）

## 环境要求

- PyTorch >= 2.4.0
- torchvision
- numpy
- matplotlib (用于可视化)

## 使用方法

```python
# 创建配置
config = FF_DTG_Config()
config.input.dataset = "cifar10"  # 选择数据集
config.model.num_layers = 4       # 设置架构深度

# 创建模型
model = FF_DTG_Model(config)

# 训练循环（示例）
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config.training.learning_rate,
    weight_decay=config.training.weight_decay
)

for epoch in range(config.training.epochs):
    # 您的训练循环实现
    # ...
    outputs = model(batch_inputs, batch_labels)
    loss = outputs["Loss"]
    loss.backward()
    optimizer.step()
```

## 引用

如果您使用此代码，请引用原论文：

```
@article{DTG-FF-2025,
  title={Dynamic Temperature Goodness for Enhanced Forward-Forward Learning},
  author={Yucheng Chen},
  booktitle={ICCV},
  year={2025}
}
```