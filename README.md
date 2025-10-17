# DQN在highway-env环境下的实现

这个项目实现了使用Deep Q-Network (DQN)算法在highway-env环境中训练自动驾驶智能体。项目使用stable-baselines3框架实现DQN算法，并在highway-fast-v0环境中进行了实验。

## 项目结构

```
Project/
├── logs/              # tensorboard日志文件
├── models/            # 保存的模型文件
└── videos/            # 记录的视频文件
```

## 环境配置

1. 使用conda创建并激活环境：
```bash
conda env create -f environment.yml
conda activate DQN_demo
```

2. 主要依赖包：
- gymnasium
- highway-env
- stable-baselines3
- pytorch
- numpy
- opencv-python

## 使用说明

1. 训练模型：
   - 运行`demo.py`文件进行模型训练
   - 可以通过修改学习率（lr）和折扣因子（gamma）等超参数来调整训练效果

2. 查看训练过程：
```bash
tensorboard --logdir=Project/logs
```

3. 模型评估：
   - 训练完成后的模型将保存在`Project/models`目录下
   - 评估过程中的视频将保存在`Project/videos`目录下

## 实验结果

项目包含了不同超参数配置下的训练结果：
- 学习率：0.0005, 0.001
- 折扣因子：0.8, 0.95

## 注意事项

- 确保系统已安装CUDA（如果使用GPU训练）
- 建议使用Python 3.10或以上版本
- 训练过程中会自动创建必要的目录结构

## 贡献指南

欢迎提交Issues和Pull Requests来改进这个项目。

## 许可证

MIT License
=======
# DQN_using_highway-env
该项目是 "Autonomous-Driving-RL-Agents" 的核心部分之一，专注于使用深度Q网络 (Deep Q-Network, DQN) 算法来训练一个自动驾驶智能体，智能体在 highway-env 模拟环境中学习驾驶，目标是在多车道的高速公路上实现高效、安全的自主导航。
>>>>>>> 69ec68b5931a1bfa90f3af953a61a5d87ba14f79
