# Watch Your Step：面向人形机器人运动的预判感知

<p align="center">
  <a href="https://leggedrobotics.github.io/watch-your-step/"><b>项目主页</b></a> &nbsp;·&nbsp;
  <a href="https://youtu.be/3bLU6_OPv6s"><b>演示视频</b></a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2610.29612"><b>arXiv 论文</b></a> &nbsp;·&nbsp;
  <a href="#引用"><b>引用</b></a>
</p>

<p align="center">
  Matthias Heyrman, Chenhao Li, Ioannis Dadiotis, Marco Hutter<br>
  苏黎世联邦理工学院机器人系统实验室（Robotic Systems Lab）与 ETH AI Center
</p>

<p align="center">
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/blocks.webp" alt="障碍块，以及策略读取的两个局部地形块" width="32%"></a>
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/gaps.webp" alt="沟隙地形" width="32%"></a>
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/drops.webp" alt="落差地形" width="32%"></a>
  <br>
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/course.webp" alt="障碍路线" width="32%"></a>
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/corridor.webp" alt="仿真中的围墙走廊" width="32%"></a>
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/robot_view.webp" alt="机器人在楼梯上获取的机载高程地图" width="32%"></a>
</p>

本项目提供基于 Isaac Lab 的宇树 G1 人形机器人感知运动训练与评估代码，使用落脚点预测模型（Foothold Prediction Model，FPM）辅助运动控制。FPM 根据机器人本体状态和速度指令预测接下来的两个落脚点，策略网络只读取这些落脚点周围的 `7 × 7` 局部高程块。FPM 与运动策略共同训练。

原项目展示了策略向真实机器人的零样本迁移，实机实验和浏览器中的策略演示见[项目主页](https://leggedrobotics.github.io/watch-your-step/)。本仓库已接入 G1_23（23 自由度）资产，并保留原 G1（29 自由度）任务；G1_23 的 Isaac Sim 动态运行与训练效果尚未验证。

## 安装

依赖 Isaac Sim 5.1、Isaac Lab 2.3.0 和 `rsl-rl-lib==3.3.0`。在 Isaac Lab 的 Python 环境中执行：

```bash
pip install -e .
```

## 项目结构

```text
fpm/
  foot_pred/    FPM 网络、落脚点标签生成、模型重新拟合与训练调度
  mdp/          落脚点局部高度扫描、奖励、事件、指令和对称数据增强
  terrains/     训练地形、运动能力验证地形和走廊验证地形
  env_cfg.py    训练、播放和验证环境配置
  agents/       PPO 算法与网络配置
  assets/       G1（29 自由度）和 G1_23（23 自由度）的 USD 资产与独立配置
scripts/
  train.py                  联合训练运动策略和 FPM
  play.py                   播放策略并导出模型
  validate_locomotion.py    验证运动能力
  validate_corridor.py      验证走廊穿越能力
  validate_foot_pred.py     验证落脚点预测误差
demo/           原 G1（29 自由度）的预训练策略和 FPM
```

## 训练

```bash
python scripts/train.py --headless --seed 1
```

脚本默认训练 G1_23，任务名称为 `FPM-G1-23-v0`。机器人配置是 `fpm/assets/__init__.py` 中的 `G1_23_CFG`，训练日志保存在 `logs/rsl_rl/g1_23_fpm/`。

如需训练原来的 29 自由度 G1，添加 `--task FPM-G1-v0`，日志保存在 `logs/rsl_rl/g1_fpm/`。两种机器人均使用 `0.25` 的关节位置动作缩放系数；关节镜像增强根据机器人的实际关节名称和顺序生成映射。

前 500 次迭代中，落脚点固定在机体下方。之后每隔 500 次迭代，使用当前策略采集的 `5 × 10^5` 条状态转移数据，从头重新训练 FPM。FPM 检查点与策略检查点保存在同一目录，文件名为 `foot_model_<iteration>.pt`，采用 TorchScript 格式并包含输入归一化参数。恢复训练时，使用 `--resume <run>/model_<iteration>.pt` 指定策略检查点。

## 策略播放与模型导出

```bash
python scripts/play.py --show_footholds
```

未指定 `--checkpoint` 时，脚本会从所选机器人的日志目录中查找最新策略，默认目录为 `logs/rsl_rl/g1_23_fpm/`。G1_23 需要先完成训练；仓库附带的演示策略和 FPM 适用于 29 自由度 G1。播放原 G1 演示策略的命令为：

```bash
python scripts/play.py --task FPM-G1-Play-v0 --show_footholds
```

播放脚本会将 `policy.pt`、`policy.onnx` 和 `foot_model.onnx` 导出到检查点所在目录的 `exported/` 子目录中。FPM 输出机体坐标系下两个落脚点的平面坐标；G1_23 的系统状态输入为 58 维，原 G1 为 70 维。

G1_23 的主要输入输出维度如下：

| 项目 | 维度 | 说明 |
| --- | --- | --- |
| 动作 | 23 | 各可控关节的位置动作 |
| 策略网络观测 | 183 | 本体状态、速度指令、上一步动作与落脚点扫描 |
| 价值网络观测 | 876 | 包含完整地形高度扫描，共 693 个采样点 |
| 落脚点扫描 | 102 | 两只脚各自的平面落脚点坐标和 `7 × 7` 高度块 |
| FPM 系统状态输入 | 58 | 机体状态、速度指令、关节相对位置和相对速度 |

播放和验证时，策略检查点所在目录中需要有训练得到的 `foot_model_*.pt`。添加 `--joystick` 可使用手柄控制，添加 `--video` 可录制视频。

## 评估

各验证脚本从同一目录加载策略检查点与最新的 `foot_model_*.pt`。设置 `PYTHONHASHSEED=0`，可使不同运行中的走廊地形保持一致。

以下命令使用 Bash：

```bash
export PYTHONHASHSEED=0
python scripts/validate_locomotion.py --headless --checkpoint logs/rsl_rl/g1_23_fpm/<run>/model_40000.pt
python scripts/validate_corridor.py --headless --checkpoint logs/rsl_rl/g1_23_fpm/<run>/model_40000.pt
python scripts/validate_foot_pred.py --headless --checkpoint logs/rsl_rl/g1_23_fpm/<run>/model_40000.pt
```

验证脚本同样默认使用 G1_23。验证 29 自由度 G1 的检查点时，通过 `--task` 分别选择 `FPM-G1-Validate-Locomotion-v0`、`FPM-G1-Validate-Corridor-v0` 或 `FPM-G1-Validate-FootPred-v0`。

添加 `--save_results`，可将每回合的评估结果保存到对应运行目录。

## 引用

```bibtex
@misc{heyrman2026watchstepanticipatedperception,
  title         = {Watch Your Step: Anticipated Perception for Humanoid Locomotion},
  author        = {Matthias Heyrman and Chenhao Li and Ioannis Dadiotis and Marco Hutter},
  year          = {2026},
  eprint        = {2610.29612},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.29612}
}
```

## 许可证

© 2026 ETH Zurich<br>
原作者：Matthias Heyrman、Chenhao Li、Ioannis Dadiotis 和 Marco Hutter<br>
本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

第三方文件保留各自的原始许可证，详见 [licenses/dependencies](licenses/dependencies)：

- `fpm/assets/data/G1/`：来自 [unitree_ros](https://github.com/unitreerobotics/unitree_ros) 的宇树 G1 模型（29 自由度，rev 1.0），经转换为 USD 格式，采用 BSD-3-Clause 许可证。
