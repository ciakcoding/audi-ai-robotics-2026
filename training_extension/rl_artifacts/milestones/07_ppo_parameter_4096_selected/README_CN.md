# 冻结 PPO 参数残差模型（累计 4,096 投篮）

## 结论

这是当前选中的真实 RL 策略，不是 CEM，也不是最终训练步的模型。

- 算法：PPO（on-policy）
- 专家中心：冻结 CEM v17
- 目标中心：`(2.2, 0.0, 1.2)`
- 固定有效半径：`0.10 m`
- 动作：15 个轨迹参数的极小残差；释放时机冻结
- 轨迹：保留两步走路、下蹲、腿部伸展、双手举球、右手随挥和双手分离恢复
- 禁止：根节点平移走路、扣篮、手伸入篮圈、篮板球、跌倒

## 300 个未见种子的固定评估

| 策略 | 成功 | 平均穿圈误差 | 最大误差 | 篮板 | 跌倒 |
|---|---:|---:|---:|---:|---:|
| 冻结 CEM v17 | 293/300 | 3.865 cm | 38.514 cm | 0 | 0 |
| PPO 1,024 | 294/300 | 3.701 cm | 38.956 cm | 0 | 0 |
| **PPO 4,096（本冻结版）** | **294/300** | **3.545 cm** | 39.141 cm | **0** | **0** |

与冻结 CEM 相比，选中 PPO 的平均误差下降约 8.3%，成功数增加 1。
限制是最大离群误差增加约 0.63 cm；这项结果没有隐藏。

## 代表视频（seed 100000）

- 直接进球：YES
- 穿圈误差：2.69 cm
- 球的空中水平距离：1.69 m
- 释放时球距篮圈：1.70 m
- 最近的手距篮圈：1.08 m
- 篮板接触：NO
- 跌倒：NO

视频：`ppo_4096_seed100000.mp4`

## MuJoCo 交互播放

从项目根目录运行：

```powershell
& "D:\mujoco_rl_env\python.exe" -m training_extension.view_ppo_parameters
```

或直接运行脚本：

```powershell
& "D:\mujoco_rl_env\python.exe" "D:\RLlearing-new\audi-ai-robotics-2026-simulation03\training_extension\view_ppo_parameters.py"
```

## 文件

- `selected_model.zip`：PPO actor/value 网络
- `selected_vecnormalize.pkl`：与模型匹配的观测归一化状态
- `evaluation_300_summary.json`：300 回合汇总
- `evaluation_300_episodes.json`：逐回合结果
- `finetune_metadata.json`：恢复训练元数据
- `source_snapshot/`：冻结时使用的环境、训练、评估、渲染源码
- `SHA256SUMS.txt`：完整性校验
