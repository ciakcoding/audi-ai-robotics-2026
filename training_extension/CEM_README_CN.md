# LEVEL03 CEM 轨迹优化

该分支建立在 `feature/simulation03-derived-baseline` 之上，只增加 CEM
轨迹优化、最终参数和精选里程碑。世界模型、目标 `(2.2, 0.0, 1.2)` 和固定
`0.10 m` 球心成功半径不变。

**CEM 是开环轨迹优化，不是强化学习。**

最终冻结结果：

- 成功：`20/20`
- 平均穿圈误差：`3.03 cm`
- 直接进球：`20/20`
- 碰篮板：`0`
- 跌倒：`0`

在同学最新版 `v031` 上重新执行 seeds `50000..50099`：

- 成功/直接进球：`98/100`
- 平均穿圈误差：`3.64 cm`
- 最大离群误差：`36.73 cm`
- 碰篮板：`0`
- 跌倒：`0`

两个 miss 的逐回合数据保存在
`cem_artifacts/selected/evaluation100_seed50000/episodes.json`，没有把
98/100 写成 100/100。

无 CEM 的父分支 baseline 是 `0/100`，平均穿圈平面误差 `1.1032 m`。

直接播放：

```powershell
python -m training_extension.view_direct
```

重新验证：

```powershell
python -m training_extension.validate_direct `
  --state training_extension/cem_artifacts/selected/state.json `
  --episodes 100 --seed 50000 --workers 8 `
  --output outputs/cem_eval100_seed50000
```

`cem_artifacts/milestones/` 保存了伸手入框、低位横扫、单手化、双手间距
过大、第二步落地和投后双臂交叉等阶段，包含状态、验证和精选视频，用于
展示为什么仅有数值成功仍不能接受。

本分支不包含 PPO、TD3、SAC、RL 模型、replay buffer 或 TensorBoard。
