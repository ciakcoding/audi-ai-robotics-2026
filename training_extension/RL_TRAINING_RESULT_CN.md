# LEVEL03 投篮 RL 训练结果（2026-07-26）

## 当前结论

已得到一个进一步改善的可复现 PPO 结果，并冻结在：

`training_extension/frozen/ppo_parameters_12288_selected_20260726/`

选中的是累计 12,288 个完整投篮 episode 的检查点，不是最后的 14,848。
模型使用 RTX 4060 Laptop GPU 做网络更新，8 个 MuJoCo 环境在 CPU 并行
收集物理轨迹。每 512 个投篮均保存检查点。MLP 很小而 MuJoCo 仿真主要在
CPU，因此 GPU 利用率不高，但训练设备确实为 CUDA。

## 为什么最终采用参数残差 PPO

每 20 ms 输出多关节 residual 的 PPO/TD3 会在约 520 个控制时刻累积偏移，
容易破坏已经合格的 CEM 动作。最终环境让 RL 每个 episode 只决定一次小幅
关键帧参数残差，之后仍通过原有三次平滑插值执行完整动作。

这仍是真实 PPO 强化学习：保存了可部署 policy/value 网络，policy 根据初始
物理观测输出动作，并从 MuJoCo episode reward 更新；CEM v17 只作为专家中心。

## 不变物理合同

- 世界模型与物理篮圈不变。
- 篮圈中心 `(2.2, 0.0, 1.2)`。
- 球心穿越半径固定为 `0.10 m`。
- 球必须下降穿越篮圈平面。
- 释放时球距篮圈至少 `1.10 m`。
- 释放时骨盆距篮圈至少 `1.20 m`。
- 空中水平距离至少 `1.00 m`。
- 最近的手距篮圈至少 `0.45 m`。
- 碰篮板、跌倒、扣篮或伸手入圈均不能算成功。

## 同一 300-seed 比较

| 策略 | 成功 | 平均误差 | 最大误差 | 篮板 | 跌倒 |
|---|---:|---:|---:|---:|---:|
| CEM v17 | 293/300 | 3.865 cm | 38.514 cm | 0 | 0 |
| PPO 1,024 | 294/300 | 3.701 cm | 38.956 cm | 0 | 0 |
| PPO 4,096（旧选中） | 294/300 | 3.545 cm | 39.141 cm | 0 | 0 |
| PPO 12,288（新选中） | 297/300 | 3.336 cm | 31.255 cm | 0 | 0 |

新 PPO 相对旧 PPO 的平均穿圈误差下降约 5.9%，成功数增加 3，最大离群
误差下降约 20.1%。相对 CEM 的平均误差下降约 13.7%，成功数增加 4。

补充的恢复期复评使用相同 `300` 个 seeds，并在正常穿圈评分终点后继续
仿真 `10 s`。结果为穿圈前 `0/300` 跌倒、恢复期 `0/300` 跌倒、最终帧
`0/300` 跌倒；最低骨盆高度 `0.6980 m`，最大骨盆俯仰/横滚绝对值分别为
`2.064°` / `0.611°`。这修正了旧评估只检查到穿圈时刻的覆盖范围说明。

## 已保留的训练里程碑

- `td3_smoke_10k_seed2026_20260726`：TD3 崩塌。
- `td3_smoke_zero_init_10k_seed2027_20260726`：零均值初始化后仍崩塌。
- `td3_smoke_filtered_20k_seed2028_20260726`：低通后仍崩塌。
- `ppo_residual_smoke_25k_seed2030_20260726`：逐帧 PPO 未崩塌但精度下降。
- `sac_parameters_gate_2k_seed2043_20260726`：尾部误差改善，平均误差下降失败。
- `ppo_parameters_gate_2k_seed2051_20260726`：首次得到正向 1,024 检查点。
- `ppo_parameters_finetune_4k_from1024_seed2052_20260726`：从 1,024 保守精修，
  保存累计 1,536 到 5,120 的每 512-step 检查点；最终选择 4,096。
- `ppo_parameters_continue_8k_from4096_seed2053_20260726`：继续到 12,288，
  保存全部 16 个里程碑；300-seed 选择阶段最佳 10,752。
- `ppo_parameters_refine_4k_from10752_seed2054_v2_20260726`：用更小学习率
  精修到 14,848；最终选择累计 12,288，而不是 final。

完整 run 保留在本地忽略目录 `training_extension/runs/`。GitHub 分支只收录
可复现源码、最终冻结模型、完整固定评估和紧凑失败/中间里程碑；不上传
TensorBoard、replay buffer 和重复检查点。同学原始
`scripts/view_baselines_LEVEL03_v031!.py` 没有被修改。

## 播放

```powershell
cd D:\RLlearing-new\audi-ai-robotics-2026-simulation03
& "D:\mujoco_rl_env\python.exe" -m training_extension.view_ppo_parameters
```

视频：

`training_extension/frozen/ppo_parameters_12288_selected_20260726/ppo_12288_seed100000.mp4`
