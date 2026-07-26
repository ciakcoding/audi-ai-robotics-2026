# LEVEL03 修改版 baseline（不含 CEM / RL）

本目录基于同学当前的
`scripts/view_baselines_LEVEL03_v031!.py`。同学原文件没有被修改；代码通过
继承策略并覆盖少量关键帧实现修改。

主要修改：

- 减少第一步和第二步落地时的后跟砸地与膝关节锁死；
- 保留两步物理走路、下蹲、腿部伸展和抛球；
- 限制右手腕过度折叠；
- 让左引导手在释放后向外下落，避免双臂交叉穿模；
- 将物理篮圈中心改为 `(2.2, 0.0, 1.2)`；
- 使用 16 段物理圆环和固定 `0.10 m` 球心穿越标准。

运行：

```powershell
cd D:\path\to\audi-ai-robotics-2026
python -m pip install -r requirements.txt
python -m training_extension.view_derived_baseline
```

Windows 建议把仓库克隆到纯英文路径，例如
`D:\projects\audi-ai-robotics-2026`。MuJoCo 3.10 在绝对路径包含中文字符时
可能无法打开 XML，这属于底层路径编码限制。

评估和质量检查：

```powershell
python -m training_extension.evaluate_derived_baseline --episodes 100
python -m unittest discover -s training_extension/tests -v
python -m training_extension.quality_check --smoke-episodes 2
```

固定 seeds `17000..17099` 的实际结果：

- 成功：`0/100`
- 平均穿圈平面误差：`1.1032 m`
- 最大误差：`1.1068 m`
- 碰篮板：`0`
- 跌倒：`0`

这个结果刻意保留为真实的无 CEM 对照，不能用后续 CEM 参数冒充 baseline。
`artifacts/baseline_motion_milestones/` 还保存了两个精选动作迭代版本，用于
展示脚步和投球动作的改进过程；它们不会被运行时代码导入。

该分支不需要任何 CEM `state.json`，也不包含 PPO、TD3、SAC、模型权重或
TensorBoard 文件。CEM 优化单独放在 `feature/simulation03-cem`。
