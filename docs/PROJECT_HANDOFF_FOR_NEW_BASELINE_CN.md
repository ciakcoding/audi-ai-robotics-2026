# ADC 2026 项目完整交接：世界模型、Baseline、RL 与下一步

> 用途：在新的 Codex 对话中作为首要上下文。新对话开始时，应先完整阅读本文件，再阅读本文件列出的权威代码与文档。
> 更新日期：2026-07-24。
> 当前工作区：`D:\RLlearing-new`。

## 1. 老师的最新明确要求（最高优先级）

老师在群里明确说明：

1. 基础工作区设置参考：<https://robotlabor.github.io/humanoid-robot-fundamentals/workspace_setup/>
2. 必做任务虽然官方用词是 “drop”，实际要求是让球**轻微向前抛出**；不要求大力远投，但必须看得出**手臂摆动**。
3. 难度等级：
   - Level 0（必做）：抛球，机器人骨盆固定。
   - Level 1 option 1：抛到标记目标位置。
   - Level 1 option 2：骨盆不固定，机器人在抛球时保持平衡。
   - Level 2：抛到标记目标位置，同时机器人保持平衡。
   - Level 3：机器人向前行走并同时抛球。

### 当前判断

单机器人正式模型已经同时具备：

- 明确目标位置；
- 骨盆没有 weld 到世界；
- 躯干也没有结构性冻结；
- 非任务关节由位置执行器维持站立；
- 有实际的躯干高度、倾角和 fall 判定；
- 抛球动作有明显手臂摆动。

因此可以合理描述为“已经实现 Level 2 的技术配置”。但最终汇报时最好说 **“Level 2 implementation with scripted whole-body standing control”**，不要夸大成已解决通用动态平衡；当前平衡仍主要依靠 G1 的位置执行器和名义站立姿态。

## 2. 信息优先级

发生冲突时按以下顺序处理：

1. 老师的 PPT、任务 PDF、老师群消息；
2. 已冻结并验证的 Task 1 代码和参数；
3. 实际训练 metadata、evaluation CSV/JSON 和测试结果；
4. 项目中的说明文档；
5. 同学或旧版本总结的 Markdown。

同学总结的 Markdown 只能参考，不能覆盖老师要求或已验证代码。

## 3. 目录隔离与冻结规则

### Task 1：正式单机器人非学习 Baseline

- 目录：`D:\RLlearing-new\task1`
- 冻结分支：`simulationimprovement`
- 冻结 commit：`7a370663cbcc1aa96438dffc9f6331d3bf4ef35c`
- tag：`task1-baseline-v1.0`
- 备份：`D:\RLlearing-new\backups\task1-baseline-v1.0-7a37066.zip`
- 备份 SHA-256：`A9B9A8CF301C7DBDE684647BD1673D3D7AFF816D3CA6E8739C110075B4DCB19D`

**Task 1 已封存，不得为了 RL 或新实验直接修改。**

### Task 2：PPO 强化学习

- 目录：`D:\RLlearing-new\task2`
- 只使用冻结快照：`task2/vendor/task1_baseline_v1`
- 训练、模型、日志、评估、视频全部留在 `task2/`
- 不允许反向修改 Task 1 的物理世界来美化 RL 结果。

### Task 1 v2：两个机器人互相抛球的高级原型

- 目录：`D:\RLlearing-new\task1_v2`
- 与正式 Task 1、Task 2 均隔离；
- 它是给下一版 baseline 的概念原型，不替代冻结的单机器人 baseline。

## 4. 权威世界模型参数（新 Baseline 默认继承）

### 4.1 机器人

- 模型：MuJoCo Menagerie 官方 `unitree_g1/g1.xml`
- 版本：Unitree G1 **29DoF / 29 actuators**
- 正式模型路径：
  - `task1/assets/g1.xml`
  - 原始参考：`external/mujoco_menagerie/unitree_g1/g1.xml`
- 正式版本不能使用 `g1_with_hands.xml` 作为主模型。
- `g1_with_hands.xml` 会在 29 个身体执行器之外增加 14 个手指执行器，总计 43 个执行器；它不是老师要求的纯 29DoF 版本。
- 手部抓持功能可以抽象，不需要控制五指。
- 正式单机器人通过 `right_wrist_yaw_link` 表示手部功能。

### 4.2 球

- 球半径：`0.04 m`（老师/团队已确认；直径 `0.08 m`）
- 球质量：`0.05 kg`
- 类型：sphere
- 自由关节：`throw_ball_free`
- 单机器人 XML 初始位置：`(0.35, -0.30, 1.20) m`
- collision：`contype=1`、`conaffinity=1`、`condim=3`
- 编译后的摩擦：`(1.0, 0.005, 0.0001)`
- 球必须能正常落地并滚动，不能穿入地面。
- 视频中的“球不穿地”视觉修正只允许用于渲染展示，不能改变正式评估所使用的物理状态。

### 4.3 地面、重力和仿真

- 重力：`(0, 0, -9.81) m/s²`
- MuJoCo timestep：`0.002 s`
- 控制周期：`0.02 s`
- 每次控制的物理 substeps：`10`
- integrator：`implicitfast`
- solver：Newton
- solver iterations：`100`
- 单机器人 episode：`1.8 s`
- floor：世界坐标地面 plane
- floor collision：`contype=1`、`conaffinity=1`、`condim=3`
- 编译后的 floor friction：`(1.0, 0.005, 0.0001)`

### 4.4 目标与成功判定

- 正式单机器人目标中心：`(0.55, 0.00, 0.00) m`
- 目标显示几何体：不参与碰撞
- 成功半径：`0.10 m`
- 以球和地面的**第一次接触位置**作为 landing point；
- landing error：球第一次接触地面时，球中心 XY 与目标中心 XY 的欧氏距离；
- success：`landing_error_xy <= 0.10 m` 且机器人没有 fall。
- Reward 不能替代最终 success/landing error 定义。

### 4.5 抓持、释放和 weld

- `hold_throw_ball` weld **应该存在**：它是允许的手部抓持功能抽象。
- 单机器人正式 weld：
  - `body1="right_wrist_yaw_link"`
  - `body2="throw_ball"`
  - `relpose="0.105 0.039 0.015 1 0 0 0"`
- 球已经人工校准到掌心内侧，不在手背、掌根或手掌内部穿模。
- 释放只关闭持球 weld；之后球由 MuJoCo 按释放瞬间的运动状态、重力和碰撞自由飞行。
- scripted release time：`0.65 s`
- `glue_robot_to_world` 不应存在于 Level 2 正式模型。
- `freeze_torso` 不应存在于 Level 2 正式模型。
- 非任务关节可以由明确记录的位置控制保持名义姿态，但不能用隐藏 weld 假装平衡。

### 4.6 稳定性

- 正式模型 floating base 保留；
- 当前 fall 判断：
  - torso height `< 0.60 m`，或
  - torso tilt `> 45°`
- Task 1 验证中：
  - passive 2.0 s 无 fall；
  - 100 回合 fall rate `0%`；
  - 最大 torso tilt 约 `2.798°`。

## 5. 已修复的旧 Baseline 问题

旧同学版本曾有以下问题，新的 baseline 不应重新引入：

1. 使用带手指模型导致 43 actuators，而不是 29。
2. mesh 缺失、路径不可复现。
3. XML、环境构造器和 viewer 的目标位置相互冲突。
4. `_ball_vel()` 把角速度当成线速度。
5. 没有第一次触地事件和统一 landing metric。
6. action index 与 MuJoCo actuator ID 混用。
7. 出现不可能的 `-270°` 肩关节命令。
8. 文档说 learned release，实际却是 scripted release。
9. pelvis/torso weld 导致不能诚实评价平衡和 fall。
10. 没有使用相同指标的确定性 baseline evaluator。

修复记录见：

- `WORLD_MODEL_PARAMETERS.md`
- `TASK1_BASELINE_ISSUES_AND_FIXES.md`
- `TASK1_FROZEN.md`

## 6. 正式单机器人 Baseline 的现状

- 模型能编译为 29 actuators；
- contract tests：`2/2` 通过；
- 确定性评估：`100/100` 成功；
- 当前冻结版本 README 报告：
  - mean first-contact landing error：约 `1.15 cm`
  - fall rate：`0%`
- 双击运行：`task1/run_baseline.bat`
- PowerShell：`task1/run_baseline.ps1`
- evaluator：`task1/scripts/evaluate_baseline.py`
- 实际动作源：`task1/scripts/view_baselines v031.py` 中的 `OptionCSwingPolicy`
- 环境：`task1/envs/g1_fixed_body_throw_env.py`
- 世界 XML：`task1/assets/scene_throw.xml`

## 7. 两机器人互相抛球原型（明天新 Baseline 的直接起点）

当前原型：`task1_v2/baseline.py`

设计目标：

- 两个官方 G1 29DoF 实例；
- 两个 floating base 都保留；
- 两个机器人面对面；
- A 抛给 B，B 接住后再抛回 A；
- 默认要求 6 次交替 exchange；
- fully scripted、deterministic、non-learning；
- 只抽象抓持和接球，不做五指抓握；
- 每个机器人使用 7 个右臂关节；
- 抛球时只关闭当前持球 weld；
- 进入接球手的 catch radius 后开启接球方 weld 并交换角色。

当前代码常量：

- robot center distance：`1.2 m`
- ball radius：`0.04 m`
- ball mass：`0.05 kg`
- catch radius：`0.20 m`
- physics dt：`0.002 s`
- control dt：`0.02 s`
- arm prepare/swing time：`0.40 s`
- release progress：轨迹的 `55%`
- recovery time：`0.55 s`
- minimum catch time：`0.25 s`
- maximum flight time：`0.80 s`

运行：

```powershell
D:\mujoco_rl_env\python.exe task1_v2\baseline.py --exchanges 6 --viewer
```

或双击：

```text
task1_v2/RUN_DEMO.bat
```

### 重要警告

`task1_v2/results/summary.json` 是旧运行遗留结果，里面仍写着 `2.0 m` 和 `two fixed-base instances`，与当前代码/README 的 `1.2 m`、floating base 不一致。明天不能把这个旧 summary 当作权威结果；完成新 baseline 后必须重新运行并覆盖/另存新的结果。

### 新 Baseline 应优先完成

1. 从冻结单机器人世界参数出发，不修改球、重力、timestep 和正式 G1 来源。
2. 两个机器人都不得 weld 到世界。
3. 把站立/平衡控制写清楚，区分“位置执行器保持站立”与真正 learned balance。
4. 让球的初始位置、抓持点和接球点都由手部 site 明确定义，避免穿模。
5. catch weld 只能在合法接球条件满足时开启。
6. 记录每次：
   - thrower/receiver
   - release time
   - flight time
   - receiver-hand minimum distance
   - catch success
   - ball floor contact
   - 两台机器人的 torso height/tilt/fall
7. 重新生成 `exchanges.csv` 和 `summary.json`，不能沿用旧结果。
8. 增加测试确认：
   - 每个机器人恰好 29 actuators；
   - 两个 floating base 存在；
   - 没有 world/pelvis weld；
   - 球半径和质量正确；
   - A/B 角色按预期交替；
   - 球未接住时会真实落地；
   - viewer 和 evaluator 使用同一参数。

## 8. PPO RL 已有工作

### 方法

- 算法：PPO，Stable-Baselines3；
- 不是完全从零动作学习，而是 **residual PPO around the frozen scripted baseline**；
- 这是合法的行为先验/混合控制，但必须如实披露。
- observation：33 维；
- action：7 维右臂 residual correction；
- residual scale：`0.2`；
- release time 固定为 `0.65 s`；
- policy MLP：`[256, 256]`；
- learning rate：`1e-4`；
- `n_steps=256`、`batch_size=256`、`n_epochs=5`；
- `gamma=0.99`、`GAE lambda=0.95`；
- PPO clip：`0.15`；
- entropy coefficient：`0.001`；
- `8` 个并行环境；
- seed：`2026`；
- CPU 负责并行 rollout，CUDA 负责 PPO update。

### 固定目标训练进步证据

权威 run：

`task2/runs/ppo_fixed_target_progress_evidence_100k`

同一协议：100 回合、相同 seeds、目标 `(0.55, 0)`、success radius `0.10 m`、额外右臂初始扰动 `±0.08 rad`。

| steps | success | mean error | p90 error | mean reward |
|---:|---:|---:|---:|---:|
| 0 | 100% | 1.168 cm | 1.381 cm | 65.74 |
| 25k | 100% | 1.052 cm | 1.240 cm | 66.32 |
| 50k | 100% | 0.892 cm | 1.114 cm | 67.11 |
| 75k | 100% | 0.858 cm | 1.114 cm | 67.20 |
| 100k | 100% | 0.708 cm | 0.968 cm | 67.94 |

结论：成功率一开始就饱和，所以主要证据是 landing precision 和 reward 随训练改善；0 到 100k 的 mean error 改善约 `39.4%`。

### 1M robust run

权威 run：

`task2/runs/ppo_robust_1m_seed2026_20260723`

正式 nominal 对比：

- baseline：100% success，mean error `1.150 cm`，0 fall；
- PPO best：100% success，mean error `0.674 cm`，0 fall；
- mean error 改善约 `41.4%`。

`±0.08 rad` joint noise：

- baseline mean error：`1.179 cm`
- PPO best mean error：`0.754 cm`
- mean error 改善约 `36.0%`

注意：1M run 的 **final model 明显退化**，nominal success 只有 `16%`；必须使用 `best_model.zip`，同时保留 final 和 checkpoints 作为“训练后期可能退化”的工程证据。

### 视频用模型

训练进度视频使用的是以下一组，而不是 1M robust run 的 100k/200k/... checkpoints：

- `initial_model.zip`
- `ppo_throw_25000_steps.zip`
- `ppo_throw_50000_steps.zip`
- `ppo_throw_75000_steps.zip`
- `ppo_throw_100000_steps.zip`

它们位于：

`task2/runs/ppo_fixed_target_progress_evidence_100k/`

视频和证据由：

- `task2/scripts/evaluate_training_progress.py`
- `task2/scripts/render_training_progress_media.py`

生成。

视频必须：

- 每阶段持续足够长；
- 同一连续镜头显示抛出、第一次落地和之后滚动；
- 不在落地/滚动之间生硬切镜头；
- 球接触地面时视觉上不能卡进地里；
- 明确标出 0/25k/50k/75k/100k。

## 9. 已冻结/不再使用的 RL 方向

- 随机目标、goal-conditioned 和手动目标实验已封存；
- 说明：`task2/TARGET_RANDOMIZATION_FROZEN.md`
- 原因：组长最终需要的是“训练过程中逐渐变好”的视频、截图和数据，不再要求用目标变化制造 baseline/RL 的巨大差距。
- 不得删除这些实验，但最终固定目标证据不要与它们混在一起。

## 10. 已完成的文档与云端证据

- Google Drive 目标文件夹：
  <https://drive.google.com/drive/folders/1jbC6QGj9VHCN7tWO3IF3wvFarZsw1NVL>
- RL 证据子文件夹：
  <https://drive.google.com/drive/folders/1LZSIcsyZBOXjD1EAqScIMBMd7uFAzREJ>
- 已填写英文工程笔记：
  `task2/docs/RL_ADC2026_Completed.docx`
- Drive 中原始 `RL_ADC2026.docx` 未覆盖，完成版名为：
  `RL_ADC2026_Completed.docx`

已上传证据包括：

- `TRAINING_PROGRESS_EVIDENCE.md`
- `ppo_training_progress.mp4`
- `training_progress_metrics.png`
- `ppo_training_progress_montage.png`
- `checkpoint_summary.csv`
- `summary.json`
- `stage_100000_first_contact.png`
- `stage_100000_after_roll.png`

## 11. GitHub 状态

### Task 1

- 分支：`simulationimprovement`
- 已冻结，不得继续改。

### Task 2 / RL

- 本地发布 checkout：`D:\RLlearing-new\feature_rl_publish`
- 远端：`https://github.com/aim-t/audi-ai-robotics-2026.git`
- 分支：`feature/rl`
- 已有 commit：`ef0db23 Add reproducible residual PPO pipeline and evidence`
- 绝对不能推送到 `main`。

截至本交接文件写入时，远端已经有 1M robust run 的 selected best/final/100k-1M checkpoints，但视频使用的独立 0/25k/50k/75k/100k fixed-target progress 模型需要单独确认并补传。补传后应放在：

`outputs/models/training_progress/`

## 12. 环境

- 推荐并已验证的可复用 Python：
  `D:\mujoco_rl_env\python.exe`
- Python 3.11；
- 能运行 MuJoCo、Gymnasium、Stable-Baselines3；
- CPU 并行环境 + GPU PPO update 已成功训练；
- 不要用系统 Python 3.13 直接运行 MuJoCo 项目；
- `task1_v2/baseline.py` 会在缺依赖时尝试自动切换到该 Python。

## 13. 新对话的开场指令建议

把本文件交给新对话后明确说：

> 先完整阅读 `PROJECT_HANDOFF_FOR_NEW_BASELINE_CN.md`，再检查其中列出的权威代码。我要写新的 baseline。Task 1、Task 1 v2 和 Task 2 必须严格隔离；老师要求和冻结世界模型参数不能擅自改变。不要信任旧 `task1_v2/results/summary.json`，必须重新验证当前代码。
