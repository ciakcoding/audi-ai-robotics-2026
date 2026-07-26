# LEVEL03 RL 上传范围与复现清单

正式分支：`feature/rl-on-lv3`。草稿 PR 以
`feature/simulation03-cem` 为 base，不直接修改 `main`。

## 分支关系

RL 使用独立的叠加分支：

```text
feature/simulation03
  -> feature/simulation03-derived-baseline
    -> feature/simulation03-cem
      -> feature/rl-on-lv3
```

RL PR 以 `feature/simulation03-cem` 为 base，绝不直接指向 `main`。

## 建议上传

### 训练、评估与播放代码

- `training_extension/sac_parameter_env.py`
- `training_extension/train_ppo_parameters.py`
- `training_extension/finetune_ppo_parameters.py`
- `training_extension/evaluate_ppo_parameters.py`
- `training_extension/render_ppo_parameters.py`
- `training_extension/view_ppo_parameters.py`
- `training_extension/tests/test_training_extension.py`
- `training_extension/requirements.txt`

### 说明与结果

- `training_extension/RL_TRAINING_RESULT_CN.md`
- `training_extension/RL_UPLOAD_MANIFEST_CN.md`
- 最终选中冻结目录中的：
  - `README_CN.md`
  - `selected_model.zip`
  - `selected_vecnormalize.pkl`
  - `evaluation_300_summary.json`
  - `evaluation_300_episodes.json`
  - `finetune_metadata.json`
  - `source_snapshot/*.py`
  - `source_snapshot/scene_throw_LEVEL03_ring.xml`
  - 代表视频、接触图和释放细节图
  - `SHA256SUMS.txt`

### 失败/中间里程碑

只上传足以说明实验过程的小型材料：

- TD3 原始 residual：崩塌结论、评估摘要、训练元数据；
- TD3 零初始化：崩塌结论、评估摘要、训练元数据；
- TD3 低通控制：崩塌结论、评估摘要、训练元数据；
- 逐帧 PPO residual：未崩塌但精度下降的摘要；
- SAC 参数 residual：尾部改善但平均误差未改善的摘要；
- PPO 1,024、旧选中 PPO 4,096，以及本次续训的代表检查点摘要。

失败实验默认不上传完整 replay buffer、TensorBoard event、VecNormalize
中间副本或数百 MB 模型，只保留可审计的 JSON/CSV、必要代码和少量图像。

## 明确排除

- `training_extension/runs/` 全量目录；
- `__pycache__/`、`*.pyc`；
- TensorBoard event；
- SAC replay buffer；
- 临时 stdout/stderr 日志；
- 重复模型和重复视频；
- 与本次参数 residual PPO 无关的历史原型；
- baseline 和 CEM PR 已经包含的重复资源。

## 上传前必须重新执行

1. 从干净的 ASCII 路径 Git 导出包运行测试；
2. 验证最终冻结目录的 `SHA256SUMS.txt`；
3. 在同一组未见种子上比较 CEM、旧 PPO、新 PPO；
4. 确认目标 `(2.2, 0.0, 1.2)`、固定 `0.10 m` 圆环与所有防作弊条件未变；
5. 用 `git diff --cached --name-only` 人工确认没有 `runs/` 和 RL 之外的文件；
6. 只创建 draft PR，base 必须是 `feature/simulation03-cem`。
