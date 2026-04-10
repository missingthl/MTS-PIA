# Phase15 主线冻结说明

更新时间：2026-03-21

## 0. 角色调整说明

自 2026-03-22 起，这份文档的角色已经调整为：

- 历史 `Freeze` 主线说明
- 旧 `z-space -> Step1B -> Gate -> LinearSVC` 路线的诊断入口
- 用于说明这条线为什么不再作为当前论文主体

当前主体骨架已经切换为：

- [docs/ROUTE_B_MAIN_BODY.md](/home/THL/project/MTS-PIA/docs/ROUTE_B_MAIN_BODY.md)

## 1. 当前主线冻结说明

当前分类工程只冻结一条默认主线：

- `raw trial -> 几何特征 z -> Step1B-style 多方向 PIA -> Gate1 / Gate2 -> LinearSVC`

当前默认主线模块：

- 几何特征提取：
  - SPD / log-center / 上三角向量化
- 默认主增强版本：
  - `Step1B` 多方向 PIA
- 默认主筛选版本：
  - `Gate1 / Gate2`

当前仍只是实验分支，不进入默认主线：

- `Step1C`
- `Step1D`
- `Gate3`
- `Fisher / C0 / safe-axis`
- `Controller Lite`
- `raw-bridge`

## 2. MiniROCKET 外部基线可信说明

当前 MiniROCKET 的正式可信口径只认：

- `out/raw_minirocket_official_protocol_20260318/official_protocol_status.csv`

它包含两类结果：

- 公开 fixed split：
  - `HAR / NATOPS / FingerMovements / SelfRegulationSCP1`
- SEED 家族原生协议：
  - `SEED / SEED_IV / SEED_V`

以下旧快照只保留为工程历史参考，不应再拿来逐点对齐正式表：

- `out/raw_minirocket_external_align_seed3_20260318/`

另外，2026-03-21 又新增了 6 个 aeon / UEA fixed-split 数据集的官方补实验：

- `out/raw_minirocket_official_fixedsplit_aeon6_20260321/official_protocol_status.csv`

它是当前 fixed-split 外部对齐面的补充，不替代原来的 7 数据集正式比较总表。

## 3. 当前正式主表

当前已经闭环的主线冻结总表：

- [main_performance_table.csv](/home/THL/project/MTS-PIA/out/phase15_mainline_freeze_20260319_formal/main_performance_table.csv)
- [mechanism_diagnosis_table.csv](/home/THL/project/MTS-PIA/out/phase15_mainline_freeze_20260319_formal/mechanism_diagnosis_table.csv)

注意：

- 这两张表当前正式并表的仍是 5 组：
  - `seed1`
  - `har`
  - `natops`
  - `fingermovements`
  - `selfregulationscp1`
- `seediv` 的 protocol-locked 主线冻结结果已经单独闭环，见：
  - `out/phase15_mainline_freeze_20260319_seedfamily/seediv/summary_per_seed.csv`
  - `out/phase15_mainline_freeze_20260319_seedfamily/seediv/mechanism_per_seed.csv`
- `seedv` 的 protocol-locked 主线冻结结果现在也已单独闭环，见：
  - `out/phase15_mainline_freeze_20260319_seedfamily/seedv/summary_per_seed.csv`
  - `out/phase15_mainline_freeze_20260319_seedfamily/seedv/mechanism_per_seed.csv`
- 但 `seediv / seedv` 还没有并入当前 formal 总表重构。

## 3.1 并行升级线索引

`raw-bridge` 当前不进入主线冻结表，但已经有独立的 simple-set probe 结果：

- [simple_set_probe_summary.csv](/home/THL/project/MTS-PIA/out/raw_bridge_simple_set_probe_20260320/simple_set_probe_summary.csv)
- [raw_bridge_structure_fidelity_summary.csv](/home/THL/project/MTS-PIA/out/raw_bridge_simple_set_probe_20260320/raw_bridge_structure_fidelity_summary.csv)
- [raw_bridge_distribution_stability_summary.csv](/home/THL/project/MTS-PIA/out/raw_bridge_simple_set_probe_20260320/raw_bridge_distribution_stability_summary.csv)
- [raw_bridge_probe_conclusion.md](/home/THL/project/MTS-PIA/out/raw_bridge_simple_set_probe_20260320/raw_bridge_probe_conclusion.md)
- [raw_bridge_target_vs_mapping_analysis.md](/home/THL/project/MTS-PIA/out/raw_bridge_simple_set_probe_20260320/raw_bridge_target_vs_mapping_analysis.md)

另外，2026-03-20 已新增一组更靠近 raw 端 target 质量验证的 very small pilot：

- [bridge_curriculum_pilot_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_pilot_20260320/bridge_curriculum_pilot_summary.csv)
- [bridge_curriculum_target_health_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_pilot_20260320/bridge_curriculum_target_health_summary.csv)
- [bridge_curriculum_fidelity_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_pilot_20260320/bridge_curriculum_fidelity_summary.csv)
- [bridge_curriculum_pilot_conclusion.md](/home/THL/project/MTS-PIA/out/bridge_curriculum_pilot_20260320/bridge_curriculum_pilot_conclusion.md)

这套结果只用于判断 raw-bridge 是否值得继续保留为升级线，不参与当前主线 freeze 的正式对外比较。

2026-03-20 随后又补了第二个 bridge-coupled pilot（`NATOPS`）：

- [bridge_curriculum_natops_pilot_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_natops_pilot_20260320/bridge_curriculum_natops_pilot_summary.csv)
- [bridge_curriculum_natops_target_health_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_natops_pilot_20260320/bridge_curriculum_natops_target_health_summary.csv)
- [bridge_curriculum_natops_fidelity_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_natops_pilot_20260320/bridge_curriculum_natops_fidelity_summary.csv)
- [bridge_curriculum_natops_pilot_conclusion.md](/home/THL/project/MTS-PIA/out/bridge_curriculum_natops_pilot_20260320/bridge_curriculum_natops_pilot_conclusion.md)
- [bridge_curriculum_har_vs_natops_compare.md](/home/THL/project/MTS-PIA/out/bridge_curriculum_natops_pilot_20260320/bridge_curriculum_har_vs_natops_compare.md)

这说明 `curriculum target -> bridge -> raw MiniROCKET` 不再只是 HAR 单点成立，而是已经在第二个 harder simple-set 数据集上得到正方向复核；但它依然只属于独立升级线，不并入当前主线 freeze 正式总表。

2026-03-21 又补了第三个 simple-set pilot（`SelfRegulationSCP1`）：

- [bridge_curriculum_scp1_pilot_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_scp1_pilot_20260321/bridge_curriculum_scp1_pilot_summary.csv)
- [bridge_curriculum_scp1_target_health_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_scp1_pilot_20260321/bridge_curriculum_scp1_target_health_summary.csv)
- [bridge_curriculum_scp1_fidelity_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_scp1_pilot_20260321/bridge_curriculum_scp1_fidelity_summary.csv)
- [bridge_curriculum_scp1_pilot_conclusion.md](/home/THL/project/MTS-PIA/out/bridge_curriculum_scp1_pilot_20260321/bridge_curriculum_scp1_pilot_conclusion.md)
- [bridge_curriculum_har_natops_scp1_compare.md](/home/THL/project/MTS-PIA/out/bridge_curriculum_scp1_pilot_20260321/bridge_curriculum_har_natops_scp1_compare.md)

这轮结果说明：Route B 在 `SCP1` 上没有复制 `HAR / NATOPS` 的完整正闭环，但 multiround 依然稳定优于 single-round，且 bridge fidelity 继续更干净；因此 `SCP1` 更像 Route B 的 simple-set 边界集，而不是其否定例。

2026-03-21 随后又补了第四个 simple-set pilot（`FingerMovements`）：

- [bridge_curriculum_fm_pilot_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_fm_pilot_20260321/bridge_curriculum_fm_pilot_summary.csv)
- [bridge_curriculum_fm_target_health_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_fm_pilot_20260321/bridge_curriculum_fm_target_health_summary.csv)
- [bridge_curriculum_fm_fidelity_summary.csv](/home/THL/project/MTS-PIA/out/bridge_curriculum_fm_pilot_20260321/bridge_curriculum_fm_fidelity_summary.csv)
- [bridge_curriculum_fm_pilot_conclusion.md](/home/THL/project/MTS-PIA/out/bridge_curriculum_fm_pilot_20260321/bridge_curriculum_fm_pilot_conclusion.md)
- [bridge_curriculum_all_simple_sets_compare.md](/home/THL/project/MTS-PIA/out/bridge_curriculum_fm_pilot_20260321/bridge_curriculum_all_simple_sets_compare.md)

这轮结果说明：`FingerMovements` 当前更像 Route B 的 high-volatility edge case。multiround 并没有在 aggregate 上优于 single-round，但 bridge fidelity 依旧更干净，所以它更像 raw 端高波动下的 target usefulness 边界，而不是 bridge 自身失效。

## 3.1.1 2026-03-21 新增 aeon fixed-split 6 数据集补实验

当前又新增了 6 个 aeon / UEA fixed-split 数据集的正式补实验：

- `BasicMotions`
- `HandMovementDirection`
- `UWaveGestureLibrary`
- `Epilepsy`
- `AtrialFibrillation`
- `PenDigits`

它们已经统一接入：

- `datasets/aeon_fixedsplit_trials.py`
- `datasets/trial_dataset_factory.py`

并补齐了和旧 fixed split 四组一致的两条实验线：

1. strict official MiniROCKET fixed-split
   - [official_protocol_status.csv](/home/THL/project/MTS-PIA/out/raw_minirocket_official_fixedsplit_aeon6_20260321/official_protocol_status.csv)
2. Phase15 mainline freeze fixed-split
   - [summary_per_seed.csv](/home/THL/project/MTS-PIA/out/phase15_mainline_freeze_aeon6_20260321/summary_per_seed.csv)
   - [mechanism_per_seed.csv](/home/THL/project/MTS-PIA/out/phase15_mainline_freeze_aeon6_20260321/mechanism_per_seed.csv)

另外提供一张并排总表，便于快速看 `MiniROCKET / baseline / Step1B / Step1B+Gate`：

- [aeon6_fixedsplit_summary.csv](/home/THL/project/MTS-PIA/out/aeon6_fixedsplit_experiments_20260321/aeon6_fixedsplit_summary.csv)

当前补实验的直观结论是：

- `BasicMotions`
  - baseline 与 MiniROCKET 都达到 `1.0000`
  - Step1B / Gate 反而略降
- `HandMovementDirection`
  - Step1B / Gate 相对 baseline 有小幅改善
  - 但仍低于 MiniROCKET
- `UWaveGestureLibrary`
  - Step1B 小幅优于 baseline
  - 但与 MiniROCKET 仍有明显差距
- `Epilepsy`
  - Gate 相对 baseline 有小幅改善
  - 但 MiniROCKET 仍明显更强
- `AtrialFibrillation`
  - 当前单 seed 下 z-space Step1B+Gate 高于 MiniROCKET
  - 但这是极小样本集，现阶段只宜视为高波动观察点
- `PenDigits`
  - MiniROCKET 极强
  - z-space freeze 明显不占优
  - 更像 symbolic short-sequence 边界集

`PenDigits` 还有一个需要明确留痕的实现细节：

- 它的原始序列长度只有 `8`
- `aeon` MiniROCKET 要求 `n_timepoints >= 9`
- 因此 strict official runner 对它做了最小零填充到 `9`
- 该兼容信息已经写入对应 `note`

这些新增数据集当前仍不并入现有的 Phase15 formal 主表；它们更像“新补齐的 fixed-split 对齐面”，用于扩展外部比较与后续数据集筛选。

## 3.1.2 2026-03-21 aeon6 的最新 Route B 补实验

针对这 6 个新增 fixed-split 数据集，我们也补了当前最新的 `multiround curriculum target -> bridge -> raw MiniROCKET` 路线。

统一输出目录：

- `out/bridge_curriculum_aeon6_20260321/`

按当前 aggregate 结果看：

- `BasicMotions / Epilepsy / AtrialFibrillation`
  - 更像 flat set
  - 三条线大体持平
- `HandMovementDirection`
  - `multiround bridge > single-round bridge`
  - 当前属于中间带正信号
- `UWaveGestureLibrary / PenDigits`
  - 当前更像 negative boundary set
  - `multiround bridge` 没有复制 `HAR / NATOPS` 的正闭环

这说明：

- Route B 已经不只是 `HAR / NATOPS` 两个单点
- 但它也不是 universal bridge upgrade
- 当前更像：在一部分 structured / harder 数据集上成立，在另一部分 symbol-like / high-constraint 数据集上触边

## 3.2 增强样本几何参考价值诊断

另有一组 diagnostic-only 补实验，用于判断当前 `Step1B` 增强样本到底更像：

- 训练污染源
- 还是几何参考资源

这组实验当前只在 simple-set 上做最小版本：

- 只重估全局 `log-center` 参考点
- 增强样本不参与任何最终监督训练
- 不进入当前 Phase15 mainline freeze 正式主表

输出目录：

- `out/geometry_recenter_probe_20260320_v2/geometry_recenter_probe_summary.csv`
- `out/geometry_recenter_probe_20260320_v2/geometry_recenter_reference_shift_summary.csv`
- `out/geometry_recenter_probe_20260320_v2/geometry_recenter_probe_conclusion.md`

当前结论是：

- `HAR / SelfRegulationSCP1 / FingerMovements` 上，`aug_recenter_orig_train` 都与 `baseline_orig_plane` 持平
- `step1b_direct_train` 在这轮 simple-set probe 中反而都略低于 baseline
- 因此当前还没有证据把 Step1B 增强样本写成“几何参考型增强成立”
- 更准确的表述是：在这个最小版本里，增强样本对参考结构重估更像 `geometry_neutral`

## 3.3 PIA Feedback Upgrade Probe

另有一组独立的 `PIA feedback upgrade` small-set probe，用于判断当前 PIA 是否能从“稳定但中性”的增强器，推进为更可训练的增强器。

这组实验：

- 不进入 Phase15 mainline freeze 正式主表
- 只在 fixed-split 小数据集上运行
- 当前比较：
  - `baseline`
  - `Step1B`
  - `Fisher/C0 safe-axis weighted`
  - `feedback_weighting`

输出目录：

- `out/phase15_feedback_upgrade_20260320/feedback_upgrade_performance_summary.csv`
- `out/phase15_feedback_upgrade_20260320/feedback_direction_health_summary.csv`
- `out/phase15_feedback_upgrade_20260320/feedback_upgrade_conclusion.md`

当前结论是：

- `HAR / NATOPS` 上出现了相对 `Step1B` 的正收益
- `SelfRegulationSCP1 / FingerMovements` 上仍未转正
- 因此更准确的定位是：
  - `局部升级线已成立`
  - `但还不是稳定跨数据集升级线`

## 3.4 2026-03-21 Fisher/C0 + Curriculum 上游 target 净化

在 Route B 已经形成“正例 / 边界例 / 负例”分层之后，当前又新增了一条上游 target 净化实验线：

- [fisher_curriculum_performance_summary.csv](/home/THL/project/MTS-PIA/out/phase15_fisher_curriculum_core_20260321/fisher_curriculum_performance_summary.csv)
- [fisher_curriculum_direction_health_summary.csv](/home/THL/project/MTS-PIA/out/phase15_fisher_curriculum_core_20260321/fisher_curriculum_direction_health_summary.csv)
- [fisher_curriculum_prior_cleaning_summary.csv](/home/THL/project/MTS-PIA/out/phase15_fisher_curriculum_core_20260321/fisher_curriculum_prior_cleaning_summary.csv)
- [fisher_curriculum_conclusion.md](/home/THL/project/MTS-PIA/out/phase15_fisher_curriculum_core_20260321/fisher_curriculum_conclusion.md)

第一批主判断集固定为：

- `NATOPS`
- `SelfRegulationSCP1`
- `FingerMovements`

当前结果是：

- `NATOPS`
  - `fisher_c0_plus_curriculum` 相对纯 `curriculum` 仍有小幅正增益
- `SelfRegulationSCP1`
  - `fisher_c0_plus_curriculum` 相对纯 `curriculum` 出现更明确正增益
- `FingerMovements`
  - 当前基本持平，更像 neutral

结合 prior cleaning 摘要，更合理的解释是：

- Route B 当前的下一步瓶颈，至少有一部分仍在上游方向库质量
- `Fisher/C0` 作为“先验方向净化 + curriculum”的前端，在 `NATOPS / SCP1` 上已经出现有效信号
- 但它还远没有达到“可替代默认主线方向库”的程度

## 4. 当前主结论

截至 2026-03-21，当前最强、也最安全的结论只有这些：

1. `Step1B` 仍应冻结为当前默认主增强版本。
   这不是因为它已经在 7 个核心数据集上全部完成 protocol-locked 验证，
   而是因为当前没有任何后续版本在已闭环证据上稳定替代它。

2. `Gate1 / Gate2` 仍是当前默认 Gate 组合。
   在 fixed split 四组里，它有局部正信号，但还不能写成“普遍稳定增益”；
   `seed1` 这轮 protocol-locked 结果里，Gate 也没有带来额外收益。

3. 当前 z-space 主线在当前已闭环的 5 组数据上，并没有普遍超过官方协议 MiniROCKET。
   从当前表看，`seed1 / HAR / NATOPS / FingerMovements / SelfRegulationSCP1` 上，MiniROCKET 仍然更强。

4. 当前工作最有价值的地方，不是“已经全局胜过 raw baseline”，
   而是已经形成了一条机制可解释的几何增强主线，并能通过：
   - `flip_rate`
   - `margin_drop_median`
   - `knn_intrusion_rate`
   - `dir_profile`
   去诊断增强收益和失败来源。

5. `raw-bridge` 与“方案A 动态流形分类”当前都不应进入主论文主体。
   更合适的位置是：
   - 后续升级方向
   - future work / extension

## 5. 当前边界

当前必须明确承认的边界：

- `seediv` 已单独闭环，但还没有并入当前 formal 7 数据集主表
- `seedv` 现在也已单独闭环，但同样还没有并入当前 formal 7 数据集主表
- 因此当前不能把 7 数据集主线矩阵写成“已全部完成”
- 也不能把 `raw-bridge` 在 `seed1 / seediv` 上写成有效或无效

## 6. 下一阶段建议

下一阶段只建议做 3 件事：

1. 先把 `seediv + seedv` 正式并入新的 formal 主表重构，
   结束“单独闭环但未并表”的状态。
2. 在主表重构完成之前，继续把 `Route B` 和 `Fisher/C0 + curriculum` 保持为独立升级线，
   不推回当前主论文主体。
3. 主论文叙事先固定为：
   - `Step1B + Gate` 是当前默认 z-space 主线
   - MiniROCKET 官方协议结果是外部对齐基线
   - 机制诊断是当前工作的主要解释贡献
