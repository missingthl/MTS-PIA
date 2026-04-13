# PIA 实验复现指南

说明：

- 本文档主要描述旧阶段/全量复现实验的工作流
- 这些旧工作流使用的 `experiments/`、`logs/`、`promoted_results/` 工作区，现已归档到 [archive/legacy_workspace](../archive/legacy_workspace)
- 旧双流入口依赖的 `aggregation/`、`fusion/`、`pipeline/`、`block/` 代码层，以及旧 `pia_unified_demo.py` 入口，现已归档到 [archive/legacy_code](../archive/legacy_code)
- 若主动运行旧脚本，旧脚本仍可能在仓库根目录重新生成这些目录

本文档提供完整的实验复现步骤。

---

## 1. 环境配置

### 方式 A: Conda (推荐)

```bash
# 创建环境
conda env create -f environment.yml

# 推荐直接通过统一入口运行
scripts/run_in_pia.sh python -c "import torch; import mne; import pyriemann; print('OK')"
```

说明：

- 当前工程默认统一走 `pia` 环境
- 推荐优先使用 [scripts/run_in_pia.sh](../scripts/run_in_pia.sh)
- 主线自检入口是 [scripts/devtools/verify_current_stack_in_pia.sh](../scripts/devtools/verify_current_stack_in_pia.sh)

### 方式 B: pip

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
# 安装依赖
pip install -r requirements.txt
```

---

## 2. 数据准备

### SEED 数据集

1. 下载 SEED 数据集并放置到指定目录：

```
data/SEED/
├── SEED_EEG/
│   ├── ExtractedFeatures_1s/   # Official DE 特征 (1s 窗口)
│   ├── ExtractedFeatures_4s/   # Official DE 特征 (4s 窗口)
│   ├── Preprocessed_EEG/       # 预处理后的 EEG
│   ├── SEED_RAW_EEG/           # Raw EEG (.cnt 文件)
│   ├── SEED_RAW_EEG/time.txt   # Trial 时间戳
│   └── SEED_stimulation.xlsx   # 标签文件
└── channel_62_pos.locs         # 62 通道位置文件
```

2. 验证数据完整性：

```bash
# 检查 Raw EEG 文件数量 (应为 45 个 .cnt)
ls data/SEED/SEED_EEG/SEED_RAW_EEG/*.cnt | wc -l

# 检查 DE 特征文件
ls data/SEED/SEED_EEG/ExtractedFeatures_1s/*.mat | wc -l
```

### SEED-V 数据集

```
data/SEED_V/
├── EEG_DE_features/            # DE 特征
├── channel_62_pos.locs         # 通道位置
└── ... (其他必要文件)
```

---

## 3. 生成数据清单

在运行实验前，需要生成 trial manifest：

```bash
# 生成 SEED Raw EEG manifest
scripts/run_in_pia.sh scripts/data_prep/build_seed_raw_manifest_full.py

# 输出:
# - logs/seed_raw_trial_manifest_full.json
# - logs/seed_raw_trial_manifest_full.csv
```

## 4. 运行实验

实验结果（checkpoints, logs）将保存在 `experiments/` 目录中。

---

### 4.1 DCNet 空间流 (SEED-V)

```bash
# 单折快速测试
scripts/run_in_pia.sh archive/legacy_code/pia_unified_demo.py \
    --dataset seedv \
    --stream spatial \
    --backend torch \
    --folds 1 \
    --epochs 10

# 完整 3 折
scripts/run_in_pia.sh archive/legacy_code/pia_unified_demo.py \
    --dataset seedv \
    --stream spatial \
    --backend torch \
    --folds 3 \
    --epochs 80 \
    --batch-size 2048
```

### 4.2 流形流 (SEED-V)

```bash
scripts/run_in_pia.sh archive/legacy_code/pia_unified_demo.py \
    --dataset seedv \
    --stream manifold \
    --folds 3
```

### 4.3 双流融合 (SEED-V)

```bash
scripts/run_in_pia.sh archive/legacy_code/pia_unified_demo.py \
    --dataset seedv \
    --stream dual \
    --fusion-alpha 0.6 \
    --folds 3
```

### 4.4 SEED-1 Official DE Baseline

```bash
# 使用 1s 窗口
scripts/run_in_pia.sh archive/legacy_code/pia_unified_demo.py \
    --dataset seed1 \
    --stream spatial \
    --backend torch \
    --spatial-align-baseline \
    --spatial-input topo \
    --seed-de-root data/SEED/SEED_EEG/ExtractedFeatures_1s \
    --seed-de-var de_LDS1 \
    --seed-freeze-align \
    --epochs 80 \
    --batch-size 2048 \
    --folds 3
```

---

## 5. 预期结果

### SEED-V 空间流 (DCNet)

| 指标 | 预期范围 |
|------|----------|
| Sample Accuracy | ~75-85% |
| Trial Accuracy | ~80-90% |

### SEED-V 双流融合

| 指标 | 预期范围 |
|------|----------|
| Fused Trial Accuracy | ~85-92% |

---

## 6. 实验结果位置

运行后结果保存在 `logs/` 目录：

```
logs/
├── seed_train_test_index_*.json   # 训练/测试划分
├── manifold_deep_**/              # 深度流形实验
├── official_baseline_**/          # 基线实验
└── ... (其他实验日志)
```

---

## 7. 常见问题

### Q: CUDA out of memory

减小 batch size：

```bash
scripts/run_in_pia.sh archive/legacy_code/pia_unified_demo.py ... --batch-size 1024
```

### Q: ModuleNotFoundError

确保在项目根目录运行，并设置 PYTHONPATH：

```bash
cd /path/to/PIA
export PYTHONPATH=$PWD
scripts/run_in_pia.sh archive/legacy_code/pia_unified_demo.py ...
```

如需直接手写 `conda` 命令，也应等价地使用 `conda run -n pia ...`。

### Q: 数据文件找不到

检查数据路径是否正确：

```bash
# 检查数据目录
ls -la data/SEED/SEED_EEG/

# 检查 time.txt
cat data/SEED/SEED_EEG/SEED_RAW_EEG/time.txt
```
