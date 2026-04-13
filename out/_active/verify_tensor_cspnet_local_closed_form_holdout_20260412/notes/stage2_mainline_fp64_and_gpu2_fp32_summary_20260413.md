# Stage-2 Mainline FP64 And GPU2 FP32 Summary

日期：2026-04-13

## 1. 本轮结果在二阶段中的位置

这批结果对应二阶段当前最关键的两条验证：

- `fp64 mainline`
  - 当前正式主线
  - 单进程、全量 `BCIC-IV-2a holdout`
  - 用于判断 `E0 / E1 / E2` 当前谁更稳
- `GPU2 fp32`
  - `SPD fp32` 数值策略探索
  - 先通过 `subject 1 / subject 5` 等价性验证
  - 再做 `E0 / E1 / E2` 的 GPU2 手动并发验证

## 2. 先验结论

### 2.1 `SPD fp32` 已经从“想法”变成“可运行候选”

单 subject 等价性验证：

- `subject 1`
  - `fp64`: `acc=0.8403`, `loss=0.425822`, `wallclock=488.0s`
  - `fp32`: `acc=0.8403`, `loss=0.425701`, `wallclock=316.3s`
- `subject 5`
  - `fp64`: `acc=0.6007`, `loss=1.097744`, `wallclock=486.3s`
  - `fp32`: `acc=0.6007`, `loss=1.097461`, `wallclock=324.8s`

当前读法：

- `SPD fp32` 在至少两个关键 subject 上已显示基本等价
- 同时 wallclock 有大约 `1.5x` 的收益

### 2.2 `E2` 当前还没有建立优势

不管看 `fp64 mainline`，还是看 `GPU2 fp32`，当前都没有证据表明 `E2` 已经稳定优于 `E0 / E1`。

## 3. 当前可直接引用的结果

### 3.1 `fp64 mainline`

结果目录：

- [E0 fp64 summary](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e0/seed1_stage2_main_e0_fp64_b29_seed1/summary.json)
- [E1 fp64 summary](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e1/seed1_stage2_main_e1_fp64_b29_seed1/summary.json)
- [E2 fp64 summary](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e2/seed1_stage2_main_e2_fp64_b29_seed1/summary.json)

均值结果：

- `E0 fp64 = 0.7103`
- `E1 fp64 = 0.7083`
- `E2 fp64 = 0.7018`

当前读法：

- `E0` 仍是当前最稳的宿主
- `E1` 没有形成稳定提升
- `E2` 暂时落后

### 3.2 `GPU2 fp32`

结果目录：

- `E0`
  - [sub1](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e0/seed1_stage2_gpu2_full_e0_fp32_sub1_seed1_conc9/summary.json)
  - [sub5](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e0/seed1_stage2_gpu2_full_e0_fp32_sub5_seed1_conc9/summary.json)
- `E1`
  - [sub1](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e1/seed1_stage2_gpu2_full_e1_fp32_sub1_seed1_conc9/summary.json)
  - [sub5](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e1/seed1_stage2_gpu2_full_e1_fp32_sub5_seed1_conc9/summary.json)
- `E2`
  - [sub1](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e2/seed1_stage2_gpu2_full_e2_fp32_sub1_seed1_conc9/summary.json)
  - [sub5](/home/THL/project/MTS-PIA/out/_active/verify_tensor_cspnet_local_closed_form_holdout_20260412/e2/seed1_stage2_gpu2_full_e2_fp32_sub5_seed1_conc9/summary.json)

聚合均值：

- `E0 fp32 = 0.7118`
- `E1 fp32 = 0.7114`
- `E2 fp32 = 0.6983`

当前读法：

- `fp32` 没把整体精度打坏
- `E0` 与 `E1` 目前非常接近
- `E2` 仍未建立优势

## 4. 如何正确阅读 `GPU2 fp32` 这轮

这轮是：

- 单卡
- `conc9`
- 按 `subject` fan-out 并发

因此：

- 可以直接拿它读 `test_macro_acc`
- 可以直接看 `subject` 级排序
- 但**不应直接把 `train_seconds` 和 `fp64 mainline` 的单进程结果做 wallclock 对比**

因为这轮的 `train_seconds` 被同卡 `9` 并发显著放大，更多反映的是调度方式，不是纯模型快慢。

## 5. 当前阶段结论

当前可以先收成四句话：

1. `SPD fp32` 已经被验证为一个真实可用的候选数值策略。
2. `fp64 mainline` 和 `GPU2 fp32` 都支持同一判断：`E2` 当前没有赢。
3. `E1` 可以作为稳定对照保留，但目前没有显示出稳定优于 `E0` 的证据。
4. 二阶段当前最稳的口径仍然是：
   - 宿主 `Tensor-CSPNet` 可靠
   - `SPD fp32` 值得继续保留
   - 但方法主臂 `E2` 需要继续改
