# Scheme A Log-Euclidean Operator Probe Conclusion

更新时间：2026-03-29

当前 probe 保留 TELM2 多轮闭式更新后再拉伸，不做交替迭代。

## natops
- log-Euclidean 自洽算子是否优于当前向量版：`not_yet`
- vector delta vs raw: `+0.0273`
- log-Euclidean delta vs raw: `-0.0000`
- log-Euclidean delta vs vector: `-0.0273`
- 机制对比：`flip 0.0000->0.0000`，`classwise_cov_dist 1.1689->0.9797`，`cond_A 4.0712->3.3467`

## selfregulationscp1
- log-Euclidean 自洽算子是否优于当前向量版：`not_yet`
- vector delta vs raw: `-0.0138`
- log-Euclidean delta vs raw: `-0.0172`
- log-Euclidean delta vs vector: `-0.0033`
- 机制对比：`flip 0.0373->0.0336`，`classwise_cov_dist 2.6573->2.7492`，`cond_A 1.0213->1.0208`

结论标签：`scheme_a_probe_complete`
