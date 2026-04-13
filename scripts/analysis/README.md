# Analysis Scripts

这一层存放：

- 结果汇总
- 报表重建
- 诊断统计
- 分析型 probe

这些脚本默认不会被现役训练代码 import。  
它们的角色是“读结果、聚合结果、解释结果”，不是主实验入口。

当前常见入口：

- `gen_n3c_meta.py`
- `rebuild_lraes_curriculum_summary.py`
- `run_fusion_analysis.py`
- `summarize_curves.py`

如果你现在想推进当前分类主线，请优先回到：

- `scripts/hosts/`
- `scripts/route_b/`
