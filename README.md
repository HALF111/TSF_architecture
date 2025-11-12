# The Power of Architecture: Deep Dive into Transformer Architectures for Long-Term Time Series Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2507.13043-b31b1b.svg)](https://arxiv.org/abs/2507.13043)

This is the official repository for the paper:

> **The Power of Architecture: Deep Dive into Transformer Architectures for Long-Term Time Series Forecasting**  
> _Lefei Shen, Mouxiang Chen, Han Fu, Xiaoxue Ren, Xiaoyun Joy Wang, Jianling Sun, Zhuo Li, Chenghao Liu_  
> arXiv:2507.13043 [cs.LG], 2025  
> [📄 Read on arXiv](https://arxiv.org/abs/2507.13043)

In this work, we conduct a systematic study on Transformer architectures for **Long-Term Time Series Forecasting (LTSF)**. By proposing a novel taxonomy that disentangles architectural components from task-specific designs, we analyze key factors such as attention mechanisms, forecasting paradigms, aggregation strategies, and normalization layers. Our findings reveal that:
- Bi-directional attention with joint-attention performs best.
- Complete forecasting aggregation across look-back and forecasting windows improves accuracy.
- Direct-mapping paradigm outperforms autoregressive modeling.
- BatchNorm performs better for time series with more anomalies, while LayerNorm excels for more stationary time series.
- Above conclusions hold for both fixed and variable forecasting lengths.

Our unified framework with optimal architectural choices achieves consistent improvements over existing methods.

---

## 🔧 Training and Evaluation

To run experiments on all datasets, follow these steps:

```bash
cd TSF_architecture/
bash scripts/all_models/etth1.sh
bash scripts/all_models/etth2.sh
# ...... repeat for other datasets
```

Menwhile, all scipts for all 8 datasets are provided in the ./scripts/all_models/ directory.

Note:
You should use the "bash" command instead of the "sh" command in the Linux environment, otherwise errors may occur!

