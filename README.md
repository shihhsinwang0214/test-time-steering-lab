# Test-time Steering Lab（DMA U7）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GITHUB_USER/test-time-steering-lab/blob/main/u7_minimal.ipynb)

你有一個訓練好的 1D flow matching 模型（參數不動）和一個公開的 reward $R(x)$。
目標：用任何 test-time 方法生成 **2,000 個點**，讓它們的分布盡量接近 $p_R(x)\propto p_{\text{model}}(x)\,e^{3R(x)}$。

| 檔案 | 用途 |
|---|---|
| `dma-u7-test-time-steering-lab.ipynb` | Starter notebook：載入模型、reward、看圖、一個空的 `my_method()`、輸出 `submission.csv` |
| `fm_model.pt` | 模型權重（598 KB，state_dict；網路定義在 notebook 裡） |

怎麼跑：點上面的 Colab 按鈕，或 `git clone` 後在本機開 notebook（需要 `torch numpy pandas matplotlib`）。
notebook 第一格會自動從這個 repo 下載 `fm_model.pt`。

交作業：把 `submission.csv`（兩欄 `ID,x`，恰 2,000 列）上傳到 Kaggle 競賽
**[Test-time Steering Lab](https://www.kaggle.com/competitions/test-time-steering-lab)**（邀請連結課堂公布）。
主榜是你的點到 $p_R$ 的 Wasserstein-1 distance，越小越好；$p_R$ 自己對自己約 0.05–0.17。
