# MDS

## 概要

主成分分析はデータ間の相関をベースとするが，MDS (Multi-dimensional Scaling)はデータ間の（広義の意味での）距離を要素とする距離行列をベースとしている．
MDSは距離行列の成分が距離の基本公理を満たすか否かで二分される：

- Metric MDS
- Nonmetric MDS

## 

## 参考文献
https://www.jstage.jst.go.jp/article/jje1965/13/4/13_4_137/_pdf/-char/ja

---

# t-SNE (T-distribution stochastic Neighbor Embedding)

## データの類似度

$N$個の高次元データを$\bm{x}_{1}, \ldots, \bm{x}_{N}$として，
$$
p_{j \mid i} = 
\begin{cases}
0 & j = i \\
\cfrac{\exp \left( - \lvert\lvert \bm{x}_i - \bm{x}_j \rvert\rvert^2 / 2 \sigma_i^2 \right)}{\sum_{k \ne i} \exp \left( - \lvert\lvert \bm{x}_i - \bm{x}_k \rvert\rvert^2 / 2 \sigma_i^2 \right)} & j \ne i
\end{cases}
$$
を定義する．
これは，データ$\bm{x}_i$を基準としてデータ$\bm{x}_j$を選択するとき，$\bm{x}_i$を中心とするガウス分布に比例して選択されると解釈できる．$\sigma_i$はperplexity $Perp(P_i)$を（我々のほうで）指定し，
$$
Perp \left( P_i \right) = 2^{H\left( P_i \right)}
$$
を満たすよう決定される．ここで，$H(P_i)$はShanonnエントロピーである：
$$
H(P_i) = - \sum_{j} p_{j \mid i} \log_2 p_{j \mid i}
$$
そして，$p_{i\mid j}$を対称化：
$$
p_{ij} = \cfrac{p_{j \mid i} + p_{i \mid j}}{2N}
$$
し，これを各データ対の類似度とする．

## 低次元空間への埋め込み

高次元データ$\left( \bm{x}_{1}, \ldots, \bm{x}_{N} \right)$を低次元空間内のデータ$\left( \bm{y}_{1}, \ldots, \bm{y}_{N} \right)$へ最適に対応させる方法を考えたい．
1. まず，データ間のコーシー分布と仮定する：
    $$
    q_{ij} = \cfrac{\left( 1 + \lvert\lvert \bm{y}_i - \bm{y}_j \rvert\rvert^2 \right)^{-1}}{\sum_k \sum_{l \ne k} \left( 1 + \lvert\lvert \bm{y}_k - \bm{y}_l \rvert\rvert^2 \right)^{-1}}.
    $$
2. 両者のKL情報量：
    $$
    \mathrm{KL} \left( P \mid\mid Q \right) = \sum_{i \ne j} p_{ij} \ln \cfrac{p_{ij}}{q_{ij}}
    $$
    を損失関数$C$とする．
3. 損失関数を最小化する$\left( \bm{y}_{1}, \ldots, \bm{y}_{N} \right)$：
    $$
    \mathrm{argmin}_{\bm{y}_1, \ldots, \bm{y}_N} C \left( \bm{y}_1, \ldots, \bm{y}_N \right)
    $$
    が求めたいものである．

## 勾配効果法

### パラメタ
- $Perp$: perplexity
- $\bm{\eta}$: learning rate
- $\bm{\alpha}(t)$: momentum

### 概要

損失関数の$\bm{y}_i$微分は
$$
\frac{\delta C}{\delta \bm{y}_i} = 4 \sum_{j} \frac{\left(p_{ij} - q_{ij}\right)}{1 + \lvert\lvert \bm{y}_i - \bm{y}_j \rvert\rvert^2} (\bm{y}_i - \bm{y}_j)
$$
であり，
$$
\bm{Y}^{t} = \bm{Y}^{t-1} + \bm{\eta} \frac{\delta C}{\delta \bm{Y}^{t-1}} + \bm{\alpha} \left(t\right) \left( \bm{Y}^{t-1} - \bm{Y}^{t-2} \right)
$$
のように$\bm{Y}^t = \left( \bm{y}_{1}^t, \ldots, \bm{y}_{N}^t \right)$を更新する．

## Barnes-Hut Approximation
詳しいことは[ここ](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) にある．あとでみて，加筆予定．

## 参考文献
[scikit-learn](https://scikit-learn.org/stable/modules/manifold.html#t-sne)のt-sneのところ

[“Visualizing High-Dimensional Data Using t-SNE”](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) van der Maaten, L.J.P.; Hinton, G. Journal of Machine Learning Research (2008)

[“Accelerating t-SNE using Tree-Based Algorithms”](https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) van der Maaten, L.J.P.; Journal of Machine Learning Research 15(Oct):3221-3245, 2014.