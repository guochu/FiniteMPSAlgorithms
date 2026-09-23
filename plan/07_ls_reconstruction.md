# 07 algorithms/reconstruct.jl：已知振幅的 MPS 重构——TCI 的二次优化替代

**动机**：TCI（tensor cross interpolation）从已知的部分振幅出发，通过连续的矩阵插值
（pivot 更新）构造 MPS 逼近高维张量。其本质是：**已知 MPS 的一部分振幅，重构 MPS**。
本文档给出另一条路线：把重构写成**二次优化**（最小二乘）问题，用与现有
`DMRG1`/`DMRG2`/`linsolve` 同族的 ALS 扫掠求解。与 TCI 的逐 pivot 精确插值不同，
LS 路线把全部样本**联合**逼近：

| | TCI（cross） | 本文档（LS） |
|---|---|---|
| 样本角色 | pivot：链上**精确**插值 | 数据点：全体**联合**最小二乘 |
| 求解方式 | 逐 rank-1 更新（maxvol） | 正规方程 + 键内 ALS 扫掠 |
| 振幅含噪声 | 精确复现噪声（插值性质决定） | 噪声被平均（主要卖点） |
| 键维控制 | 构造后另做截断 | 内建（two-site 截断重分割） |
| 失效模式 | pivot 退化、rank-1 相关 | 欠采样、ALS 局部极小 |
| 每步代价 | O(L·D²·d) 插值 | O(N·d²·D⁴) 构造 + O(d³D⁶) 局域解 |

## 7.1 问题的二次优化形式（loss 构造）

设目标张量 `A(𝐱)`, `𝐱 = (x₁,…,x_L) ∈ ⊙ⱼ{1..dⱼ}`。已知 **样本集**
`S = {(𝐱⁽ᵏ⁾, aₖ)}ₖ₌₁..N`，`aₖ = A(𝐱⁽ᵏ⁾)`（允许 `aₖ` 带噪声/带权重 `wₖ ≥ 0`）。
求键维 `D` 的 MPS `ψ`（张量 `ψ[𝐱] = ⟨𝐱|ψ⟩`）最小化

```
ℒ(ψ) = Σₖ wₖ |ψ[𝐱⁽ᵏ⁾] − aₖ|²
```

即 `‖√W·(P_S ψ − a)‖²`，其中 `P_S = Σₖ |𝐱⁽ᵏ⁾⟩⟨𝐱⁽ᵏ⁾|` 是样本投影算符、
`a = Σₖ aₖ|𝐱⁽ᵏ⁾⟩`。关键结构观察：

- `P_S = Σₖ ⊗ⱼ |xⱼ⁽ᵏ⁾⟩⟨xⱼ⁽ᵏ⁾|` 是 **N 个直积投影算符之和**；
  `a` 是 **N 个直积态之和**——本问题 = `linsolve(P_S, a)` 的特例，
  但 `P_S` 的 MPO 键维 = N（不实用），专用实现按 **样本维度显式求和**（见 7.4）。
- `ℒ` 在全张量空间是凸二次；MPS 流形上的 ALS 扫掠继承单调性
  （精确局域解 + NoTruncation ⇒ loss 单调，与 DMRG2 测试纪律一致）。
- **无需 KKT 尺度恢复**：正规方程是非齐次的，数据项本身固定整体标度
  （对照 `dmrg2.jl` 驱动器的 `lmul!`——此处不需要）。

**局域化（site s）**：固定其余位点，记左/右语境

```
ℓₖ = ℓ_s(𝐱⁽ᵏ⁾) = ψ₁[x₁]⋯ψ_{s−1}[x_{s−1}] ∈ ℂ^{D_{s−1}}
rₖ = r_s(𝐱⁽ᵏ⁾) = ψ_{s+1}[x_{s+1}]⋯ψ_L[x_L] ∈ ℂ^{D_s}
```

则 `ψ[𝐱⁽ᵏ⁾] = ℓₖ · ψ_s · (e(x_s⁽ᵏ⁾) ⊗ rₖ)`，特征向量 `fₖ = ℓₖ ⊗ eₖ ⊗ rₖ`
（`eₖ = e(x_s⁽ᵏ⁾)` 为物理基）。正规方程（`z = vec(ψ_s)`）：

```
M_s z = b_s,   M_s = Σₖ wₖ fₖ fₖ†   （d·D_{s−1}·D_s 维，半正定 Hermitian）
b_s = Σₖ wₖ āₖ fₖ
```

`M_s` 按**物理桶**结构存储：`M_s = Σ_p B_s(p)·E_p`（`E_p` 物理单热阵），
`B_s(p) = Σₖ: x_s⁽ᵏ⁾=p wₖ (ℓₖ⊗rₖ)(ℓₖ⊗rₖ)†`（4 腿 `D_{s−1}×D_{s−1}×D_s×D_s`，
每物理值一个桶）。求和**不可**在 k 上分解（ℓ 与 r 的样本关联必须保留），
桶内必须逐样本外积累加。

**两点版（键维适应）**：对 `(s, s+1)` 联合构造 `fₖ = ℓₖ ⊗ eₖ ⊗ eₖ₊₁ ⊗ rₖ`
（维 `d·D_{s−1}` × `d·D_s`），正规方程解 `z₂`（维 `d²·D_{s−1}·D_s`）后
`tsvd` 按 `alg.trunc` 重分割（复用 `dmrg2.jl` 的 `_als2_update!` 分割逻辑与
`_gauge_left/_gauge_right` 移规范）。

**正则化**：欠采样（N 小于有效自由度）时 `M_s` 奇异/病态；
加 Tikhonov 项 `λ‖ψ‖²`（λ 默认 0）等价于 `M_s += λI`，
或用 `pinv`-型截断 SVD 求解。λ>0 的解偏向小范数（低秩偏置，对 MPS 有利）。

## 7.2 算法配置（追加到 `src/algorithms/algdefs.jl`）

```julia
"""
	LSRecon(; maxiter=Defaults.maxiter, tol=Defaults.tol,
	        trunc=truncdim(Defaults.D), λ=0.0, nadd=8, nbuffer=1024, verbosity=0)

Sample-amplitude MPS reconstruction by quadratic (least-squares) optimization —
the variational alternative to tensor cross interpolation. `trunc` controls the
two-site bond re-split (`NoTruncation` = single-site-scale solves only);
`λ ≥ 0` is the Tikhonov regularization of the local normal equations;
`nadd`/`nbuffer` drive the adaptive sample-enrichment loop (7.5).
"""
@kwdef struct LSRecon{TR<:TruncationScheme} <: IterativeMPSAlgorithm
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	trunc::TR = truncdim(D=Defaults.D)
	λ::Float64 = 0.0
	nadd::Int = 8            # 每 轮自适应新增样本数
	nbuffer::Int = 1024      # 候选缓冲池大小（随机粗采样, 残差在其中挑 nadd）
	verbosity::Int = 0
end
```

## 7.3 接口（`src/algorithms/reconstruct.jl`，唯一导出接口 `reconstruct`）

```julia
# 黑盒 oracle：自适应采样（7.5）+ 内环 ALS，主入口
reconstruct(Afun::Function, ds::NTuple{L,Int}; alg::LSRecon = LSRecon())
	-> (ψ::CanonicalMPS, info::NamedTuple{(:loss, :maxres, :nsamples, :rounds)})

# 稠密张量 oracle（小系统验证用；同上自适应循环, Afun = A[𝐱]）
reconstruct(A::DenseArray, ds; alg)

# 固定样本集（无自适应）：samples::Vector{Pair{NTuple{L,Int},T}} 或 (𝐱, a) 元组
reconstruct(samples, ds; alg)

# in-place 精化（样本固定；调用方提供初值）
reconstruct!(ψ::CanonicalMPS, samples, alg::LSRecon) -> (ψ, loss)
```

命名：`reconstruct`（从振幅数据重构链）；弃用 `fit`（与 StatsAPI 混淆）、
`tci`（本路线明确**不是** cross）、`recovery`（冗长）。

## 7.4 求解器：样本环境 ALS 扫掠（伪代码）

```
输入: 样本集 S = {(𝐱⁽ᵏ⁾, aₖ, wₖ)}、初值 ψ⁰（randommps, D = trunc 的键帽）
约束: ψ 的位点张量 A[aL, p, aR]（与包约定一致）；扫掠规范移动复用 mult.jl 的
      _gauge_left/_gauge_right

内环一轮（ ALS 到收敛）:
  # ---- 右语境预计算（右→左） ----
  R[L]   = ones(N, 1)                    # r_L(𝐱ₖ) = 1
  for s = L-1 … 1:
      R[s][k, :] = ψ_{s+1}[x_{s+1}⁽ᵏ⁾] · R[s+1][k, :]     # 每样本一行, O(N·d·D)

  # ---- 左→右扫 ----
  Lmat[1] = ones(N, 1)                   # ℓ_1(𝐱ₖ) = 1
  for s = 1 … L-1:
      # 两点正规方程: fₖ = ℓₖ ⊗ e(x_s⁽ᵏ⁾) ⊗ e(x_{s+1}⁽ᵏ⁾) ⊗ rₖ
      M ← Σ_p B(p)·E_p   （桶: 按 (x_s, x_{s+1}) 取值对分组, 桶内逐样本外积累加）
      b ← Σₖ wₖ āₖ (ℓₖ ⊗ eₖ ⊗ eₖ₊₁ ⊗ rₖ)
      z₂ ← (M + λ·I)⁻¹ b                    # 稠密解, 维 (d·D_{s-1})·(d·D_s)
      _als2_update!(ψ, s, z₂, trunc)        # tsvd 重分割（sv 归一化, 与 dmrg2.jl 同）
      Lmat[s+1][k, :] = Lmat[s][k, :] · ψ_s[x_s⁽ᵏ⁾]        # 左语境传递
  末位直接解 1-site 正规方程（维 d·D²）

  # ---- 右→左扫（对称, 用已缓存的 Lmat, 现算右语境传递） ----
  … …

  loss ℒ = Σₖ wₖ|ψ[𝐱⁽ᵏ⁾] − aₖ|²            # 精确局域解 ⇒ 单调
  |ℒ_n − ℒ_{n−1}|/ℒ_{n−1} < tol → 收敛
```

**每扫两遍语境的更新规则**（与 DMRG1 的 hstorage 同型：扫过即更新，
**右语境栈在左扫期间只读**——dmrg2.jl 的教训：左扫每 pair 只允许一次左语境更新，
不得覆盖右语境）：

```
左语境: ℓ_{s+1}(𝐱ₖ) ← ℓ_s(𝐱ₖ)·ψ_s[x_s⁽ᵏ⁾]     (N·D² 每位点)
右语境: r_s(𝐱ₖ)   ← ψ_{s+1}[x_{s+1}⁽ᵏ⁾]·r_{s+1}(𝐱ₖ)
```

**复杂度**：每 pair `O(N·d²·D⁴)` 桶构造 + `O(d³·D⁶)` 稠密解 + `O(N·d·D²)` 语境传递；
样本存储 `O(L·N·D)`。大 `N`/大 `D` 时局域解可换 CG（`M` 只需 mat-vec：
`z ↦ Σₖ wₖ fₖ(fₖ†z)`，`O(N·d·D²)` 每次迭代）——首版用稠密解，预留分派。

## 7.5 自适应采样（替代 cross 的 pivot 选取）

```
S ← 初始样本: nbuffer 个随机点 + 若干 argmax|A| 点
repeat（至多 maxiter 外环）:
    (ψ, ℒ) ← 内环 ALS(S)                        # 7.4, 收敛即止
    在 nbuffer 个新随机候选 𝐱′ 上算残差 ρ(𝐱′) = |ψ[𝐱′] − A[𝐱′]|
    maxρ < tol → 完成
    S ← S ∪ {残差最大的 nadd 个 𝐱′}             # 残差驱动的样本增补
    （可选增强: 对 ψ 做 maxvol 取行, 用"当前 ψ 的极值点"补充结构信息）
end
```

与 TCI 的对照：TCI 的 pivot 选取 = 当前近似的**最大体积/最大振幅**行列更新；
这里等价物是**最大残差**点（信息最丰富的未解释数据）。残差驱动保证样本集
单调增长、loss 单调下降；噪声情形下 tol 应设在噪声底（`tol ≳ σ·√(N)`-级），
否则外环永远增补。

## 7.6 需要实现的函数清单

| 函数 | 位置 | 说明 |
|---|---|---|
| `LSRecon` | algdefs.jl | 算法配置（7.2） |
| `reconstruct(Afun, ds; alg)` | reconstruct.jl | 主入口（自适应） |
| `reconstruct(A::Array, ds; alg)` | reconstruct.jl | 稠密 oracle 包装 |
| `reconstruct(samples, ds; alg)` | reconstruct.jl | 固定样本 |
| `reconstruct!(ψ, samples, alg)` | reconstruct.jl | in-place |
| `struct LSCache` | reconstruct.jl | ψ、样本矩阵 X（L×N）、值 a、权重 w、语境栈 Lmat/Rmat |
| `_init_contexts_right!(c)` | reconstruct.jl | 右→左预计算 Rmat（对照 `_init_hstorage_right!`） |
| `_left_transfer!(c, s)` / `_right_transfer!(c, s)` | reconstruct.jl | 语境传递（扫掠中） |
| `_ls_reduce_two_site(c, s)` | reconstruct.jl | 两点正规方程 (M, b)（桶实现，7.1） |
| `_ls_reduce_site(c, s)` | reconstruct.jl | 单点版（末位/1-site 模式） |
| `_ls_solve!(c, s, alg)` | reconstruct.jl | `(M + λI) \ b` + `_als2_update!` 重分割 |
| `leftsweep!/rightsweep!/sweep!(c, alg::LSRecon)` | reconstruct.jl | 统一扫掠接口（包约定） |
| `_ls_loss(c)` | reconstruct.jl | 当前 ℒ（收敛判据 + 单调性测试） |
| `_residual_candidates(Afun, ψ, nbuffer)` | reconstruct.jl | 候选池残差评估（自适应外环） |
| `_enrich_samples!(c, Afun, nadd)` | reconstruct.jl | 样本增补 + 语境栈扩容 |

## 7.7 与现有机制的关系 / 原型基线

- **原型基线（验证用）**：`P_S` 作为显式 MPO（Σₖ 直积投影，键维 = N）+
  `a` 的 MPS 构造（N 个 prodmps 之和）→ 现有 `linsolve(P_S, a, DMRG2(…))`。
  只在小 N 下做交叉验证，不作实现路线。
- **局域求解器**：`_ls_solve!` 的稠密解与 `linsolve.jl` 的 `_site_solve` 同型
  （正规方程显式组装）；两点重分割复用 `dmrg2.jl` 的 `_als2_update!`。
- **扫掠纪律**：与 `dmrg2.jl` 相同——leftsweep 每 pair 恰好一次左语境更新；
  NoTruncation 下 loss 必须单调（测试钉死）。
- **无 KKT 尺度恢复**：正规方程非齐次，解的标度由数据决定；
  驱动器不需要 `lmul!`/`setscaling!` 补偿（对照 dmrg2.jl 的教训）。
- **采样算符视角**：P_S/√W 永不显式构造；一切环境按样本维显式求和。

## 7.8 测试要点

- **精确插值**：小系统（L=4, d=2），A = 已知低键 MPS 稠密化；N = d^L 全采样
  → `todense(ψ)` 与 A 机器精度一致。
- **欠采样泛化**：N ≪ d^L 但 A 键维 ≤ D → 自适应循环收敛到全张量误差 ~1e-10。
- **loss 单调**：NoTruncation + 小系统，`monotone(khist)`（与 DMRG2 测试同款）。
- **噪声稳健（卖点）**：aₖ = A(𝐱ₖ) + σ·噪声；LS 重构误差 ~σ 级，
  而 TCI 式精确插值误差不随 σ 缩小（对比基线）。
- **正则化**：重复/相关样本下 λ=0 病态 vs λ>0 稳定；λ→0 与 λ=0 一致。
- **权重**：wₖ 加权与非加权的一致性（w≡1）。
- **规范/标度**：返回链 `scaling == 1`；数据决定整体标度（非 0 解）。
- **键维**：two-site `trunc` 下 bonddim ≤ 键帽；NoTruncation 自由增长。

## 7.9 已知陷阱

1. **欠采样与秩塌缩**：N 太小或样本聚集 ⇒ `M_s` 奇异，ALS 解漂移；
   必须支持 λ>0 或截断 SVD 求解，并文档化 N 的下界（≳ D²·d·L 的经验界）。
2. **ALS 局部极小**：MPS 流形非凸；两点更新 + 自适应增补 + 随机重启缓解；
   与 TCI 相比这是主要理论短板（cross 一步到位，LS 靠迭代）。
3. **语境栈覆盖**（dmrg2.jl 的教训重演风险）：左扫期间右语境只读；
   增补样本后**整栈重算**（`_init_contexts_right!`），不要增量修补。
4. **桶实现的 k-关联**：`Σₖ (ℓₖ⊗rₖ)(ℓₖ⊗rₖ)†` 不可分解成
   `(Σℓℓ†)⊗(Σrr†)`——样本关联必须在桶内保留，否则静默出错（loss 仍降但错误）。
5. **复数振幅**：正规方程的 `b` 用 `āₖ`（无共轭的值乘特征向量见 7.1 推导），
   共轭位置错会得到"共轭目标"——单点解析验证钉死。
6. **噪声底 tol**：外环 tol 低于噪声水平 ⇒ 样本无限增补；
   文档要求 tol ≳ 噪声标准差，或提供 maxres 显式出口。
7. **d 不均匀**：`ds::NTuple` 各位点物理维不同；桶数/单热维按位点取。
