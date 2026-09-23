# 07 algorithms/reconstruct.jl：已知振幅的 MPS 重构——TCI 的二次优化替代

**动机**：TCI（tensor cross interpolation）从已知的部分振幅出发，通过连续的矩阵插值
（pivot 更新）构造 MPS 逼近高维张量。其本质是：**已知 MPS 的一部分振幅，重构 MPS**。
本文档给出另一条路线：把重构写成**二次优化**（最小二乘）问题，用与现有
`DMRG1`/`seq2seq` 同族的 **single-site ALS** 扫掠求解。与 TCI 的逐 pivot 精确插值不同，
LS 路线把全部样本**联合**逼近：

| | TCI（cross） | 本文档（LS） |
|---|---|---|
| 样本角色 | pivot：链上**精确**插值 | 数据点：全体**联合**最小二乘 |
| 求解方式 | 逐 rank-1 更新（maxvol） | 正规方程 + single-site ALS 扫掠 |
| 振幅含噪声 | 精确复现噪声（插值性质决定） | 噪声被平均（主要卖点） |
| 键维控制 | 构造后另做截断 | 固定 `alg.D`（single-site，同 DMRG1） |
| 失效模式 | pivot 退化、rank-1 相关 | 欠采样、ALS 局部极小 |
| 每步代价 | O(L·D²·d) 插值 | O(N·D⁴) 构造 + O(d³D⁶) 局域解 |

## 7.1 问题的二次优化形式（loss 构造）

设目标张量 `A(𝐱)`, `𝐱 = (x₁,…,x_L) ∈ ⊙ⱼ{1..dⱼ}`。已知 **样本集**
`S = {(𝐱⁽ᵏ⁾, aₖ)}ₖ₌₁..N`，其中 `aₖ ∈ ℂ` 是基矢 `|𝐱⁽ᵏ⁾⟩ = ⊗ⱼ|xⱼ⁽ᵏ⁾⟩` 下的
**实际振幅**（允许含噪声）。求键维 `D` 的 MPS `ψ`（波函数）最小化

```
ℒ(ψ) = Σₖ |⟨𝐱⁽ᵏ⁾|ψ⟩ − aₖ|²
```

其中 `⟨𝐱⁽ᵏ⁾|ψ⟩` 是波函数 `ψ` 在基矢 `|𝐱⁽ᵏ⁾⟩` 上的投影（振幅）。即
`‖P_S ψ − a‖²`，`P_S = Σₖ |𝐱⁽ᵏ⁾⟩⟨𝐱⁽ᵏ⁾|` 是样本投影算符、
`a = Σₖ aₖ|𝐱⁽ᵏ⁾⟩`。关键结构观察：

- `P_S = Σₖ ⊗ⱼ |xⱼ⁽ᵏ⁾⟩⟨xⱼ⁽ᵏ⁾|` 是 **N 个直积投影算符之和**；
  `a` 是 **N 个直积态之和**——本问题 = `linsolve(P_S, a)` 的特例，
  但 `P_S` 的 MPO 键维 = N（不实用），专用实现按 **样本维度显式求和**（见 7.4）。
- `ℒ` 在全张量空间是凸二次；MPS 流形上的 ALS 扫掠继承单调性
  （精确局域解 ⇒ loss 单调，与 seq2seq/DMRG 测试纪律一致）。
- **无需 KKT 尺度恢复**：正规方程是非齐次的，数据项本身固定整体标度
  （对照 `dmrg2.jl` 驱动器的 `lmul!`——此处不需要）。

**局域化（site s）**：固定其余位点，记左/右语境

```
ℓₖ = ℓ_s(𝐱⁽ᵏ⁾) = ψ₁[x₁]⋯ψ_{s−1}[x_{s−1}] ∈ ℂ^{D_{s−1}}
rₖ = r_s(𝐱⁽ᵏ⁾) = ψ_{s+1}[x_{s+1}]⋯ψ_L[x_L] ∈ ℂ^{D_s}
```

则 `⟨𝐱⁽ᵏ⁾|ψ⟩ = ℓₖ · ψ_s · (e(x_s⁽ᵏ⁾) ⊗ rₖ)`，特征向量 `fₖ = ℓₖ ⊗ eₖ ⊗ rₖ`
（`eₖ = e(x_s⁽ᵏ⁾)` 为物理基）。正规方程（`z = vec(ψ_s)`）：

```
(M_s + α·R_s) z = b_s,   M_s = Σₖ fₖ fₖ†      （d·D_{s−1}·D_s 维，半正定 Hermitian）
b_s = Σₖ āₖ fₖ
```

`M_s` 按**物理桶**结构存储：`M_s = Σ_p B_s(p)·E_p`（`E_p` 物理单热阵），
`B_s(p) = Σₖ: x_s⁽ᵏ⁾=p (ℓₖ⊗rₖ)(ℓₖ⊗rₖ)†`（4 腿 `D_{s−1}×D_{s−1}×D_s×D_s`，
每物理值一个桶）。求和**不可**在 k 上分解（ℓ 与 r 的样本关联必须保留），
桶内必须逐样本外积累加。

**正则化（与 seq2seq 相同的 ridge）**：欠采样（N 小于有效自由度）时 `M_s`
奇异/病态，ALS 解漂移。与 `ml/seq2seq.jl` 一致，在**局域正规方程**上加
Hilbert-Schmidt ridge `α·‖ψ‖²_HS`（默认 `α = 0.01`，沿用 seq2seq/MPSLearning）：
局域 Hessian 加 `α·R_s`，其中

```
R_s = g_s ⊗ I_d ⊗ g_{s+1},   g_s = ⟨ψ_{<s}|ψ_{<s}⟩_HS   (D_{s-1}×D_{s-1} 转移矩阵)
```

即 `gstorage` 第三环境栈（seq2seq 的 `gstorage` 对 MPS 的秩 3 版本：
`g ← Σ_{p} conj(A[aL,p,aR])·g·A[aL',p,aR']` 转移）。要点：
- ridge **只加在局域求解上，不计入报告的 loss**（seq2seq 的纪律）；
- ridge 是**规范依赖**的（真正的目标 ℒ 规范不变）——它只在当前规范下
  条件化局域解，这正是稳定 ALS 所需；
- α>0 的解偏向小范数（低秩偏置，对 MPS 有利）；α = 0 退化到纯插值。

single-site 固定键维意味着样本不能"撑开"键维：表达能力完全由 `alg.D` 决定，
这与 DMRG1（`linsolve` 等）的取舍一致——键型先验给定，靠 ridge 与样本增补保稳定。

## 7.2 算法配置（追加到 `src/algorithms/algdefs.jl`）

```julia
"""
	ALSRecon(; maxiter=Defaults.maxiter, tol=Defaults.tol, D=Defaults.D,
	        α=0.01, nadd=8, nbuffer=1024, verbosity=0)

Sample-amplitude MPS reconstruction by quadratic (least-squares) optimization —
the variational alternative to tensor cross interpolation. Single-site ALS sweeps
with a fixed bond profile `alg.D` (the DMRG1 discipline: the out-of-place entry
points draw a random guess of bond `alg.D`, the in-place route re-fits the
caller's guess with `changebond!`). `α ≥ 0` is the Hilbert-Schmidt ridge added
to the local normal equations for conditioning (the seq2seq regularization; not
part of the reported loss); `nadd`/`nbuffer` drive the adaptive
sample-enrichment loop (7.5).
"""
@kwdef struct ALSRecon <: IterativeMPSAlgorithm
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	D::Int = Defaults.D
	α::Float64 = 0.01        # HS ridge on the local solves (seq2seq regularizer)
	nadd::Int = 8            # 每 轮自适应新增样本数
	nbuffer::Int = 1024      # 候选缓冲池大小（随机粗采样, 残差在其中挑 nadd）
	verbosity::Int = 0
end
```

## 7.3 接口（`src/algorithms/reconstruct.jl`，唯一导出接口 `reconstruct`）

`alg` 按包约定以**位置参数**传入（带默认值，对照 `linsolve(A, y, alg::DMRG1=DMRG1())`、
`seq2seq(xs, ys, alg::DMRG1=DMRG1())`）：

```julia
# 黑盒 oracle：自适应采样（7.5）+ 内环 ALS，主入口
# （稠密张量 A 的调用方直接传 Afun = (𝐱) -> A[𝐱...]）
reconstruct(Afun::Function, ds::NTuple{L,Int}, alg::ALSRecon = ALSRecon())
	-> (ψ::CanonicalMPS, info::NamedTuple{(:loss, :maxres, :nsamples, :rounds)})

# 固定样本集（无自适应）：samples::Vector{Pair{NTuple{L,Int},T}} 或 (𝐱, a) 元组
reconstruct(samples, ds, alg = ALSRecon())

# in-place 精化（样本固定；调用方提供初值；键型由 changebond! 调整到 alg.D）
reconstruct!(ψ::CanonicalMPS, samples, alg::ALSRecon) -> (ψ, loss)
```

命名：`reconstruct`（从振幅数据重构链）；弃用 `fit`（与 StatsAPI 混淆）、
`tci`（本路线明确**不是** cross）、`recovery`（冗长）。

## 7.4 求解器：样本环境 single-site ALS 扫掠（伪代码）

```
输入: 样本集 S = {(𝐱⁽ᵏ⁾, aₖ)}、初值 ψ⁰（randommps, D = alg.D；in-place 时
      先 changebond!(ψ; D = alg.D)，对照 seq2seq!）
约束: ψ 的位点张量 A[aL, p, aR]（与包约定一致）；扫掠规范移动复用 mult.jl 的
      _gauge_left/_gauge_right（seq2seq 同款 QR/LQ 移规范）

内环一轮（ALS 到收敛）:
  # ---- 右语境预计算（右→左） ----
  R[L]   = ones(N, 1)                    # r_L(𝐱ₖ) = 1
  for s = L-1 … 1:
      R[s][k, :] = ψ_{s+1}[x_{s+1}⁽ᵏ⁾] · R[s+1][k, :]     # 每样本一行, O(N·d·D)
  # ridge 栈（seq2seq 的 gstorage, ⟨ψ|ψ⟩_HS 转移, 秩 3 版）
  g[L+1] = ones(1, 1)
  for s = L … 1:
      g[s] ← Σ_{p} conj(ψ_s[aL, p, aR])·g[s+1]·ψ_s[aL', p, aR']

  # ---- 左→右扫（seq2seq 的 leftsweep! 结构） ----
  Lmat[1] = ones(N, 1)                   # ℓ_1(𝐱ₖ) = 1
  for s = 1 … L:
      # 单点正规方程: fₖ = ℓₖ ⊗ e(x_s⁽ᵏ⁾) ⊗ rₖ
      M ← Σ_p B(p)·E_p   （桶: 按 x_s 取值分组, 桶内逐样本外积累加）
      b ← Σₖ āₖ (ℓₖ ⊗ eₖ ⊗ rₖ)
      w ← (M + α·(g[s] ⊗ I_d ⊗ g[s+1]))⁻¹ b        # ridge 条件化的稠密解, 维 d·D²
      loss_s ← 当前全局数据目标（由 w、M、b 直接组合, seq2seq 的 _site_loss）
      q, r ← _gauge_left(w);  ψ_s = q;  ψ_{s+1} ← ψ_{s+1}·r
      Lmat[s+1][k, :] = Lmat[s][k, :] · ψ_s[x_s⁽ᵏ⁾]        # 左语境传递
      g[s+1] ← ⟨ψ_≤s|ψ_≤s⟩_HS 转移                          # ridge 栈同步传递

  # ---- 右→左扫（对称, LQ 移规范, 右语境栈现算） ----
  … …

  loss ℒ = Σₖ |⟨𝐱⁽ᵏ⁾|ψ⟩ − aₖ|²             # 精确局域解 ⇒ 单调（ridge 不计入）
  |ℒ_n − ℒ_{n−1}|/ℒ_{n−1} < tol → 收敛
```

**每扫两遍语境的更新规则**（与 DMRG1 的 hstorage 同型：扫过即更新，
**右语境栈在左扫期间只读**——dmrg2.jl 的教训：左扫每 site 只允许一次左语境更新，
不得覆盖右语境）：

```
左语境: ℓ_{s+1}(𝐱ₖ) ← ℓ_s(𝐱ₖ)·ψ_s[x_s⁽ᵏ⁾]     (N·D² 每位点)
右语境: r_s(𝐱ₖ)   ← ψ_{s+1}[x_{s+1}⁽ᵏ⁾]·r_{s+1}(𝐱ₖ)
```

**复杂度**：每 site `O(N·D⁴)` 桶构造 + `O(d³·D⁶)` 稠密解 + `O(N·d·D²)` 语境传递；
样本存储 `O(L·N·D)`。大 `N`/大 `D` 时局域解可换 CG（`M` 只需 mat-vec：
`z ↦ Σₖ fₖ(fₖ†z)`，`O(N·d·D²)` 每次迭代）——首版用稠密解，预留分派。

## 7.5 自适应采样（替代 cross 的 pivot 选取）

```
S ← 初始样本: nbuffer 个随机点 + 若干 argmax|A| 点
repeat（至多 maxiter 外环）:
    (ψ, ℒ) ← 内环 ALS(S)                        # 7.4, 收敛即止
    在 nbuffer 个新随机候选 𝐱′ 上算残差 ρ(𝐱′) = |⟨𝐱′|ψ⟩ − A(𝐱′)|
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
| `ALSRecon` | algdefs.jl | 算法配置（7.2；single-site, 固定 `D`） |
| `reconstruct(Afun, ds, alg)` | reconstruct.jl | 主入口（自适应） |
| `reconstruct(samples, ds, alg)` | reconstruct.jl | 固定样本 |
| `reconstruct!(ψ, samples, alg)` | reconstruct.jl | in-place（`changebond!` 到 `alg.D`） |
| `struct ALSReconCache` | reconstruct.jl | ψ、样本矩阵 X（L×N）、值 a、语境栈 Lmat/Rmat、ridge 栈 gstorage |
| `_init_contexts_right!(c)` | reconstruct.jl | 右→左预计算 Rmat + gstorage（对照 `_init_hstorage_right!` / seq2seq） |
| `_left_transfer!(c, s)` / `_right_transfer!(c, s)` | reconstruct.jl | 语境传递（扫掠中） |
| `_g_transfer!(c, s)` | reconstruct.jl | ridge 栈 ⟨ψ\|ψ⟩_HS 转移（秩 3 版 seq2seq `_g_updateleft`） |
| `_ls_reduce_site(c, s)` | reconstruct.jl | 单点正规方程 (M, b)（桶实现，7.1） |
| `_add_ridge!(M, c, s, alg)` | reconstruct.jl | 局域 Hessian 加 `α·(g_s ⊗ I ⊗ g_{s+1})`（对照 seq2seq `_add_ridge!`） |
| `_ls_solve(c, s, alg)` | reconstruct.jl | `(M + α·R) \ b` → `w`（附 `(w, t, H)` 供 `_site_loss`） |
| `_site_loss(c, s, w, t, H)` | reconstruct.jl | 该 site 更新后的精确全局 loss（seq2seq 同款） |
| `leftsweep!/rightsweep!/sweep!(c, alg::ALSRecon)` | reconstruct.jl | 统一扫掠接口（包约定） |
| `_ls_loss(c)` | reconstruct.jl | 当前 ℒ（收敛判据 + 单调性测试；ridge 不计入） |
| `_residual_candidates(Afun, ψ, nbuffer)` | reconstruct.jl | 候选池残差评估（自适应外环） |
| `_enrich_samples!(c, Afun, nadd)` | reconstruct.jl | 样本增补 + 语境栈扩容 |

## 7.7 与现有机制的关系 / 原型基线

- **原型基线（验证用）**：`P_S` 作为显式 MPO（Σₖ 直积投影，键维 = N）+
  `a` 的 MPS 构造（N 个 prodmps 之和）→ 现有 `linsolve(P_S, a, DMRG1(…))`。
  只在小 N 下做交叉验证，不作实现路线。
- **正则化与扫掠结构**：与 `ml/seq2seq.jl` 同构——三环境栈
  （逐样本二次/线性栈 + HS ridge 栈）、`_add_ridge!` 加在局域 Hessian、
  ridge 不计入报告 loss、默认 `α = 0.01`（MPSLearning）、QR/LQ 移规范 +
  每 site 稠密正规方程；仅 ridge 转移是 MPS 秩 3 版（对照 seq2seq 的秩 4
  `_g_updateleft`），样本语境按"振幅特征向量"替代 seq2seq 的 x/y 链转移。
- **键维纪律**：与 `DMRG1` 相同——固定 `alg.D`，out-of-place 入口
  `randommps(...; D)`，in-place 入口 `changebond!`（对照 `seq2seq!`）。
- **扫掠纪律**：leftsweep 每 site 恰好一次左语境更新（右语境只读，
  dmrg2.jl 的教训）；精确局域解 ⇒ loss 单调（测试钉死）。
- **无 KKT 尺度恢复**：正规方程非齐次，解的标度由数据决定；
  驱动器不需要 `lmul!`/`setscaling!` 补偿（对照 dmrg2.jl 的教训）。
- **采样算符视角**：P_S 永不显式构造；一切环境按样本维显式求和。

## 7.8 测试要点

- **精确插值**：小系统（L=4, d=2），A = 已知低键 MPS 稠密化；N = d^L 全采样
  → `todense(ψ)` 与 A 机器精度一致。
- **欠采样泛化**：N ≪ d^L 但 A 键维 ≤ D → 自适应循环收敛到全张量误差 ~1e-10。
- **loss 单调**：小系统上 `monotone(khist)`（与 seq2seq/DMRG 测试同款）。
- **噪声稳健（卖点）**：aₖ = ⟨𝐱ₖ|A⟩ + σ·复噪声；LS 重构误差 ~σ 级，
  而 TCI 式精确插值误差不随 σ 缩小（对比基线）。
- **正则化**：重复/相关样本下 α=0 病态 vs α>0 稳定；α→0 与 α=0 解一致；
  α 量级参照 seq2seq 默认 0.01。
- **规范/标度**：返回链 `scaling == 1`；数据决定整体标度（非 0 解）。
- **键维**：out-of-place 结果 `bonddim == alg.D`；in-place 的 `changebond!`
  路径与 `seq2seq!` 一致。

## 7.9 已知陷阱

1. **欠采样与秩塌缩**：N 太小或样本聚集 ⇒ `M_s` 奇异，ALS 解漂移；
   必须保持 ridge α>0（seq2seq 默认 0.01）或截断 SVD 求解，
   并文档化 N 的下界（≳ D²·d·L 的经验界）。
2. **single-site 的表达能力上限**：固定 `alg.D`，无键维增长机制
   （对照 DMRG2）——目标链的真实键维超过 `alg.D` 时只能逼近到截断极限；
   需要在文档里说明 D 的选取（先验已知键维，或从小到大扫描）。
3. **ALS 局部极小**：MPS 流形非凸；随机重启 + 自适应增补缓解（seq2seq 用
   0.1·randn 小值初值避免坏起点——同款处理）；
   与 TCI 相比这是主要理论短板（cross 一步到位，LS 靠迭代）。
4. **语境栈覆盖**（dmrg2.jl 的教训重演风险）：左扫期间右语境只读；
   增补样本后**整栈重算**（`_init_contexts_right!`），不要增量修补。
5. **桶实现的 k-关联**：`Σₖ (ℓₖ⊗rₖ)(ℓₖ⊗rₖ)†` 不可分解成
   `(Σℓℓ†)⊗(Σrr†)`——样本关联必须在桶内保留，否则静默出错（loss 仍降但错误）。
6. **复数振幅**：正规方程的 `b` 用 `āₖ`（共轭位置见 7.1 推导），
   共轭位置错会得到"共轭目标"——单点解析验证钉死。
7. **噪声底 tol**：外环 tol 低于噪声水平 ⇒ 样本无限增补；
   文档要求 tol ≳ 噪声标准差，或提供 maxres 显式出口。
8. **d 不均匀**：`ds::NTuple` 各位点物理维不同；桶数/单热维按位点取。
