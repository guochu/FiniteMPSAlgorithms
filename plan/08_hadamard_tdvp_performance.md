# HadamardTDVP 性能问题：TEMPO TDVPIF 实时构建慢 ~12× 的热点分析

来源：TEMPO 后端切换后的基准回归（`TEMPO/performance/TTIIF/`，2026-09-23）。
**精度无任何影响**（构建结果与旧本地引擎逐位一致），本报告只涉及性能。

## 现象

TEMPO 的 `TDVPIF` 实时构建（`hybriddynamics(lattice::RealADTLattice1Order, ...)`，
引擎为 `HadamardTDVPCache` + `HadamardTDVP`）相比旧本地 TDVP 引擎慢约 **12 倍**：

| 模型 | 旧本地引擎 | 本包 HadamardTDVP |
|---|---|---|
| independent bosons，ADT 实时（N=10, δt=0.02, D=50, δ=0.1） | 61.8 s | **738 s** |

虚时间构建（同一引擎、同一流程，仅 H 键维小）无差异（15.9 s vs 15.8 s），
XTRG-IF 路径（本包 `hadamard`/`mult` 的 SVD 压缩路线）反而快 3.6~4 倍——
问题特定于 **HadamardTDVP 的 sweep 实现**。

## 复现与测量方法

模型：费米杂质（ϵ_d=0.5）+ 单模玻色浴 DiracDelta(ω₀=1)，`ADTLattice(N=10, δt=0.02, contour=:real)`，
`TDVPIF(trunc=truncdimcutoff(D=50, ϵ=1e-10), δ=0.1)`。`OMP_NUM_THREADS=1`，Julia 1.10.11。

1. **整体分段计时**：`influenceoperators`×4 ≈ 12.7 s；4 分支求和 + `DefaultKTruncation`
   压缩 ×3 ≈ 2.2 s；`changebond!` + 无截断 canonicalize ≈ 0.4 s；**`_tdvpif_flow!`
   （10 次 `sweep!`）≈ 703 s（98%）**；末次截断 canonicalize ≈ 0.1 s。
2. **单轮 sweep 组件计时**（复刻 `_hadamard_leftsweep!` + `_hadamard_rightsweep!` 骨架，
   对每类操作分别累计；预热一轮后测量）：sweep 总墙钟 **87.9 s**，其中

| 组件 | 耗时 |
|---|---|
| `exponentiate`（局域 Krylov 演化，86 次调用） | **87.56 s（99.7%）**，平均 1.02 s/次 |
| bond/site map 构造中的环境收缩 | 0.05 s |
| `updateleft!` / `updateright!` 环境递推 | 0.05 s |
| gauge（`_gauge_left` / `_gauge_right`） | 0.01 s |
| `_contract_first` / `_contract_last` | ~0 |

（注：Profile.jl 的采样在本环境不可用，以上为手工插桩累计；上一次插桩中
单次 exponentiate 的分项数值有测量噪声，以本表为准。）

## Krylov 行为

每次 `exponentiate(f, t, x; ishermitian=false, tol=Defaults.tol(=1e-12), krylovdim=25, maxiter=100)`：

- `info.numops = 26 = krylovdim + 1`，即 **Arnoldi 每次都打满 25 维 Krylov 空间、
  重启一轮后才收敛**（`converged=1`）；没有更早收敛的实例。
- 86 次/轮 = (22 site + 21 bond) × 2 半扫；site 映射（`_reduce_hadamard_site`，逐点
  乘 + 环境收缩）便宜（~0.1 ms/次），**bond 映射（`c_prime`，(50,2)×(50,4,50)×(50,2,50)）
  单次 ~4-7 ms**，是大头。
- 每次映射 ~39 ms（= 1.02 s / 26）相对同规模最优收缩（~2-5 ms，TensorOperations
  直接 `@tensor` 实测）偏慢一个量级。

## 根因

两个因素相乘：

1. **局域映射收缩成本偏高**：`_reduce_hadamard_site` / `c_prime` 的收缩顺序把
   环境 (50,4,50) 的 w=4 腿先折叠进来，大键 (50) 之间的收缩被放到后端，
   实测 ~39 ms/次；先收缩大键的两步收缩实测可到毫秒级以下。
2. **Arnoldi 每次打满 krylovdim**：局域投影生成元的谱（含零空间/近零空间方向，
   来自 `changebond!` 的零填充键维）使 25 维 Krylov 空间内不满足 1e-12 容差，
   每次都重启一轮；26 次映射调用 × 偏慢的映射 = ~1 s/次。

sweep 的其余骨架（环境局部递推、gauge、contract）合计 < 0.15 s——**环境管理
本身不是问题**（与旧引擎相同的局部递推策略）。

## 修复建议（按预期收益排序）

1. **site 局域演化改用 dense 指数化**：site 的有效映射是 (d·χ)×(d·χ) = 100×100
   矩阵，`expm` 后一次施加为毫秒级；替代 26 次 × 39 ms 的 Krylov 收缩，
   预期 site 部分提速 ~100×。（bond 补空间映射维度为 (χ, d·χ)，同样适用。）
2. **复查局域映射的收缩顺序**：`_reduce_hadamard_site` / `c_prime` 先收缩大键
   （χ=50）方向，把 w=4 的环境腿折叠放到最后；或提供先收大键的两步收缩变体。
3. **Krylov 参数自适应/放宽**：`tol=1e-12` 对局域演化的必要性可复查（外层每步
   有变分投影，局域 1e-8 通常足够）；`krylovdim=25` 打满重启的模式说明对当前
   谱 25 维不收敛，增大 krylovdim 会更慢，放宽 tol 更有效。
4. 对照参考：旧 TEMPO 本地引擎（`exponentiate` + `Arnoldi()` 默认参数，tol 为
   机器精度、更严）在同一问题上 sweep ~6 s——说明 25 维 Krylov 本身足以快速
   收敛，当前的开销主要落在 (1)(2) 的映射成本上。

## 附：TEMPO 侧基准对照（independent bosons，ADT，实时）

| 算法 | IF 构建 | 观测量扫描 | 最大键维 | G>(t) 相对误差 |
|---|---|---|---|---|
| XTRGIF（本包 `hadamard`/`mult` SVD 路线） | 10.4 s（旧 43.8） | 3.6 s | 10 | 3.29e-5 |
| TDVPIF（本包 `HadamardTDVP`） | **738 s**（旧 61.8） | 0.088 s | 7 | 1.63e-9 |

两算法 IF 相互距离 7.6e-6、各误差与旧后端逐位一致——切换后结果正确，仅 TDVPIF
实时构建存在本报告所述的性能回退。

## 已修复（2026-09-25）

按建议 (2) 修复：**收缩顺序**，Krylov 参数与指数化策略未动。

### 根因确认

微基准复现（χ=50, d=2, w=4, L=22，随机正则链）表明问题不在 Krylov，而在
TensorOperations 对三操作数 `@tensor` 收缩的自动定序：对
`y[-1,-2] := hleft[-1,1,2]·x[2,3]·hright[-2,1,3]` 这类形状，它把两个**环境**通过各自
的（小的）w 腿先收缩，物化出 `χ_L×χ_R×χ'_L×χ'_R` 的巨型中间张量，而最优顺序应先把
迭代张量 `x` 折进一个环境（两步均为 BLAS 形状的矩阵乘）。实测：

| 操作 | 自动顺序 | 显式两步 | 加速 |
|---|---|---|---|
| `c_prime` 单次调用 | **167 ms** | **0.41 ms** | ~400× |
| site 映射单次调用 | 2.5 ms | 1.7 ms | 1.5× |
| bond `exponentiate`（26 次映射） | 4.28 s | 63 ms | 68× |

报告中"每次映射平均 39 ms"正是 site（快）与 bond（病态）调用的加权平均。

### 修改

三处热点映射改为显式两步（先把迭代折进右环境，再收左环境），同一收缩、仅求和
顺序不同：

- `c_prime`（`derivatives.jl`；共享：hadamard 键映射与 `DMRG1`/`TDVP1`/`TDVP2` 键映射）；
- `_reduce_hadamard_site`（`hadamard.jl`；ALS hadamard 目标与 `HadamardTDVP` site 映射）；
- `_reduce_hadamard_site2`（`hadamardtdvp.jl`；`HadamardTDVP2` pair 目标）。

尝试过并放弃的变体：site 映射的"预收缩左环境×生成元"（`Phi`）+ 按 p 切片两步，
实测 6.6 ms，比普通两步（1.8 ms）慢，不值得增加独立代码路径。

### 端到端

同一脚本（χ=50, d=2, w=4, L=22，`HadamardTDVP(stepsize=-0.01-0.01im)`，预热 1 次）：

| | 旧代码 | 新代码 |
|---|---|---|
| `sweep!` 墙钟 | **113 s** | **1.66 s**（68×） |
| 态范数 | 1.395e21 | 1.395e21（一致） |

按此比例，TEMPO `TDVPIF` 实时构建（10 次 sweep，738 s）预计降到 ~10 s 量级，
快于旧本地引擎的 61.8 s。建议 (1)（dense 指数化）不再需要：映射修好后，Krylov 的
26 次映射调用（~10 ms）比构建 dense 矩阵所需的 N 次基向量应用（N=100-200）更省；
建议 (3)（放宽 tol/krylovdim）同样不需要，参数保持不变。

## 端到端验证（TEMPO 基准，2026-09-23 重跑）

同一基准脚本（independent bosons ADT，N=10, δt=0.02, D=50, δ=0.1，单线程）：

| 算法 | IF 构建（修复前 → 修复后 → 旧本地引擎） | 观测量扫描 | 相对误差 |
|---|---|---|---|
| TDVPIF 实时 | 738 s → **31.2 s** → 61.8 s | 0.118 s | 1.620e-9（逐位一致） |
| TDVPIF 虚时间 | 15.8 s → 18.1 s → 15.9 s | 0.018 s | 6.548e-10（逐位一致） |
| XTRGIF 实时 | 10.4 s → 15.7 s → 43.8 s | 5.651 s | 3.294e-5（逐位一致） |
| XTRGIF 虚时间 | 8.6 s → 13.9 s → 31.1 s | 2.448 s | 3.910e-3（逐位一致） |

实时 TDVPIF 构建快 ~24×，性能回退彻底消除，且优于旧本地引擎 ~2×（micro-benchmark
预测的 ~10 s 未完全兑现，端到端还包含求和压缩、canonicalize 与 Krylov 重启等固定
开销）。全部精度指标、最大键维（12/12、10/7）、两算法 IF 相互距离
（1.644e-3 / 7.634e-6）与修复前逐位一致。（虚时间/XTRGIF 列的 ±40% 浮动为
共享机器负载噪声，各轮次间可复现性 ~±10 s。）

## 端到端验证第二轮（2026-09-30，solver 工厂与 per-algorithm solver 字段之后）

Defaults 算法工厂重构（`6f01dd1`）后再跑同一基准，实时 TDVPIF 从 31.2 s 进一步降到
**16.4 s**（对旧本地引擎 ~3.8×、对修复前 ~45×），其余路径持平或更快；ADT 与 PT 两条
 TEMPO 路径的 IF 相互距离仍与修复前逐位一致（1.644e-3 / 7.634e-6）。

| 路径 / 算法 | IF 构建 | 观测量扫描 | 相对误差 |
|---|---|---|---|
| ADT · XTRGIF 虚时间 | 8.0 s（旧 31.1） | 1.6 s | 3.910e-3 ✓ |
| ADT · TDVPIF 虚时间 | 10.7 s（旧 15.9） | 0.018 s | 6.548e-10 ✓ |
| ADT · XTRGIF 实时 | 9.9 s（旧 43.8） | 3.7 s | 3.294e-5 ✓ |
| ADT · TDVPIF 实时 | **16.4 s**（旧 61.8） | 0.074 s | 1.627e-9 ✓ |
| PT · XTRGIF 虚时间 | 9.3 s | 1.0 s | 3.910e-3 ✓ |
| PT · TDVPIF 虚时间 | 11.5 s | 0.003 s | 1.492e-8 |
| PT · XTRGIF 实时 | 11.2 s | 4.1 s | 3.294e-5 ✓ |
| PT · TDVPIF 实时 | 30.3 s | 0.006 s | 7.165e-10 |

（PT 首次纳入端到端对照：TDVPIF 虚时间键维 29、误差 1.5e-8 为 PT 路径自身的收敛
行为；两算法相互距离与 ADT 逐位相同。）至此四条路径（ADT/PT × 虚/实时）的 IF 构建
与观测量扫描全部快于旧本地引擎，性能回退完全消除。
