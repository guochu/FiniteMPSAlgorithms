# FiniteMPSAlgorithms 实现计划

无对称性的有限 MPS / MPO 张量网络算法包。目标是用普通 `Array` 存储张量、用 `TensorOperations` 收缩，
把 MPSKit 的核心算法干净地实现一遍。底层张量分解操作直接采用 `TEMPO/src/tensorops` 的实现。

**范围**：
- 只支持基于 MPS 和 MPO 的算法；
- **只实现 single-site 算法**；局域本征求解直接用 **KrylovKit**（不自写 Lanczos）；
- 算法配置只有两种：`SVDCompression`（SVD 扫描路线）与 `DMRG1`（ALS 变分路线，**定义参照 TEMPO**）；
  `mult` / `add` / `compress` 是截断算法的唯一导出接口，按算法类型分派；
- DMRG / TDVP 统一暴露 `leftsweep!` / `rightsweep!` / `sweep!` 三个接口（TDVP 的 `sweep!` = 一个完整时间步）；
- TEBD 只提供基本构件：`AbstractGate`（`UnitaryGate` / `GenerateGate`）+ `apply!`，
  `UnitaryGate` 构造时检查幺正性；时间演化由调用方循环施加门。

**不实现**：算符项层（QTerm/QuantumOperator 等）、2-site 算法、correlation（双时关联）、VOMPS、
超算符/Lindblad、热态构造、自写 Lanczos、`timeevo!`/`tdvp_step!`/QuantumCircuit/Stepper 等循环与编排结构、
stable_* 变体、extensions.jl（texp/零空间基等）。

## 参考实现对应关系

| 本包 | 来源 |
|---|---|
| 底层张量操作（截断/QR/LQ/SVD/张量分解） | [TEMPO/src/tensorops](file:///home/guochu/Documents/Missile/TEMPO/src/tensorops)（vendor 直接拷贝） |
| `CanonicalMPS` / `CanonicalMPO` 数据布局 | TEMPO 的 `ADT` / `ProcessTensor`（字段 `data`、`s`、`scaling` 完全一致） |
| `SVDCompression` / `DMRG1` / `Orthogonalize` 算法定义 | TEMPO `src/algorithms.jl`（原样语义：`TruncationWithD`、`initguess`、`callback` 等） |
| `MPO` / `MPOHamiltonian`、门演化、ALS 环境 | [QuantumSpins](file:///home/guochu/Documents/Meteor/QuantumSpins)（`MPO`、`apply_gates`、`iterativeariths`） |
| 算法清单与组织方式 | MPSKit（DMRG1、TDVP、multiplicators 变分压缩、激发态、MPOHamiltonian） |

注意：MPSKit 的 `plansor`/flip/对称张量设施在本包中**不实现**——所有张量都是普通稠密 `Array`，指标排列用 `permutedims` 即可。

## 设计原则

1. **无对称性**：张量即 `Array{T,N}`；不引入 Abelian sector、fusion tree、`flip`、`plansor`。
2. **两层结构**：底层 `tensorops`（纯张量工具，TEMPO 原样移植）+ 上层网络层（states/operators 与 algorithms）。
3. **收缩一律走 `TensorOperations.@tensor`**；矩阵分解走 `MatrixAlgebraKit`（TEMPO tensorops 的后端）。
4. **数据布局与 TEMPO 严格一致**：
   - MPS 位点张量 `Array{T,3}`，轴序 `(左键 aL, 物理 p, 右键 aR)`；
   - MPO 位点张量 `Array{T,4}`，轴序 `(aL, p_out, aR, p_in)`；
   - `CanonicalMPS`/`CanonicalMPO` 含 `s::Vector{Union{Missing,Vector{R}}}`（L+1 个键的 Schmidt 值）与
     `scaling::Ref{Float64}`（**逐位点**缩放语义：总缩放 = `scaling^L`，与 TEMPO 相同）；
   - 所有构造器**强制左右边界维 = 1**（不设 `isstrict` 查询）。
5. **类型分工**：
   - `CanonicalMPS`（states/）：正则形式量子态（右正则 + Schmidt 谱），对应 `ADT`；
   - `CanonicalMPO`（states/）：正则形式 4-指标链，语义为**密度矩阵**（混合态），对应 `ProcessTensor`，边界真空；
   - `MPO` / `MPOHamiltonian`（operators/）：量子算符的 4-指标张量链（只有 `data`）；
     `MPOHamiltonian` 是哈密顿量语义的子类型，`ground_state`/`excited_state`/TDVP 的标准输入。
6. **算法分层**：
   - **精确（严格）算法**（无截断的 `*`、`+`）→ `states/linalg.jl` 与 `operators/linalg.jl`；
   - **截断算法**（`mult`/`add`/`compress`，唯一导出接口）→ `algorithms/`，分派 `SVDCompression` | `DMRG1`；
   - DMRG1 / 激发态 / TDVP → `algorithms/`，统一 `leftsweep!` / `rightsweep!` / `sweep!` 接口。
7. **规范中心用扫描式处理**（QuantumSpins 风格）：不维护显式 center 对象，环境缓存记录中心位点，扫过即 QR/LQ 移中心。
8. **数值稳健**：SVD 用 `SDD()`（MatrixAlgebraKit `SafeDivideAndConquer`，失败自动回退）；缩放因子防溢出；
   截断至少保留 1 个奇异值（`add_back`）。

## 张量与指标约定

```
MPS 张量:   A[aL, p, aR]           Array{T,3}   边界键维 1
MPO 张量:   W[aL, p_out, aR, p_in] Array{T,4}   p_out=输出物理指标, p_in=输入物理指标

CanonicalMPS: data::Vector{Array{T,3}}            (ADT 布局)
              s::Vector{Union{Missing,Vector{R}}} (L+1, s[b+1]=第 b 键 Schmidt 值)
              scaling::Ref{Float64}

CanonicalMPO: data::Vector{Array{T,4}}            (ProcessTensor 布局, 边界真空)
              s, scaling 同上

MPO / MPOHamiltonian: data::Vector{Array{T,4}}    (仅张量链, 左边界维 1)

门(AbstractGate):   positions::NTuple{N,Int}, op::Array{T,2N}   (2 体约定 (i1',i2',i1,i2))

环境:
  ⟨ψA|·|ψB⟩ 转移矩阵  h::AbstractMatrix      (bra 键 × ket 键)
  ⟨ψA|W·|ψB⟩ 环境张量 h::Array{T,3}          (bra 键, W 键, ket 键)
  ⟨hA|·|hB⟩  转移矩阵 h::AbstractMatrix      (hA 键 × hB 键)
```

## 依赖

```toml
[deps]
TensorOperations   # @tensor 收缩
MatrixAlgebraKit   # left_orth!/right_orth!/svd_compact!（TEMPO tensorops 的后端）
KrylovKit          # 局域本征求解 eigsolve、TDVP exponentiate
LinearAlgebra, Random, Statistics, Logging
```

## 源码目录结构

```
src/
├── FiniteMPSAlgorithms.jl
├── defaults.jl                    # Defaults.D / tol / tolgauge / maxiter; DefaultTruncation
├── abstractdefs.jl                # AbstractMPS/AbstractMPO、MPSTensor/MPOTensor 别名、空间函数
├── tensorops/                     # ← TEMPO/src/tensorops 四个文件原样移植（详见 01, 无 extensions）
│   ├── truncation.jl
│   ├── matrixalgebra.jl
│   ├── tensorfactorizations.jl
│   └── distance.jl
├── states/                        # 量子态: CanonicalMPS / CanonicalMPO（详见 02）
│   ├── canonicalmps.jl
│   ├── canonicalmpo.jl
│   ├── orth.jl                    # 规范形 leftorth!/rightorth!/canonicalize!
│   ├── transfer.jl                # updateleft/updateright 转移基元
│   ├── linalg.jl                  # dot/norm/tr/缩放 + 精确算法: MPO·MPS、MPS+MPS
│   ├── initializers.jl            # prod/random/DensityOperator/increase_bond!
│   └── observables.jl             # expectation/熵
├── operators/                     # 算符: MPO / MPOHamiltonian（详见 02）
│   ├── mpo.jl
│   ├── mpohamiltonian.jl
│   ├── orth.jl
│   ├── transfer.jl                # MPO-MPO 重叠、tr 转移
│   ├── linalg.jl                  # dot/tr + 精确算法: MPO·MPO、MPO+MPO
│   └── initializers.jl            # identity_mpo / prodmpo / randommpo
├── environments/                  # 环境缓存（详见 03）
│   ├── finiteenv.jl  overlap.jl  projected.jl
└── algorithms/                    # 算法（详见 03/04）
    ├── algdefs.jl                 # MPSAlgorithm/DMRGAlgorithm; SVDCompression; DMRG1(TEMPO 定义)
    ├── derivatives.jl             # ac_prime / c_prime / CentralHeff（局域求解用 KrylovKit）
    ├── dmrg.jl                    # ground_state; 统一接口 leftsweep!/rightsweep!/sweep!
    ├── excited.jl                 # DMRG1 + ProjectedExpectationCache(无独立算法类型)
    ├── tdvp.jl                    # TDVP1: sweep! = 一个完整时间步(实 dt 即虚时)
    ├── gate.jl                    # AbstractGate / UnitaryGate(检查幺正) / GenerateGate + apply! + swap!
    ├── mult.jl                    # mult 唯一接口(SVDCompression|DMRG1): MPO·MPS / MPO·MPO / MPO·CanonicalMPO
    ├── add.jl                     # add 唯一接口
    ├── compress.jl                # compress 唯一接口
    ├── hadamard.jl                # hadamard_inv/hadamard_inv!/hadamard_div/hadamard_div! 唯一接口(NewtonSchulz, 逐元素除法; 详见 05)
    └── elementwise.jl             # elementwise 唯一接口(Chebyshev, 函数对 MPS 的逐点作用; 详见 06)
```

## 实施里程碑

| 阶段 | 内容 | 计划文档 | 交付判据 |
|---|---|---|---|
| M0 | 包骨架、vendor tensorops、defaults | 01 | tensorops 单测通过 |
| M1 | states/ + operators/ 数据结构、规范形、精确算法（linalg.jl） | 02 | 规范性/正交性/精确收缩数值检验 |
| M2 | algorithms/: mult/add/compress（SVDCompression 与 DMRG1 两路线） | 03 | 与精确小系统对比、ALS 收敛 |
| M3 | DMRG1（sweep 接口 + KrylovKit）、激发态 | 03 | Ising/Heisenberg 基态对角化验证 |
| M4 | 1-site TDVP1（sweep! = 一个时间步） | 03 | 与 exp(H·t) 精确演化对比 |
| M5 | AbstractGate/UnitaryGate/GenerateGate + apply!/swap! | 04 | 幺正检查、门作用/位点交换与精确 U 对比 |
| M6 | 逐元素除法 hadamard_inv/hadamard_div（NewtonSchulz，经典 PDE 扩展） | 05 | 正场收敛、均值远离 1 预缩放、热启动等价、零场保护 |
| M7 | 函数对 MPS 的逐点作用 elementwise（Chebyshev 级数，经典 PDE 扩展） | 06 | exp/ψ² 等与稠密参考对比、nterms 指数收敛、值域自动估计、键维上界 |

## 测试策略

- **基线**：小自旋链（L=8~12）精确对角化；MPO 哈密顿量由 `prodmpo` 组装（键维 1 直积链 + `+`），
  用测试辅助函数转稠密矩阵对比。
- DMRG1：`E0`、局域磁化、二联体熵与精确值对比（~1e-10）；逐 sweep 损失单调下降并以
  `std(kvals)/mean(kvals) < tol` 收敛。
- TDVP：与 `exp(-i·H·t)|ψ⟩` 精确振幅逐分量对比；保范数（1e-12）；实 dt 虚时演化收敛到 DMRG1 的 E0。
- 门演化（TEBD 构件）：`UnitaryGate` 拒绝非幺正输入；单门作用与精确 U 对比；手动循环施加门 vs TDVP 互验。
- 压缩：`mult(h, ψ; alg=SVDCompression(...))` 误差 ≤ 截断误差上界；`DMRG1` ALS 收敛测试。
- 规范形：每键左环境 = I、`s²` 与环境一致、`dot` 规范不变性。

## 已知移植陷阱（来自参考代码审查）

1. QuantumSpins `mpo/orth.jl` 存在未定义变量 `h`（应为 `psi`）的 bug——移植 MPO 规范形时需重写并测试。
2. QuantumSpins `_check_mps_space` 只检查左边界；本包两种边界都强制检查（与 TEMPO 一致）。
3. TEMPO 的 `scaling` 是**逐位点**语义（`dot` 中为 `scaling^L`），务必保持一致，不要与 DMRG 常见的全局缩放混淆。
4. 本包只实现 single-site 算法（TEMPO/QuantumSpins 的 DMRG 亦为 1-site，可直接参考；不做 2-site）。
5. MPO 张量的 SVD 分组必须把物理双指标 `(p_out, p_in)` 合在一起处理，之后 `permute(q, (1,2,4,3))` 恢复轴序。

## 详细计划索引

- [01_tensorops.md](file:///home/guochu/Documents/Meteor/FiniteMPSAlgorithms/plan/01_tensorops.md) — 底层张量操作层接口
- [02_structures.md](file:///home/guochu/Documents/Meteor/FiniteMPSAlgorithms/plan/02_structures.md) — states/（CanonicalMPS、CanonicalMPO）与 operators/（MPO、MPOHamiltonian）数据结构、规范形、精确算法
- [03_algorithms.md](file:///home/guochu/Documents/Meteor/FiniteMPSAlgorithms/plan/03_algorithms.md) — algorithms/：mult/add/compress、环境缓存、DMRG1、激发态、TDVP
- [04_gate.md](file:///home/guochu/Documents/Meteor/FiniteMPSAlgorithms/plan/04_gate.md) — AbstractGate/UnitaryGate/GenerateGate + apply!（基本 TEBD 构件）
- [05_hadamard_arith.md](file:///home/guochu/Documents/Meteor/FiniteMPSAlgorithms/plan/05_hadamard_arith.md) — algorithms/hadamard.jl：逐元素除法 hadamard_inv/hadamard_div（Newton–Schulz，经典 PDE 求解扩展）
- [06_elementwise_func.md](file:///home/guochu/Documents/Meteor/FiniteMPSAlgorithms/plan/06_elementwise_func.md) — algorithms/elementwise.jl：函数对 MPS 的逐点作用 elementwise（Chebyshev 级数，经典 PDE 求解扩展）
- [07_ls_reconstruction.md](file:///home/guochu/Documents/Meteor/FiniteMPSAlgorithms/plan/07_ls_reconstruction.md) — algorithms/reconstruct.jl：已知振幅的 MPS 重构 reconstruct（二次优化/最小二乘，TCI 的变分替代）
