# 03 algorithms/：mult/add/compress、环境缓存、DMRG1、激发态、TDVP

**位置约定**：
- **精确（严格）算法**（无截断的 `*`、`+`）在 `states/linalg.jl` 与 `operators/linalg.jl`（文档 02 §2.8）；
- **截断算法**全部在 `src/algorithms/`，**只 export 唯一接口** `mult` / `add` / `compress`，
  以算法类型分派：`SVDCompression`（SVD 扫描路线）与 `DMRG1`（ALS 变分路线，定义参照 TEMPO）；
  `svd_*` / `iterative_*` 等内部实现不导出；无 stable_* 变体；
- **DMRG 只实现 single-site**（TDVP 另有 2-site 变体 `TDVP2`，见 §3.10）；局域本征求解直接用 **KrylovKit**（不自写 Lanczos）；
- DMRG / TDVP 统一暴露 `leftsweep!` / `rightsweep!` / `sweep!` 三个接口。

## 3.1 算法配置（`src/algorithms/algdefs.jl`，参照 TEMPO `src/algorithms.jl`）

```julia
abstract type MPSAlgorithm end
abstract type DMRGAlgorithm <: MPSAlgorithm end

# SVD 扫描压缩路线
struct SVDCompression{T<:TruncationScheme} <: DMRGAlgorithm
    trunc::T; verbosity::Int
end
SVDCompression(trunc::TruncationScheme; verbosity::Int=0)
SVDCompression(; trunc=truncdimcutoff(D=Defaults.D, ϵ=Defaults.tol, add_back=0), verbosity=0)
Base.similar(x::SVDCompression; trunc=x.trunc, verbosity=x.verbosity)

# ALS 单格点变分路线(TEMPO 定义原样)
const TruncationWithD = Union{TruncateDim, TruncateDimCutoff}   # 必须带 D(播种初猜)
const AllowedInitGuesses = (:svd, :pre, :rand)

struct DMRG1{T<:TruncationWithD} <: DMRGAlgorithm
    trunc::T; maxiter::Int; tol::Float64; initguess::Symbol; verbosity::Int; callback::Function
end
DMRG1(trunc::TruncationWithD; maxiter::Int=5, tol::Float64=1e-12,
      initguess::Symbol=:svd, verbosity::Int=0, callback=Returns(nothing))
DMRG1(; trunc::TruncationWithD=DefaultTruncation, kwargs...)
Base.similar(x::DMRG1; kwargs...)

const DefaultMultAlg = DMRG1(DefaultTruncation)
```

## 3.2 mult（`src/algorithms/mult.jl`，唯一导出接口）

```julia
mult(h::AbstractMPO, ψ::CanonicalMPS; alg::DMRGAlgorithm=DefaultMultAlg) -> (CanonicalMPS, err)
mult(hA::AbstractMPO, hB::AbstractMPO; alg=DefaultMultAlg) -> (MPO, err)
mult(h::AbstractMPO, ρ::CanonicalMPO; alg=DefaultMultAlg) -> (CanonicalMPO, err)   # 作用于密度矩阵
# 分派:
#   alg::SVDCompression → svd_* 内部路线(左→右 QR 累积 + 右→左 SVD 截断)
#   alg::DMRG1          → iterative_* 内部路线(ALS 变分, 见下)

# 内部实现(不导出):
svd_mult(h, ψ; trunc::TruncationScheme)          # MPO·MPS
svd_mult(hA, hB; trunc)                          # MPO·MPO
iterative_mult(h, ψ, alg::DMRG1)                 # ALS
iterative_mult(hA, hB, alg::DMRG1)
```

ALS 核心（同文件内）：

```julia
struct MPOMPSIterativeMultCache
    mpo::AbstractMPO; imps::CanonicalMPS; omps::CanonicalMPS
    hstorage::Vector{Array{T,3}}         # ⟨omps|mpo|imps⟩ 三指标环境栈
end
init_hstorage_right(imps, mpo, omps)     # 边界 ones(1,1,1), 右→左填充
sweep!(m, alg::DMRG1; direction) -> kvals
    # 左扫: mpsj = reduceD_single_site(imps[s], mpo[s], C[s], C[s+1]); QR 移中心
    # 右扫: LQ 对称; 结束后 omps[1] *= R
reduceD_single_site(A, X, Cleft, Cright)     # @tensor 局部目标 = Cleft·A·X·Cright

struct MPOMPOIterativeMultCache; mpo; impo; ompo; hstorage::Vector{Array{T,3}}; end
reduceH_single_site(A, m, cleft, cright)     # 局部目标 4 阶张量
# QR 分组 (1,2,4)|(3); LQ 分组 (1)|(2,3,4); permute(q,(1,2,4,3)) 恢复轴序
```

## 3.3 add（`src/algorithms/add.jl`，唯一导出接口）

```julia
add(ψs::Vector{<:CanonicalMPS}; alg::DMRGAlgorithm=DefaultMultAlg) -> (CanonicalMPS, err)
add(ψA, ψB; alg) = add([ψA, ψB]; alg)
# 精确版: Base.:+(ψA, ψB) 在 states/linalg.jl
# 分派同 mult: SVDCompression → svd_add(逐位点 cat + tsvd); DMRG1 → iterative_add(ALS)

struct MPSIterativeAddCache
    omps::CanonicalMPS; imps::Vector{CanonicalMPS}
    hstorage::Vector{<:OverlapCache}     # 每个 imps_n 一个 ⟨omps|imps_n⟩ 环境
end
# 局部目标 = Σ_n cstorage[n][site]·imps[n][site]·cstorage[n][site+1]
```

## 3.4 compress（`src/algorithms/compress.jl`，唯一导出接口）

```julia
compress(ψ::CanonicalMPS; alg::DMRGAlgorithm=DefaultMultAlg) -> (CanonicalMPS, err)
compress(h::AbstractMPO; alg=DefaultMultAlg) -> (AbstractMPO, err)
compress!(x; alg)
# 分派同 mult: SVDCompression → svdcompress(MPS 左→右 SVD / MPO 左 QR + 右 SVD);
#              DMRG1 → iterative_compress!(复用 OverlapCache, 最大化 overlap 的 ALS)
```

## 3.5 环境缓存（`src/environments/`）

```julia
# finiteenv.jl —— ⟨ψ|W|φ⟩ 型缓存(DMRG1/TDVP 主力)
struct ExpectationCache{M<:AbstractMPO, V<:CanonicalMPS}
    mpo::M
    mps::V
    hstorage::Vector{Array{T,3}}    # hstorage[s] = ⟨mps[1:s-1]|W[1:s-1]|mps[1:s-1]⟩
    center::Base.RefValue{Int}      # 规范中心
end
environments(h::AbstractMPO, ψ::CanonicalMPS; center=length(ψ)) -> ExpectationCache
updateleft!(env, site); updateright!(env, site)
recalculate!(env, ψ, center)
increase_bond!(env; D)          # 扩键 + canonicalize! + 重建(DMRG1 :pre 播种)

# overlap.jl —— ⟨ψA|ψB⟩ 重叠缓存(iterative 算法共用载体)
struct OverlapCache{A, B}
    A::A; B::B
    cstorage::Vector{AbstractMatrix}    # cstorage[s] = ⟨A[1:s-1]|B[1:s-1]⟩
end
environments(ψA::CanonicalMPS, ψB::CanonicalMPS) -> OverlapCache
environments(hA::AbstractMPO, hB::AbstractMPO)
bra(x) = x.A; ket(x) = x.B

# projected.jl —— 激发态投影缓存
struct ProjectedExpectationCache
    mpo::AbstractMPO; mps::CanonicalMPS
    projectors::Vector{CanonicalMPS}
    hstorage::Array{T,3}
    cstorages::Vector{<:OverlapCache}
end
environments(h::AbstractMPO, ψ, projectors::Vector{CanonicalMPS})
```

## 3.6 局域有效哈密顿量（`src/algorithms/derivatives.jl`）

```julia
ac_prime(x::MPSTensor, W::MPOTensor, hleft::Array{T,3}, hright::Array{T,3})
#   单格点: hleft·x·W·hright 收缩
c_prime(x::AbstractMatrix, hleft::Array{T,3}, hright::Array{T,3})
#   键空间(补空间投影子): hleft·x·hright —— TDVP 第二投影子用

struct CentralHeff{T}                   # 预收缩 hleft⊙W
    left::Array{T,5}; right::Array{T,3}
end
Heff(W::MPOTensor, hleft, hright) -> CentralHeff
ac_prime(x::MPSTensor, heff::CentralHeff)

# 局域本征求解直接用 KrylovKit(不自写 Lanczos):
#   KrylovKit.eigsolve(f, x0, 1, :SR; tol=..., maxiter=...)        # DMRG1
#   KrylovKit.exponentiate(f, t, x0; ishermitian=...)              # TDVP
```

## 3.7 DMRG1 基态（`src/algorithms/dmrg.jl`，仅单格点）

DMRG 型算法统一暴露 **`leftsweep!` / `rightsweep!` / `sweep!`** 三个接口，
对缓存类型（`ExpectationCache` / `ProjectedExpectationCache`）分派；收敛循环参照 TEMPO
`iterative_compute!`。

```julia
# DMRG 型算法的统一接口
leftsweep!(env, alg)  -> kvals   # 左→右, 返回逐站点损失(局域能量)
rightsweep!(env, alg) -> kvals   # 右→左
sweep!(env, alg)      -> kvals   # = leftsweep! + rightsweep!
# 收敛判据(TEMPO iterative_error_2): std(kvals)/abs(mean(kvals)) < alg.tol

ground_state(h::MPOHamiltonian; alg::DMRG1=DMRG1()) -> (E::Real, ψ::CanonicalMPS)
ground_state!(ψ::CanonicalMPS, h::AbstractMPO; alg::DMRG1) -> (all_energies, err)
#   初猜按 alg.initguess 构造: :svd(svd_mult 初始化) / :rand(随机+归一) / :pre(increase_bond! 按 alg.trunc.D 播种)
#   每站点: KrylovKit.eigsolve(x -> ac_prime(x, Heff(W, hL, hR)), ψ[s], 1, :SR)
#        → QR/LQ 移中心 → updateleft!/updateright! 增量更新环境
```

## 3.8 激发态（`src/algorithms/excited.jl`，DMRG1 + 投影，无独立算法类型）

```julia
excited_state(h::MPOHamiltonian, ψ0::CanonicalMPS...; alg::DMRG1=DMRG1()) -> (E, ψ)
excited_state!(ψ, h::AbstractMPO, projectors::Vector{CanonicalMPS}; alg::DMRG1)
# 复用同一组接口: leftsweep!/rightsweep!/sweep!(::ProjectedExpectationCache, ::DMRG1)
# 局域问题: f(x) = P·ac_prime(x, W, hL, hR), P = I - Σ|p⟩⟨p| (环境形式投影)
#           KrylovKit.eigsolve(f, ψ[s], 1, :SR)
```

## 3.9 TDVP1（`src/algorithms/tdvp.jl`，仅 1-site，复用 sweep! 接口）

```julia
abstract type TimeEvolutionAlgorithm <: MPSAlgorithm end

@with_kw struct TDVP1{S<:Number} <: TimeEvolutionAlgorithm
    stepsize::S                     # 复时间增量(直接进指数: sweep! 施加 exp(stepsize·H))
                                    #   -im*τ = 实时 τ;  -τ = 虚时 τ
    D::Int = Defaults.D
    ishermitian::Bool = true        # hermitian→Lanczos, 否则 Arnoldi
    verbosity::Int = Defaults.verbosity
end

# 与 DMRG 相同的三个接口(分派到 TDVP1):
leftsweep!(env, alg::TDVP1)    # 前半步(Strang): 位点 s: exp(+dt/2·ac_prime); 键: exp(-dt/2·c_prime)
rightsweep!(env, alg::TDVP1)   # 另半步
sweep!(env, alg::TDVP1)        # = 一个完整时间步 alg.stepsize(左扫+右扫)
#   指数化: KrylovKit.exponentiate(x -> ac_prime/c_prime, t, x0; ishermitian=alg.ishermitian)
#   stepsize=-τ(负实)即虚时演化 e^{-H·τ} —— 基态求解器备选(与 DMRG1 互验);
#   stepsize=-im*τ 即实时演化 e^{-i·H·τ};
#   时间循环由调用方重复 sweep!(env, alg) 完成(不提供 timeevo! 封装)
```

## 3.10 TDVP2（同一文件，2-site，复用 sweep! 接口）

与 TDVP1 共用缓存（`DMRGCache`＋`CanonicalMPS` / `TDVPCache`＋`CanonicalMPO`）与
`leftsweep!` / `rightsweep!` / `sweep!` 三接口，只有局域更新不同：

```julia
@kwdef struct TDVP2{S<:Number, TR<:TruncationScheme} <: MPSAlgorithm
    stepsize::S                       # 同 TDVP1: -τ 虚时; -im*τ 实时
    trunc::TR = DefaultTruncation     # 两点 SVD 的键维上限(truncdim(D) 即 D)
    ishermitian::Bool = true
    verbosity::Int = Defaults.verbosity
end

# 左扫每对 (s,s+1): Θ = st[s]·st[s+1] → exp(+dt/2·ac2_prime/TwoSiteHeff)
#   → 截断 SVD(_split_two_site, 奇异值吸收进右格点, 正交中心随之右移) → updateleft!
#   → (非末端对) 新中心格点回步 exp(-dt/2·ac_prime); 右扫镜像(奇异值进左格点)
# 键维由 pair 更新动态增长: bonddim=1 的初态即可, 无需 changebond!; 上限由 trunc 控制
# 生成元随当前态更新(环境不可冻结: pair 更新可能改变键维)
# 初态必须规范(iscanonical): 环境与局域生成元都按等距链构造;
#   vectorize(infinite_temperature_state(...)) 的位点张量是纯恒等(非等距), 需先 rightorth!
# 精度: 流形完备时一步 = 精确 propagator; 键维增长的首扫投影不完全, 留下 O(dτ) 项
#   (与 MPSKit TDVP2 逐位一致, 见 benchmark/thermalstate/{tdvp2_l10,mpskit_tdvp2}.jl)
```

```julia
# ======================================================================================
## 3.11 HadamardTDVP（`src/algorithms/timeevo/hadamardtdvp.jl`，生成元与态都是 MPS）
# ======================================================================================
# 演化方程: dz/dτ = H ∘ z —— H 与 z 都是 MPS, 乘积是逐点(Hadamard)乘积;
#   一次 sweep!(env, alg) 实现逐元素 exp(stepsize·H) .* z
#   (stepsize 约定同 TDVP1/2: -τ 虚时, -im*τ 实时)
# 参考实现: TEMPO 的 tdvpif (src/influencefunctional/tdvpif/tdvpif.jl),
#   那里 H = influence operator, z(0) = 恒等影响泛函, τ: 0→1 给出 IF = e^H
#
# HadamardTDVPCache(H::CanonicalMPS, ψ::CanonicalMPS):
#   三条链环境 ⟨ψ|H|ψ⟩, 腿序 (bra bond, H bond, ket bond) —— 与 TDVPCache 的 MPO 环境同构;
#   bra 是共轭的态, 故复用 hadamard.jl 的 _updateleft/_updateright(与 ALS ⊙ 问题同一套)
# 局域生成元:
#   格点: _reduce_hadamard_site(env.H[s], y, hL, hR) —— H[s] 与 y 共享物理指标(广播, 不收缩)
#   键(补空间): c_prime(y, hL, hR) —— 生成元的键直穿, 与 MPO 情形同一个公式
# 扫描骨架与 TDVP1 完全一致(_gauge_left/_contract_first 等), 单点不做键维截断/增长:
#   流形 = 初态的键维剖面; 要让它长起来, 按 tdvpif 的做法先零填充:
#   changebond!(ψ; D=D, noise=0)（补零 + SVD/NoTruncation 规范 → 填充方向正交归位）,
#   演化结束后再 canonicalize!(ψ; alg=Orthogonalize(SVD(), trunc, false)) 截断
#
# 两个易错约定（都来自 tdvpif）:
#   (1) 只有 bra 侧共轭 → 投影后的生成元一般不厄米, Krylov 驱动默认 Arnoldi
#       (HadamardTDVP(; ishermitian=false))
#   (2) 环境只收缩生成元的位点张量, 而流动应由它的"表示值"(value = scaling^L·∏tensors)驱动:
#       故局域生成元统一乘因子 scaling(H)^L（等价于 tdvpif 的 _absorb_scaling!(H)）;
#       态自身的 scaling 只是记账, 不进入流动。否则 tomps 造出的生成元会被当成 H/scaling^L
# 精度: 流形能容纳逐点乘积时(短链全空间 / 零填充+规范的直积态)一步 = 逐元素 exp 到舍入;
#   受限流形上实现的是投影流, 余项 O(stepsize)(stepsize 减半则减半);
#   与"对角 MPO 路线"(TDVP1 + copyphydims(H))逐位一致(1e-16)
#
# HadamardTDVP2（两站点版本, 同一文件）:
#   与 TDVP2 对应: 相邻两格点联合演化 exp(+dt/2·H_pair) → 截断 SVD(trunc 控键维上限)
#   → updateleft!/updateright! → (非末端) 新中心格点回步 exp(-dt/2·H_site)
#   两站点局域目标 _reduce_hadamard_site2: 生成元的 pair(中键缩并)与态的 pair 融并,
#     物理指标共享, 再与两端的同构三链环境缩并
#   键维由 pair 更新自行增长: bonddim=1 的直积态即可, 无需 changebond!;
#     一次 sweep 即可长到 Schmidt 界剖面, 之后流形完备 → 流精确到 1e-15
#     (增长的那一扫自身仍有 O(stepsize) 投影余项)
#   与"对角 MPO 路线"(TDVP2 + copyphydims(H))逐位一致(1e-15)
#
# 附: TDVP1/TDVP2 的生成元 scaling 修复（本小节同批改动）
#   环境只收缩生成元的位点张量, 而 canonicalize!/tomps 会把范数搬进 scaling 字段
#   (value = scaling^L·∏tensors), 于是 exp(stepsize·H) 被静默地实现成
#   exp(stepsize·H/scaling^L)。现对所有局域生成元(单点/键/pair, MPS 与密度算符两条流形)
#   乘因子 scaling(H)^L（与 tdvpif 的 _absorb_scaling!(H) 等价）;
#   普通 MPO/MPOHamiltonian 没有 scaling 字段, 因子为 1, 行为不变;
#   态自身的 scaling 只是记账, 不进入流动
```

## 实现顺序建议

1. `SVDCompression` 路线（svd_mult/svdcompress）——QR+tsvd 扫描，覆盖 80% 用途。
2. `DMRG1` ALS 框架（`iterative_compute!` 收敛循环 + mult/add/compress 三个 Cache）。
3. ExpectationCache + `ground_state!`；ProjectedExpectationCache + `excited_state!`；TDVP。

## 数值检验清单（M2–M4 完判据）

- `mult(h, ψ; alg=SVDCompression(...))` 与精确收缩（无截断时）逐元素一致；截断误差与 `distance` 上界成立。
- `DMRG1` ALS 对随机问题在 `maxiter` 内收敛（`std/mean < tol`）。
- **ground_state**：横场 Ising / Heisenberg L=10，`E0` 与解析或稠密对角化误差 < 1e-10；逐 sweep 损失单调下降。
- **激发态**：第一激发能与对角化一致；`dot(ψ0, ψ1) ≈ 0`。
- **TDVP**：小系统精确态演化保范数（1e-12），保真度与 exp 对比 < 1e-8；虚时（`stepsize = -τ`）收敛到 DMRG1 的 E0。
