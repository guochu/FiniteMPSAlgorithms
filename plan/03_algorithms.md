# 03 algorithms/：mult/add/compress、环境缓存、DMRG1、激发态、TDVP

**位置约定**：
- **精确（严格）算法**（无截断的 `*`、`+`）在 `states/linalg.jl` 与 `operators/linalg.jl`（文档 02 §2.8）；
- **截断算法**全部在 `src/algorithms/`，**只 export 唯一接口** `mult` / `add` / `compress`，
  以算法类型分派：`SVDCompression`（SVD 扫描路线）与 `DMRG1`（ALS 变分路线，定义参照 TEMPO）；
  `svd_*` / `iterative_*` 等内部实现不导出；无 stable_* 变体；
- **只实现 single-site**；局域本征求解直接用 **KrylovKit**（不自写 Lanczos）；
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
