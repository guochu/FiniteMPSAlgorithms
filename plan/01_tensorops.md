# 01 底层张量操作层（tensorops）

来源：[TEMPO/src/tensorops](file:///home/guochu/Documents/Missile/TEMPO/src/tensorops) 四个文件**原样 vendor** 到 `src/tensorops/`，
仅做两处调整：(1) 收缩场景改由本包内 `using TensorOperations`（TEMPO 中为 `const TO = TensorOperations`）；
(2) 把私有的 `_truncate!` 另行导出为公开 `truncate!` 供网络层调用。
**不需要 extensions.jl**（无 texp、零空间基等扩展；`_eye` 一律用已有的 `isometry`）。矩阵分解后端为 `MatrixAlgebraKit`。

## 1.1 `truncation.jl` — 截断方案

```julia
abstract type TruncationScheme end

struct NoTruncation <: TruncationScheme end                    # 保留全部奇异值
struct TruncateDim <: TruncationScheme; D::Int; end            # 只保留 D 个
struct TruncateRelError <: TruncationScheme; ϵ::Float64; end   # 相对误差截断
struct TruncateDimCutoff <: TruncationScheme                   # cutoff 定截断点, 上限 D, 至少 add_back 个
    D::Int; ϵ::Float64; add_back::Int
end

truncdim(d::Int) / truncdim(; D::Int)
truncrelerr(ϵ::Real) / truncrelerr(; ϵ::Real)
truncdimcutoff(D::Int, ϵ::Real; add_back::Int=0) / truncdimcutoff(; D, ϵ, add_back)

# 核心（私有, 奇异值必须降序）：
_truncate!(v::AbstractVector{<:Real}, trunc::TruncationScheme, p::Real=2) -> (v′, err)
# 公开包装：
truncate!(v, trunc; p=2) = _truncate!(v, trunc, p)
```

要点：`TruncateRelError`/`TruncateDimCutoff` 返回**相对误差** `err/‖s‖`；`add_back` 保证数值上至少保留若干奇异值。

## 1.2 `matrixalgebra.jl` — 矩阵正交分解（MatrixAlgebraKit 后端）

```julia
abstract type FactorizationAlgorithm end
abstract type OrthogonalFactorizationAlgorithm <: FactorizationAlgorithm end
struct QR;  struct QRpos;  struct LQ;  struct LQpos;  struct SVD;  struct SDD;  struct Polar
# adjoint 对应：QRpos↔LQpos, QR↔LQ; SVD/SDD/Polar 自逆
const OFA = OrthogonalFactorizationAlgorithm

leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos,SVD,SDD,Polar}, atol=0) -> (Q, R)
rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos,SVD,SDD,Polar}, atol=0) -> (L, Q)
# SDD() = SafeDivideAndConquer（失败自动回退 gesvd!）; SVD() = QRIteration (LAPACK gesvd)
# 注意：输入 A 被就地覆写
```

## 1.3 `tensorfactorizations.jl` — 张量分解与工具

```julia
scalar(x::AbstractArray)                     # only(x)
permute(m::AbstractArray, perm)              # PermutedDimsArray 视图（不拷贝）
permute(m::AbstractArray, left, right)       # = permute(m, (left..., right...))
random_hermitian(::Type{T}, n) ; random_unitary(::Type{T}, n)
isometry(::Type{T}, m, n) / isometry(m, n) / isometry(d)   # 矩形单位阵(原 _eye 的角色)

tie(a::AbstractArray{T,N}, axs::NTuple{N1,Int})            # 相邻轴 reshape 分组融合
Base.kron(a::AbstractArray{Ta,N}, b::AbstractArray{Tb,N})  # 逐轴 Kronecker 积（N 阶推广）
permutation2swaps(perm)                      # 置换 → 相邻对换序列

# 截断 SVD（核心）
tsvd!(a::StridedMatrix; trunc=NoTruncation(), alg::Union{SVD,SDD}=SDD()) -> (u, s, v, err)
tsvd(a::AbstractMatrix; kwargs...)                         # 拷贝后调用 tsvd!
tsvd!(a::AbstractArray{T,N}, left::NTuple, right::NTuple; trunc=NoTruncation(), alg=SDD())
tsvd(a::AbstractArray, left, right; kwargs...)
# 张量版 ! 仅复用内部工作区拷贝，不修改输入；返回 u:(left..., md), s, v:(md, right...)
# 注: TensorOperations 不提供 permute/scalar(仅有 tensorscalar/tensorcopy), 故保留本实现

# 张量正交分解
leftorth!(A, left, right; alg::Union{QR,QRpos,SVD,SDD,Polar}=QRpos(), atol=0) -> (u:(left...,s), v:(s,right...))
rightorth!(A, left, right; alg::Union{LQ,LQpos,SVD,SDD,Polar}=LQpos(), atol=0)
leftorth(A; kwargs...) / rightorth(A; kwargs...)           # 非原地版本
leftorth!(A::StridedMatrix; alg=QRpos(), atol=0)           # 矩阵便捷版
rightorth!(A::StridedMatrix; alg=LQpos(), atol=0)

renyi_entropy(v::AbstractVector{<:Real}; α::Real=1)        # α=1 即 von Neumann/Shannon
```

MPO 张量规范形标准分组（全包统一约定）：

```julia
# 左正交分组 (1,2)|(3,4); 右正交分组 (1)|(2,3,4) 之后 permute(q,(1,2,4,3)) 恢复 (aL,p_out,aR,p_in)
```

## 1.4 `distance.jl`

```julia
distance2(x::AbstractArray, y::AbstractArray)   # |‖x‖²+‖y‖²−2Re⟨x,y⟩|
distance(x, y)
```

## 1.5 包级默认值（`src/defaults.jl`）

```julia
module Defaults
const D         = 64        # 默认最大键维
const tolgauge  = 1e-14     # 规范形/构造用截断精度
const tol       = 1e-12     # 算法收敛精度
const maxiter   = 100
const verbosity = 1
end

const DefaultTruncation = truncdimcutoff(D=Defaults.D, ϵ=Defaults.tolgauge; add_back=0)
```

## 单元测试要点

- `tsvd!` 与 `LinearAlgebra.svd` 一致性；截断误差单调性；`add_back` 生效。
- `leftorth!/rightorth!` 输出正交性 `Q'Q=I`；`A ≈ Q*R`。
- `tie/kron/permute/permutation2swaps` 往返一致性。
- `truncate!` 各方案在降序奇异值向量上的截断点。
