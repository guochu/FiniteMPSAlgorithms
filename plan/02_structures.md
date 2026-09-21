# 02 数据结构与基础操作：states/（CanonicalMPS、CanonicalMPO）与 operators/（MPO、MPOHamiltonian）

`CanonicalMPS` / `CanonicalMPO` 的数据布局与 TEMPO 的 `ADT` / `ProcessTensor` **完全一致**（`data` + `s` + `scaling`）。

## 2.1 公共抽象定义（`src/abstractdefs.jl`）

```julia
const MPSTensor{T}  = Array{T,3}      # A[aL, p, aR]
const MPOTensor{T}  = Array{T,4}      # W[aL, p_out, aR, p_in]
# (无 ValidIndices; indexing 直接用 Integer/AbstractRange/Colon 的多重派发)

abstract type AbstractMPS{T<:Number} end     # 量子态链
abstract type AbstractMPO{T<:Number} end     # 算符链

# 张量级空间函数
space_l(A::MPSTensor) = size(A,1);  space_r(A::MPSTensor) = size(A,3);  phydim(A) = size(A,2)
space_l(W::MPOTensor) = size(W,1);  space_r(W::MPOTensor) = size(W,3)
phydim(W::MPOTensor) = size(W,2)              # 输出物理维; 输入维 = size(W,4)

# 链级公共接口(对 AbstractMPS/AbstractMPO 统一实现)
Base.length / getindex / setindex! / lastindex / iterate / eltype
TO.scalartype(::Type{<:AbstractMPS{T}}) / (::Type{<:AbstractMPO{T}}) = T
Base.copy / copy! / complex / show
space_l / space_r (链级)
bond_dimension(ψ, b::Int); bond_dimensions(ψ); bond_dimension(ψ)   # 第 b 键 / 全部 / 最大
# 不设 isstrict 查询: 所有构造器强制左右边界维 = 1(见 _check_mps_space)

# 边界等距阵(维数对齐用)
r_RR(ψA::AbstractMPS, ψB::AbstractMPS)              # ones(dA, dB)
l_LL(ψA, ψB)
r_RR(ψA, h::AbstractMPO, ψB)                        # 三指标边界(bra 键, W 键, ket 键)
l_LL(ψA, h, ψB)
r_RR(hA::AbstractMPO, hB::AbstractMPO); l_LL(hA, hB)
```

---

## 2.2 states/ — CanonicalMPS（`src/states/canonicalmps.jl`，对应 TEMPO ADT）

```julia
struct CanonicalMPS{T<:Number, R<:Real} <: AbstractMPS{T}
    data::Vector{Array{T,3}}                        # 位点张量链
    s::Vector{Union{Missing, Vector{R}}}            # L+1 个键的 Schmidt 值; missing=未初始化
    scaling::Ref{Float64}                           # 逐位点缩放(总缩放=scaling^L)
    # 内部构造器校验: R == real(T); length(s) == L+1; 相邻键维匹配; 左右边界维 1
end

CanonicalMPS{T,R}(data, s, scaling::Ref{R})                       # 内部构造器
CanonicalMPS(data::Vector{<:MPSTensor{T}}, s::AbstractVector; scaling::Real=1)
CanonicalMPS(data::Vector{<:MPSTensor}; scaling::Real=1)          # s 全部 missing(边界键置 ones)
CanonicalMPS(::Type{T}, ds::AbstractVector{Int})                  # 键维 1 全 1 张量
CanonicalMPS(ds::AbstractVector{Int}; T=Float64)
CanonicalMPS(::Type{T}, L::Int; d::Int=2)
```

## 2.3 states/ — CanonicalMPO（`src/states/canonicalmpo.jl`，对应 TEMPO ProcessTensor；语义：密度矩阵）

```julia
struct CanonicalMPO{T<:Number, R<:Real} <: AbstractMPO{T}
    data::Vector{Array{T,4}}                        # (aL, p_out, aR, p_in), 边界真空维 1
    s::Vector{Union{Missing, Vector{R}}}
    scaling::Ref{Float64}
end

CanonicalMPO(data::Vector{<:MPOTensor{T}}, s; scaling=1)
CanonicalMPO(data::Vector{<:MPOTensor}; scaling=1)
CanonicalMPO(::Type{T}, ds::AbstractVector{Int})    # identity: reshape(isometry(T,d), 1, d, 1, d)
CanonicalMPO(::Type{T}, L::Int; d::Int=2)
```

## 2.4 operators/ — MPO（`src/operators/mpo.jl`，对应 QuantumSpins MPO）

```julia
struct MPO{T<:Number} <: AbstractMPO{T}
    data::Vector{Array{T,4}}                        # 仅张量链, 左边界维 1
end

MPO(data::Vector{<:MPOTensor})
MPO(::Type{T}, ds::AbstractVector{Int})             # identity(键维 1)
MPO(ds::AbstractVector{Int}; T=Float64)
MPO(h::CanonicalMPO)                                # 取 data(浅拷贝链, 张量共享)
```

## 2.5 operators/ — MPOHamiltonian（`src/operators/mpohamiltonian.jl`，对应 MPSKit MPOHamiltonian）

```julia
"""
    MPOHamiltonian{T}

哈密顿量的 MPO 表示(有限态机形式 4-指标张量链), `ground_state`/`tdvp`/`timeevo` 的标准算符输入。
与 `MPO` 的区别仅在语义与未来的扩展空间(如局域项查询、对称性友好构造)。
"""
struct MPOHamiltonian{T<:Number} <: AbstractMPO{T}
    data::Vector{Array{T,4}}
end

MPOHamiltonian(data::Vector{<:MPOTensor})
MPOHamiltonian(h::MPO)
MPO(h::MPOHamiltonian)

# 哈密顿量组装辅助(无算符项层; 用 prodmpo + '+' 组合):
#   H = prodmpo(T, ds, [1,2], [σz, σz]) * J + prodmpo(T, ds, [1], [σx])  等
```

---

## 2.6 规范形（`src/states/orth.jl`、`src/operators/orth.jl`）

```julia
# 配置类型
struct Orthogonalize{A<:Union{QR,SVD}, T<:TruncationScheme}
    orth::A; trunc::T; normalize::Bool
end
Orthogonalize(orth::Union{QR,SVD}=SVD(); trunc::TruncationScheme=NoTruncation(), normalize::Bool=false)

# CanonicalMPS —— QR/LQ 无截断; SVD 路径把键谱写入 psi.s
leftorth!(ψ::CanonicalMPS; alg::Orthogonalize=Orthogonalize(QR()))    # 左→右, Q 留本位点
rightorth!(ψ::CanonicalMPS; alg=Orthogonalize(SVD(), normalize=false))# 右→左, 填 s
canonicalize!(ψ::CanonicalMPS; alg=Orthogonalize(SVD(), NoTruncation(), false))
# = 一遍 QR 左正交(无截断) + 一遍 SVD 右正交截断 → 右正则形式 + 正确 Schmidt 谱
canonicalize(ψ; kwargs...)

# CanonicalMPO —— 同名接口; SVD/QR 分组把 (p_out,p_in) 物理双指标合在一起:
#   左正交分组 (1,2)|(3,4); 右正交分组 (1)|(2,3,4) → permute(q,(1,2,4,3)) 恢复轴序
leftorth! / rightorth! / canonicalize! / canonicalize(ψ::CanonicalMPO; ...)

# MPO / MPOHamiltonian —— 同 CanonicalMPO 的分组方式(修复 QuantumSpins mpo/orth.jl 的 bug)
leftorth! / rightorth! / canonicalize!(h::MPO; ...)

# 正交性检验(张量级 + 链级)
isleftcanonical(A::MPSTensor; atol); isrightcanonical(A::MPSTensor; atol)
isleftcanonical(A::MPOTensor; atol); isrightcanonical(A::MPOTensor; atol)
isleftcanonical(ψ; atol); isrightcanonical(ψ; atol)
iscanonical(ψ::CanonicalMPS; atol)
# 链级 canonical = 全体右正交 + s 全部初始化 + 每键 Diagonal(s.^2) ≈ 左环境

# Schmidt 值便捷访问
svectors_uninitialized(ψ)      # 是否有 missing
unset_svectors!(ψ)             # 清回 missing(边界除外)
scaling(ψ); setscaling!(ψ, v)
LinearAlgebra.normalize!(ψ)    # 置 scaling=1(TEMPO 语义)
```

实现要点：MPS 的 SVD 路径用 `tsvd!(ψ[i], (1,2), (3,))` 型分组，`U` 留本位点、`Diagonal(s)*V` 乘入下一位点，
并写 `ψ.s[i+1]=s`（右扫时写 `ψ.s[i]`）。

## 2.7 转移/环境基元（`src/states/transfer.jl`、`src/operators/transfer.jl`）

全部为独立 `@tensor` 函数（无状态），是所有上层算法的原子操作：

```julia
# states/transfer.jl
updateleft(h::AbstractMatrix, Aj::MPSTensor, Bj::MPSTensor)          # ⟨conj(A)|h|B⟩
updateright(h::AbstractMatrix, Aj::MPSTensor, Bj::MPSTensor)
updateleft(h::Array{T,3}, Aj::MPSTensor, W::MPOTensor, Bj::MPSTensor) # ⟨A|W|B⟩ 环境
updateright(h::Array{T,3}, Aj::MPSTensor, W::MPOTensor, Bj::MPSTensor)

# operators/transfer.jl
updateleft(h::AbstractMatrix, WA::MPOTensor, WB::MPOTensor)           # MPO-MPO 重叠
updateright(h::AbstractMatrix, WA::MPOTensor, WB::MPOTensor)
updatetraceleft(v::Vector{T}, W::MPOTensor)                           # tr(MPO)
updatetraceright(v::Vector{T}, W::MPOTensor)
```

## 2.8 线性代数与精确算法（`src/states/linalg.jl`、`src/operators/linalg.jl`）

**精确（严格）算法**（无截断，`Base` 重载）直接定义在这里；带截断的 mult/add/compress 见 `algorithms/`（文档 03）。

```julia
# ---- 精确算法 ----
# states/linalg.jl
Base.:*(h::AbstractMPO, ψ::CanonicalMPS) -> CanonicalMPS   # MPO·MPS: 逐位点收缩, tie 物理双指标
Base.:*(h::AbstractMPO, ρ::CanonicalMPO) -> CanonicalMPO   # 作用于密度矩阵
Base.:+(ψA::CanonicalMPS, ψB::CanonicalMPS) -> CanonicalMPS  # 块对角直和: cat dims=3/1/(1,3)
Base.:-(ψA, ψB)

# operators/linalg.jl
Base.:*(hA::AbstractMPO, hB::AbstractMPO) -> MPO           # MPO·MPO: tie (2,1,2,1) 融合物理轴
Base.:+(hA::AbstractMPO, hB::AbstractMPO) -> MPO           # 块对角直和
Base.:-(hA, hB)

# ---- 线性代数 ----
# 注意 TEMPO 的逐位点 scaling 约定: 总缩放 = scaling^L
LinearAlgebra.dot(ψA::CanonicalMPS, ψB::CanonicalMPS)     # = _dot * (scalingA*scalingB)^L
LinearAlgebra.norm(ψ)
LinearAlgebra.normalize!(ψ)
LinearAlgebra.dot(hA::AbstractMPO, hB::AbstractMPO)
tr(h::AbstractMPO)                                        # updatetraceleft 链收缩
distance(a, b); distance2(a, b)

lmul!(f::Number, ψ)          # 缩放首张量, 重整 scaling(开 L 次方)
Base.:*(ψ, f::Number) / Base.:*(f, ψ) / Base.:/(ψ, f)
```

## 2.9 可观测量（`src/states/observables.jl`）

```julia
# 纯态期望值 ⟨ψA|h|ψB⟩
expectation(ψA::CanonicalMPS, h::AbstractMPO, ψB::CanonicalMPS)   # 三指标环境扫描 + scalar
expectation(h::AbstractMPO, ψ::CanonicalMPS) = expectation(ψ, h, ψ)

# 单格点算符期望(规范形快速路径, 利用 psi.s)
expectation(ψ::CanonicalMPS, A::AbstractMatrix, site::Int)

# 混合态(CanonicalMPO 表示 ρ): 交叉收缩 tr(h·ρ)
expectation(h::AbstractMPO, ρ::CanonicalMPO; normalized::Bool=true)   # normalized ? tr(hρ)/tr(ρ) : tr(hρ)

# 纠缠
entanglement_entropy(ψ::CanonicalMPS; bond::Int=div(length(ψ),2), α::Real=1)  # renyi_entropy(s.^2)
entanglement_spectrum(ψ; bond)    # s.^2
schmidt_values(ψ; bond)           # = ψ.s[bond]
```

## 2.10 初始化器（`src/states/initializers.jl`、`src/operators/initializers.jl`）

```julia
# states/initializers.jl
prodmps(::Type{T}, ds::Vector{Int}, states::Vector{Int})     # onehot 基态
prodmps(::Type{T}, vectors::Vector{<:AbstractVector})        # 逐位点振幅向量
randommps(::Type{T}, ds::Vector{Int}; D::Int) -> CanonicalMPS   # 键维轮廓 = max_bond_dimensions(ds, D)
randommps(L::Int; d::Int=2, D::Int)
randomcanonicalmpo(::Type{T}, ds; D::Int)
DensityOperator(ψ::CanonicalMPS) -> CanonicalMPO             # ρ = |ψ⟩⟨ψ|, 逐位 ψ⊗conj(ψ), 键维平方
infinite_temperature_state(::Type{T}, ds) -> CanonicalMPO    # identity/2^{L/2}
max_bond_dimensions(ds::Vector{Int}, D::Int) -> Vector{Int}
increase_bond!(ψ::CanonicalMPS; D::Int)                      # 零填充扩键

# operators/initializers.jl
identity_mpo(::Type{T}, ds::Vector{Int}) -> MPO              # identity(即原 idmpo)
prodmpo(::Type{T}, ds, positions::Vector{Int}, ops::Vector{<:AbstractMatrix})  # 键维 1 直积链(哈密顿量组装基元)
randommpo(::Type{T}, ds; D::Int)
increase_bond!(h::AbstractMPO; D::Int)
```

## 数值检验清单（M1 完成判据）

1. `canonicalize!` 后每个位点 `isrightcanonical`；`dot(ψ,ψ) ≈ 1`（乘 scaling 语义）。
2. `s` 与转移矩阵本征值一致：`Diagonal(s[b].^2) ≈ updateleft 累积环境`。
3. `expectation(ψ, h, ψ)` 与稠密矩阵版本一致（小系统，`matrix(h)` 为测试辅助函数）。
4. `tr(identity_mpo(ds)) = ∏ d_i`；`dot(hA,hB)` 对单位算符给出 `∏ d_i`。
5. `MPOHamiltonian` 与 `MPO` 互转无损；`expectation(ψ, H, ψ)` 与稠密哈密顿量本征向量直接计算一致。
