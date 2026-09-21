# common abstract types, aliases and interface for one-dimensional tensor networks

const MPSTensor{T}  = Array{T,3}     # A[aL, p, aR]
const MPOTensor{T}  = Array{T,4}     # W[aL, p_out, aR, p_in]

abstract type AbstractMPS{T<:Number} end
abstract type AbstractMPO{T<:Number} end

# configuration of the orthogonalization scheme (see states/orth.jl and operators/orth.jl)
abstract type MatrixProductOrthogonalAlgorithm end

"""
    Orthogonalize{A<:Union{QR,SVD}, T<:TruncationScheme}

Configuration of the orthogonalization scheme, used by `leftorth!`, `rightorth!` and `canonicalize!`.

# Fields
- `orth::A`: underlying orthogonalization algorithm (`QR` or `SVD`)
- `trunc::T`: truncation scheme; only effective with `SVD`, no effect with `QR`
- `normalize::Bool`: whether to normalize (instead of distributing the norm into `scaling`)
- `verbosity::Int`: verbosity level
"""
struct Orthogonalize{A<:Union{QR,SVD}, T<:TruncationScheme} <: MatrixProductOrthogonalAlgorithm
	orth::A
	trunc::T
	normalize::Bool
	verbosity::Int
end
Orthogonalize(a::Union{QR,SVD}, trunc::TruncationScheme; normalize::Bool=false, verbosity::Int=0) =
	Orthogonalize(a, trunc, normalize, verbosity)
Orthogonalize(a::Union{QR,SVD}, trunc::TruncationScheme, normalize::Bool) =
	Orthogonalize(a, trunc, normalize, 0)
Orthogonalize(a::Union{QR,SVD}; trunc::TruncationScheme=NoTruncation(), normalize::Bool=false, verbosity::Int=0) =
	Orthogonalize(a, trunc, normalize, verbosity)
Orthogonalize(; alg::Union{QR,SVD}=SVD(), trunc::TruncationScheme=NoTruncation(), normalize::Bool=false, verbosity::Int=0) =
	Orthogonalize(alg, trunc, normalize, verbosity)

# scalartype dispatch: chains expose their scalar element type through `scalartype(x)`.
# Instances fall back to the type method, and site tensors are Arrays covered by the
# AbstractArray implementation, so no further overloads are needed.
scalartype(::Type{<:AbstractMPS{T}}) where {T} = T
scalartype(::Type{<:AbstractMPO{T}}) where {T} = T

# tensor-level space queries
space_l(A::MPSTensor) = size(A, 1)
space_r(A::MPSTensor) = size(A, 3)
phydim(A::MPSTensor)  = size(A, 2)                 # physical dimension
space_l(W::MPOTensor) = size(W, 1)
space_r(W::MPOTensor) = size(W, 3)
ophydim(W::MPOTensor) = size(W, 2)                 # output physical dimension
iphydim(W::MPOTensor) = size(W, 4)                 # input physical dimension
"""
    phydim(W::MPOTensor)

The physical dimension of an `MPOTensor`; for a square operator this is the common value of
the output (`ophydim`) and input (`iphydim`) dimensions, otherwise a `DimensionMismatch` is thrown.
"""
function phydim(W::MPOTensor)
	d = ophydim(W)
	(d == iphydim(W)) ||
		throw(DimensionMismatch("output physical dimension $d does not match input physical dimension $(iphydim(W))"))
	return d
end

isometry(::Type{T}, m::Int, n::Int, p::Int) where {T<:Number} = reshape(isometry(T, m*n, p), m, n, p)
isometry(::Type{T}, m::Int, n::Int, p::Int, q::Int) where {T<:Number} = reshape(isometry(T, m*n*p, q), m, n, p, q)

# chain-level interface
Base.length(x::Union{AbstractMPS,AbstractMPO}) = length(x.data)
Base.getindex(x::Union{AbstractMPS,AbstractMPO}, i::Integer) = getindex(x.data, i)
Base.getindex(x::Union{AbstractMPS,AbstractMPO}, r::AbstractUnitRange) = getindex(x.data, r)
Base.setindex!(x::Union{AbstractMPS,AbstractMPO}, v, i::Integer) = setindex!(x.data, v, i)
Base.lastindex(x::Union{AbstractMPS,AbstractMPO}) = length(x.data)
Base.firstindex(x::Union{AbstractMPS,AbstractMPO}) = 1
Base.iterate(x::Union{AbstractMPS,AbstractMPO}, state...) = iterate(x.data, state...)
Base.keys(x::Union{AbstractMPS,AbstractMPO}) = 1:length(x)
Base.size(x::Union{AbstractMPS,AbstractMPO}) = (length(x),)

_copy_s(x) = [ismissing(s) ? s : copy(s) for s in x.s]

# chain-level space queries
space_l(ψ::Union{AbstractMPS,AbstractMPO}) = space_l(ψ[1])
space_r(ψ::Union{AbstractMPS,AbstractMPO}) = space_r(ψ[end])
phydims(ψ::AbstractMPS) = [phydim(ψ[i]) for i in eachindex(ψ)]
ophydims(h::AbstractMPO) = [ophydim(h[i]) for i in eachindex(h)]
iphydims(h::AbstractMPO) = [iphydim(h[i]) for i in eachindex(h)]

"""
    bonddim(ψ, b)
    bonddims(ψ)
    bonddim(ψ)

Bond dimension at bond `b` (bond `b` lies between site `b` and `b+1`), the vector of all bond
dimensions, and the maximal bond dimension.
"""
bonddim(ψ::Union{AbstractMPS,AbstractMPO}, b::Integer) = space_r(ψ[b])
bonddims(ψ::Union{AbstractMPS,AbstractMPO}) = [bonddim(ψ, b) for b in 1:length(ψ)-1]
function bonddim(ψ::Union{AbstractMPS,AbstractMPO})
	length(ψ) <= 1 && return 1
	return maximum(bonddims(ψ))
end

# boundary isometries (dimension alignment helpers; boundaries always have dimension 1)
r_RR(ψA::AbstractMPS, ψB::AbstractMPS) = ones(scalartype(ψA), space_r(ψA), space_r(ψB))
l_LL(ψA::AbstractMPS, ψB::AbstractMPS) = ones(scalartype(ψA), space_l(ψA), space_l(ψB))
r_RR(ψ::AbstractMPS) = r_RR(ψ, ψ)
l_LL(ψ::AbstractMPS) = l_LL(ψ, ψ)
r_RR(hA::AbstractMPO, hB::AbstractMPO) = ones(scalartype(hA), space_r(hA), space_r(hB))
l_LL(hA::AbstractMPO, hB::AbstractMPO) = ones(scalartype(hA), space_l(hA), space_l(hB))
function r_RR(ψA::AbstractMPS, h::AbstractMPO, ψB::AbstractMPS)
	T = promote_type(scalartype(ψA), scalartype(h), scalartype(ψB))
	return ones(T, space_r(ψA), space_r(h), space_r(ψB))
end
function l_LL(ψA::AbstractMPS, h::AbstractMPO, ψB::AbstractMPS)
	T = promote_type(scalartype(ψA), scalartype(h), scalartype(ψB))
	return ones(T, space_l(ψA), space_l(h), space_l(ψB))
end

# plain MPO / MPOHamiltonian carry no scaling field: their scale always lives in the data.
# The `scaling` accessors and in-place rescaling (`setscaling!`, `normalize!`) are
# CanonicalMPS/CanonicalMPO-only (defined in states/orth.jl); the helpers below stay
# hasproperty-guarded so the algorithms can run them on plain-MPO working copies.
function _rescaling!(x, n::Real)
	hasproperty(x, :scaling) || return x
	setscaling!(x, scaling(x) * n^(1/length(x)))
	return x
end

# fold the norm of `r` into the per-site scaling and normalize `r` in place;
# returns the norm before normalization. If `normalize`, scaling is left untouched.
# For chains without a `scaling` field the norm stays in the data (r is not normalized).
function _renormalize!(x, r, normalize::Bool)
	nr = norm(r)
	if nr != zero(nr)
		if normalize
			lmul!(1/nr, r)
		elseif hasproperty(x, :scaling)
			_rescaling!(x, nr)
			lmul!(1/nr, r)
		end
	end
	return nr
end

function _renormalize_coeff!(x, normalize::Bool)
	normalize && hasproperty(x, :scaling) && setscaling!(x, 1.0)
	return x
end

# orthogonality checks of single site tensors
function isleftcanonical(A::MPSTensor; atol::Real=1e-12)
	@tensor ok[a, b] := conj(A[c, p, a]) * A[c, p, b]
	return isapprox(ok, isometry(scalartype(A), size(ok, 1), size(ok, 2)); atol)
end
function isrightcanonical(A::MPSTensor; atol::Real=1e-12)
	@tensor ok[a, b] := conj(A[a, p, c]) * A[b, p, c]
	return isapprox(ok, isometry(scalartype(A), size(ok, 1), size(ok, 2)); atol)
end
# MPO tensors are gauged with the physical double index (p_out, p_in) fused on one side:
# left-canonical  -> isometry (aL, p_out, p_in) -> aR   (leftorth!  grouping (1, 2, 4) | (3,))
# right-canonical -> isometry (p_out, aR, p_in) -> aL   (rightorth! grouping (1,) | (2, 3, 4))
function isleftcanonical(W::MPOTensor; atol::Real=1e-12)
	@tensor ok[c, d] := conj(W[a, p, c, q]) * W[a, p, d, q]
	return isapprox(ok, isometry(scalartype(W), size(ok, 1), size(ok, 2)); atol)
end
function isrightcanonical(W::MPOTensor; atol::Real=1e-12)
	@tensor ok[a, b] := conj(W[a, p, c, q]) * W[b, p, c, q]
	return isapprox(ok, isometry(scalartype(W), size(ok, 1), size(ok, 2)); atol)
end
