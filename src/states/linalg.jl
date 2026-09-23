# linear algebra and exact (strict) algorithms for CanonicalMPS
# NOTE the per-site scaling convention: total scaling = scaling^L

# unconventioned transfer-chain contraction (scaling not included)
function _dot(ψA::CanonicalMPS, ψB::CanonicalMPS)
	(length(ψA) == length(ψB)) || throw(ArgumentError("dimension mismatch"))
	hold = l_LL(ψA, ψB)
	for i in 1:length(ψA)
		hold = _updateleft(hold, ψA[i], ψB[i])
	end
	return tr(hold)
end

"""
	LinearAlgebra.dot(ψA, ψB)

Overlap `⟨ψA|ψB⟩`, including the per-site `scaling` factors (total = `(scalingA*scalingB)^L`).
"""
function LinearAlgebra.dot(ψA::CanonicalMPS, ψB::CanonicalMPS)
	return _dot(ψA, ψB) * (scaling(ψA) * scaling(ψB))^length(ψA)
end

function LinearAlgebra.norm(ψ::CanonicalMPS)
	a = real(_dot(ψ, ψ))
	a = (abs(a) >= 1.0e-14) ? a : zero(a)
	return sqrt(a) * scaling(ψ)^length(ψ)
end

function LinearAlgebra.lmul!(f::Number, ψ::CanonicalMPS)
	isempty(ψ.data) && return ψ
	ψ[1] *= f
	_renormalize!(ψ, ψ[1], false)
	return ψ
end

Base.:*(ψ::CanonicalMPS, f::Number) = lmul!(f, copy(ψ))
Base.:*(f::Number, ψ::CanonicalMPS) = ψ * f
Base.:/(ψ::CanonicalMPS, f::Number) = ψ * (1 / f)
Base.:-(ψ::CanonicalMPS) = (-1) * ψ

# scaling-free (raw data) MPO·MPS application: `h`'s `scaling` is NOT included
function _apply_data(h::AbstractMPO, ψ::CanonicalMPS)
	(length(h) == length(ψ)) || throw(ArgumentError("dimension mismatch"))
	T = promote_type(scalartype(h), scalartype(ψ))
	data = Vector{Array{T,3}}(undef, length(ψ))
	for i in 1:length(ψ)
		W = h[i]
		A = ψ[i]
		@tensor r[aL, po, aR, bL, bR] := W[aL, po, aR, pin] * A[bL, pin, bR]
		data[i] = tie(permute(r, (1, 4, 2, 3, 5)), (2, 1, 2))
	end
	return CanonicalMPS(data)
end

"""
	Base.:*(h::AbstractMPO, ψ::CanonicalMPS) -> CanonicalMPS

Exact (strict) application of the operator `h` to `ψ`: bond dimensions grow to `D_h·D_ψ`,
no truncation. The `scaling` of `ψ` is carried over.
"""
Base.:*(h::AbstractMPO, ψ::CanonicalMPS) =
	CanonicalMPS(_apply_data(h, ψ).data; scaling=scaling(ψ))

"""
	Base.:*(h::CanonicalMPO, ψ::CanonicalMPS) -> CanonicalMPS

Exact (strict) application of the operator `h` to `ψ`: bond dimensions grow to `D_h·D_ψ`,
no truncation. The `scaling` of both factors carries into the result (the represented
operator product is bilinear in the represented chains).
"""
Base.:*(h::CanonicalMPO, ψ::CanonicalMPS) =
	CanonicalMPS(_apply_data(h, ψ).data; scaling=scaling(h) * scaling(ψ))

"""
Base.:+(x::CanonicalMPS, y::CanonicalMPS) -> CanonicalMPS

Exact (strict) sum as a block-diagonal direct product (no truncation); the `scaling` of both
summands is folded into the data and the result has `scaling = 1`.
"""
function Base.:+(x::CanonicalMPS, y::CanonicalMPS)
	(length(x) == length(y)) || throw(DimensionMismatch())
	scaling_x = scaling(x)
	scaling_y = scaling(y)
	L = length(x)
	T = promote_type(scalartype(x), scalartype(y))
	r = Vector{Array{T,3}}(undef, L)
	r[1] = cat(scaling_x * x[1], scaling_y * y[1]; dims=3)
	r[L] = cat(scaling_x * x[L], scaling_y * y[L]; dims=1)
	for i in 2:L-1
		r[i] = cat(scaling_x * x[i], scaling_y * y[i]; dims=(1, 3))
	end
	return CanonicalMPS{T, real(T)}(r)
end
Base.:-(x::CanonicalMPS, y::CanonicalMPS) = x + (-y)

"""
	⊙(ψA::CanonicalMPS, ψB::CanonicalMPS) -> CanonicalMPS

Exact (strict) element-wise (Hadamard) product of two MPS: the amplitudes of the result
are the pointwise product `χ(i₁,…,i_L) = ψA(i₁,…,i_L)·ψB(i₁,…,i_L)`. The physical index is
shared and the bond indices are fused pairwise, so the bond dimensions multiply
(`D_χ = D_ψA·D_ψB`); no truncation or re-canonicalization is performed. The per-site
`scaling` of the result is `scaling(ψA)·scaling(ψB)`.

The compressed (finite-bond) approximation is obtained via [`hadamard`](@ref).
"""
function ⊙(ψA::CanonicalMPS, ψB::CanonicalMPS)
	(length(ψA) == length(ψB)) || throw(DimensionMismatch("lengths must match"))
	(phydims(ψA) == phydims(ψB)) ||
		throw(DimensionMismatch("physical dimensions must match"))
	T = promote_type(scalartype(ψA), scalartype(ψB))
	data = Vector{Array{T,3}}(undef, length(ψA))
	for i in 1:length(ψA)
		A = ψA[i]   # (aL, p, aR)
		B = ψB[i]   # (bL, p, bR)
		# insert singleton bond axes (no axis reorder, so plain reshape is safe);
		# broadcasting aligns the shared physical axis p at dim 3
		r = reshape(A, size(A, 1), 1, size(A, 2), size(A, 3), 1) .*
			reshape(B, 1, size(B, 1), size(B, 2), 1, size(B, 3))
		# r: (aL, bL, p, aR, bR); fuse the pairwise bond indices
		data[i] = tie(r, (2, 1, 2))
	end
	return CanonicalMPS(data; scaling=scaling(ψA) * scaling(ψB))
end

"""
	distance(ψA, ψB)
	distance2(ψA, ψB)

Distance between two states, based on overlaps and including `scaling` factors.
"""
function _distance2(ψA::CanonicalMPS, ψB::CanonicalMPS)
	sA = real(dot(ψA, ψA))
	sB = real(dot(ψB, ψB))
	c = dot(ψA, ψB)
	return abs(sA + sB - 2 * real(c))
end
_distance(ψA::CanonicalMPS, ψB::CanonicalMPS) = sqrt(_distance2(ψA, ψB))
distance(ψA::CanonicalMPS, ψB::CanonicalMPS) = _distance(ψA, ψB)

"""
	fidelity(ψA::CanonicalMPS, ψB::CanonicalMPS) -> Real

The fidelity `|⟨ψA|ψB⟩| / (‖ψA‖·‖ψB‖)` of the represented chains. Like [`distance`](@ref)
it includes the per-site `scaling` factors, but the absolute value discards the overall
phase: two chains differing by a global phase (or external scale) have `fidelity = 1`,
while their `distance` is generically nonzero. Computed from raw (scale-free) transfer
contractions — the `scaling` factors cancel exactly (equivalent to the direct formula
with `dot`/`norm`), so it stays finite for arbitrarily large scalings.
"""
function fidelity(ψA::CanonicalMPS, ψB::CanonicalMPS)
	(length(ψA) == length(ψB)) || throw(ArgumentError("dimension mismatch"))
	return abs(_dot(ψA, ψB)) / sqrt(_dot(ψA, ψA) * _dot(ψB, ψB))
end

"""
	infidelity(ψA::CanonicalMPS, ψB::CanonicalMPS) -> Real

The complement of [`fidelity`](@ref): `1 - fidelity`.
"""
infidelity(ψA::CanonicalMPS, ψB::CanonicalMPS) = 1 - fidelity(ψA, ψB)
distance2(ψA::CanonicalMPS, ψB::CanonicalMPS) = _distance2(ψA, ψB)

"""
	Base.sum(ψ::CanonicalMPS) -> Number

Sum of all amplitudes `Σ_{i₁,…,i_L} ψ(i₁,…,i_L)` of the represented state (all
physical indices contracted with all-ones vectors), including the `scaling^L` factor.
"""
function Base.sum(ψ::CanonicalMPS)
	v = ones(scalartype(ψ), 1)
	for i in length(ψ):-1:1
		A = ψ[i]
		# v_new[aL] = Σ_{p, aR} A[aL, p, aR]·v[aR]: the all-ones vector broadcast over p
		v = reshape(A, size(A, 1), :) * repeat(v; inner=size(A, 2))
	end
	return v[1] * scaling(ψ)^length(ψ)
end

"""
	copyphydims(ψ::CanonicalMPS) -> CanonicalMPO

Reinterpret the MPS `ψ` as an operator chain on the same bond dimension: every site
tensor `A[aL, p, aR]` becomes the physical-space diagonal `W[aL, po, aR, pi] =
A[aL, po, aR]·δ_{po,pi}`, i.e. the physical dimensions are copied to both the bra and
the ket side. Applying the result to another MPS with the same bond structure
reproduces the element-wise (Hadamard) product, `copyphydims(ψA) * ψB == ⊙(ψA, ψB)`.
The `scaling` of `ψ` is carried over. The MPS Schmidt values do not satisfy the MPO
canonical conditions, so the Schmidt values of the result are uninitialized (as for
any freshly constructed chain).
"""
function copyphydims(ψ::CanonicalMPS)
	T = scalartype(ψ)
	data = Vector{Array{T,4}}(undef, length(ψ))
	for i in eachindex(ψ)
		A = ψ[i]
		d = size(A, 2)
		W = zeros(T, size(A, 1), d, size(A, 3), d)
		for p in 1:d
			W[:, p, :, p] .= A[:, p, :]
		end
		data[i] = W
	end
	return CanonicalMPO(data; scaling=scaling(ψ))
end

"""
	permute!(x, perm; kwargs...) -> x

Reorder the site contents of the MPS/MPO chain `x` in place, so that afterwards site
`i` carries the former content of site `perm[i]` (matching `Base.permute!` on plain
vectors). Realized as the sequence of neighboring content swaps given by
[`permutation2swaps`](@ref); keyword arguments are forwarded to `swap!` (`trunc`).
Initializes the canonical form if needed.
"""
function Base.permute!(x::Union{CanonicalMPS, CanonicalMPO}, perm::AbstractVector{Int}; kwargs...)
	(length(perm) == length(x) && isperm(perm)) ||
		throw(ArgumentError("`perm` must be a permutation of 1:$(length(x))"))
	for b in permutation2swaps(perm)
		swap!(x, b; kwargs...)
	end
	return x
end

"""
	permute(x, perm; kwargs...) -> typeof(x)

Non-mutating [`permute!`](@ref): the site contents of `x` reordered according to
`perm`, returned as a new chain. This chain-level `permute` extends the tensor
`permute` of the tensorops layer.
"""
permute(x::Union{CanonicalMPS, CanonicalMPO}, perm::AbstractVector{Int}; kwargs...) =
	permute!(copy(x), perm; kwargs...)
