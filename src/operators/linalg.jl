# linear algebra and exact (strict) algorithms for operator chains

# transfer-chain contraction of two operator chains (scaling not included)
function _dot(hA::AbstractMPO, hB::AbstractMPO)
	(length(hA) == length(hB)) || throw(ArgumentError("dimension mismatch"))
	hold = l_LL(hA, hB)
	for i in 1:length(hA)
		hold = _updateleft(hold, hA[i], hB[i])
	end
	return tr(hold)
end

"""
	LinearAlgebra.dot(hA, hB)

Overlap of two operator chains `Σ tr(conj(hA)·hB)` (per site), including `scaling` of
`CanonicalMPO` inputs.
"""
function LinearAlgebra.dot(hA::AbstractMPO, hB::AbstractMPO)
	_d = _dot(hA, hB)
	if hA isa CanonicalMPO || hB isa CanonicalMPO
		sA = hA isa CanonicalMPO ? scaling(hA)^length(hA) : 1.0
		sB = hB isa CanonicalMPO ? scaling(hB)^length(hB) : 1.0
		_d *= sA * sB
	end
	return _d
end

"""
	LinearAlgebra.norm(h::AbstractMPO)

Hilbert-Schmidt norm `sqrt(tr(h†·h))` of the operator chain, including `scaling`
of `CanonicalMPO` inputs.
"""
LinearAlgebra.norm(h::AbstractMPO) = sqrt(real(dot(h, h)))

"""
	tr(h::AbstractMPO)

The trace of the operator chain, including `scaling` of `CanonicalMPO` inputs.
"""
function LinearAlgebra.tr(h::AbstractMPO)
	v = ones(scalartype(h), 1)
	for i in eachindex(h)
		v = _updatetraceleft(v, h[i])
	end
	_t = scalar(v)
	h isa CanonicalMPO && (_t *= scaling(h)^length(h))
	return _t
end

LinearAlgebra.lmul!(f::Number, h::AbstractMPO) = (isempty(h.data) || (h[1] *= f); h)

# for a canonical chain the factor is folded into site 1 and its norm pushed into the
# `scaling` field, so the canonical data gauge is preserved (site 1 stays right-isometric)
function LinearAlgebra.lmul!(f::Number, ρ::CanonicalMPO)
	isempty(ρ.data) && return ρ
	ρ[1] *= f
	_renormalize!(ρ, ρ[1], false)
	return ρ
end
Base.:*(h::AbstractMPO, f::Number) = lmul!(f, copy(h))
Base.:*(f::Number, h::AbstractMPO) = h * f
Base.:/(h::AbstractMPO, f::Number) = h * (1 / f)
Base.:-(h::AbstractMPO) = (-1) * h

"""
	Base.:*(hA::AbstractMPO, hB::AbstractMPO) -> MPO

Exact (strict) operator product: bond dimensions grow, no truncation.
"""
function Base.:*(hA::AbstractMPO, hB::AbstractMPO)
	(length(hA) == length(hB)) || throw(ArgumentError("dimension mismatch"))
	T = promote_type(scalartype(hA), scalartype(hB))
	data = Vector{Array{T,4}}(undef, length(hA))
	for i in 1:length(hA)
		WA = hA[i]
		WB = hB[i]
		# (WA·WB)[(aA,aB), poA, (aA2,aB2), pinB] = Σ_pinA WA[aA,poA,aA2,pinA]·WB[aB,pinA,aB2,pinB]
		@tensor t[aA, poA, aA2, aB, aB2, q] := WA[aA, poA, aA2, p] * WB[aB, p, aB2, q]
		tp = permute(t, (1, 4, 2, 3, 5, 6))
		data[i] = tie(tp, (2, 1, 2, 1))   # ((aA·aB), poA, (aA2·aB2), pinB)
	end
	return MPO(data)
end

"""
	Base.:+(hA::AbstractMPO, hB::AbstractMPO) -> MPO

Exact (strict) sum as a block-diagonal direct product (no truncation).
"""
function Base.:+(hA::AbstractMPO, hB::AbstractMPO)
	(length(hA) == length(hB)) || throw(DimensionMismatch())
	L = length(hA)
	T = promote_type(scalartype(hA), scalartype(hB))
	r = Vector{Array{T,4}}(undef, L)
	r[1] = cat(hA[1], hB[1]; dims=3)
	r[L] = cat(hA[L], hB[L]; dims=1)
	for i in 2:L-1
		r[i] = cat(hA[i], hB[i]; dims=(1, 3))
	end
	return MPO(r)
end
Base.:-(hA::AbstractMPO, hB::AbstractMPO) = hA + (-hB)

distance(hA::AbstractMPO, hB::AbstractMPO) = _distance(hA, hB)
distance2(hA::AbstractMPO, hB::AbstractMPO) = _distance2(hA, hB)

# exact block-native application MPOHamiltonian · MPS: sum over non-zero blocks, with the
# finite-chain boundary slicing (first site keeps the start row, last keeps the closing col)
function Base.:*(h::MPOHamiltonian, ψ::CanonicalMPS)
	(length(h) == length(ψ)) || throw(ArgumentError("dimension mismatch"))
	L = length(ψ)
	T = promote_type(scalartype(h), scalartype(ψ))
	data = Vector{Array{T,3}}(undef, L)
	r0, cL = _leftrow(h), _rightcol(h)
	for i in 1:L
		W = h[i]
		A = ψ[i]
		rows = i == 1 ? (r0:r0) : 1:size(W, 1)
		cols = i == L ? (cL:cL) : 1:size(W, 2)
		r = zeros(T, length(rows), phydim(W), length(cols), size(A, 1), size(A, 3))
		for (ia, wL) in enumerate(rows), (ib, wR) in enumerate(cols)
			contains(W, wL, wR) || continue
			O = W[wL, wR]
			@tensor tn[p, a, b] := O[p, q] * A[a, q, b]
			r[ia, :, ib, :, :] .+= tn
		end
		data[i] = tie(permute(r, (1, 4, 2, 3, 5)), (2, 1, 2))
	end
	return CanonicalMPS(data; scaling=scaling(ψ))
end

# ---------- MPOHamiltonian operands: expand the block chain into the dense MPO layer
# (exact products/sums/orth are written on 4-index site tensors; block-native paths are
# the DMRG/TDVP environments and the MPO·MPS exact product) ----------

_dense(h::MPOHamiltonian) = MPO(tompotensors(h))

_dot(hA::MPOHamiltonian, hB::AbstractMPO) = _dot(_dense(hA), hB)
_dot(hA::AbstractMPO, hB::MPOHamiltonian) = _dot(hA, _dense(hB))
_dot(hA::MPOHamiltonian, hB::MPOHamiltonian) = _dot(_dense(hA), _dense(hB))

LinearAlgebra.tr(h::MPOHamiltonian) = tr(_dense(h))

Base.:*(hA::MPOHamiltonian, hB::AbstractMPO) = _dense(hA) * hB
Base.:*(hA::AbstractMPO, hB::MPOHamiltonian) = hA * _dense(hB)
Base.:*(hA::MPOHamiltonian, hB::MPOHamiltonian) = _dense(hA) * _dense(hB)

Base.:+(hA::MPOHamiltonian, hB::AbstractMPO) = _dense(hA) + hB
Base.:+(hA::AbstractMPO, hB::MPOHamiltonian) = hA + _dense(hB)
Base.:+(hA::MPOHamiltonian, hB::MPOHamiltonian) = _dense(hA) + _dense(hB)

LinearAlgebra.lmul!(f::Number, h::MPOHamiltonian) = (isempty(h.data) || lmul!(f, h.data[1]); h)

"""
	todense(h::AbstractMPO) -> Matrix

Dense matrix of the operator chain. Per site the physical pair is ordered `(p_out,
p_in)`; the row index is `(p_out,1, …, p_out,L)` and the column index `(p_in,1, …,
p_in,L)` with **site 1 the slowest** index on both sides (the Kronecker convention of
`kron`-ing the local `(p_out, p_in)` factors site by site). The `scaling` of a
`CanonicalMPO` is included as `scaling(h)^L`.
"""
function todense(h::AbstractMPO)
	L = length(h)
	W = h[1]
	T = permutedims(W, (1, 2, 4, 3))                             # (aL, PO, PI, aR)
	for i in 2:L
		W = h[i]                                                 # (b, po, c, pi)
		@tensor T2[a, po, P, pi, Q, c] := T[a, P, Q, b] * W[b, po, c, pi]
		# merge (po, P) -> PO and (pi, Q) -> PI: the new site becomes the fastest index
		T = reshape(T2, size(T2, 1), size(T2, 2) * size(T2, 3), size(T2, 4) * size(T2, 5), size(T2, 6))
	end
	s = h isa CanonicalMPO ? scaling(h)^L : 1.0
	return T[1, :, :, 1] * s
end

"""
	todense(h::MPOHamiltonian) -> Matrix

Dense matrix of the Hamiltonian MPO: the block-sparse chain is expanded with
`tompotensors` and contracted; see [`todense(::AbstractMPO)`](@ref) for the index
convention.
"""
todense(h::MPOHamiltonian) = todense(_dense(h))
