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
`CanonicalMPO` inputs (applied per site during the transfer contraction — no `scaling^L`
power is materialized).
"""
function LinearAlgebra.dot(hA::AbstractMPO, hB::AbstractMPO)
	(length(hA) == length(hB)) || throw(ArgumentError("dimension mismatch"))
	sA = hA isa CanonicalMPO ? scaling(hA) : 1.0
	sB = hB isa CanonicalMPO ? scaling(hB) : 1.0
	hold = l_LL(hA, hB)
	for i in 1:length(hA)
		hold = (sA * sB) * _updateleft(hold, hA[i], hB[i])
	end
	return tr(hold)
end

"""
	LinearAlgebra.norm(h::AbstractMPO)

Hilbert-Schmidt norm `sqrt(tr(h†·h))` of the operator chain, including `scaling`
of `CanonicalMPO` inputs (applied per site during the contraction).
"""
LinearAlgebra.norm(h::AbstractMPO) = sqrt(real(dot(h, h)))

"""
	tr(h::AbstractMPO)

The trace of the operator chain, including `scaling` of `CanonicalMPO` inputs
(applied per site during the contraction — no `scaling^L` power is materialized).
"""
function LinearAlgebra.tr(h::AbstractMPO)
	s = h isa CanonicalMPO ? scaling(h) : 1.0
	v = ones(scalartype(h), 1)
	for i in eachindex(h)
		v = s * _updatetraceleft(v, h[i])
	end
	return scalar(v)
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

# ---------- scaling-free (raw data) arithmetic primitives: the `scaling` of
# CanonicalMPO operands is NOT included; the dispatch wrappers below attach it ----------

"""
Raw (strict) operator product of the chain data: bond dimensions grow, no truncation,
no `scaling` folded in.
"""
function _mul_data(hA::AbstractMPO, hB::AbstractMPO)
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
Raw (strict) block-diagonal sum of the chain data: no `scaling` folded in.
"""
function _plus_data(hA::AbstractMPO, hB::AbstractMPO)
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

# fold the represented per-site scale `scaling^L` of a CanonicalMPO into its raw data
# (site 1), returning a plain MPO; the canonical gauge is not preserved
function _fold_scaling(h::CanonicalMPO)
	d = copy(h.data)
	isempty(d) || (d[1] = scaling(h)^length(h) * d[1])
	return MPO(d)
end

"""
	Base.:*(hA::CanonicalMPO, hB::CanonicalMPO) -> CanonicalMPO

Exact (strict) operator product: bond dimensions grow, no truncation. The represented
product is bilinear in the represented chains, so the result carries
`scaling(hA)·scaling(hB)`.
"""
function Base.:*(hA::CanonicalMPO, hB::CanonicalMPO)
	r = _mul_data(hA, hB)
	return CanonicalMPO(r.data; scaling=scaling(hA) * scaling(hB))
end
"""
	Base.:*(hA::CanonicalMPO, hB::AbstractMPO) -> CanonicalMPO
	Base.:*(hA::AbstractMPO, hB::CanonicalMPO) -> CanonicalMPO

Exact (strict) operator product; the `scaling` of the `CanonicalMPO` factor carries
into the result.
"""
Base.:*(hA::CanonicalMPO, hB::AbstractMPO) =
	CanonicalMPO(_mul_data(hA, hB).data; scaling=scaling(hA))
Base.:*(hA::AbstractMPO, hB::CanonicalMPO) =
	CanonicalMPO(_mul_data(hA, hB).data; scaling=scaling(hB))

"""
	Base.:*(hA::AbstractMPO, hB::AbstractMPO) -> MPO

Exact (strict) operator product: bond dimensions grow, no truncation. The external
`scaling` of `CanonicalMPO` factors is included (see the specialized methods).
"""
Base.:*(hA::AbstractMPO, hB::AbstractMPO) = _mul_data(hA, hB)

"""
	Base.:+(hA::CanonicalMPO, hB::CanonicalMPO) -> CanonicalMPO

Exact (strict) sum as a block-diagonal direct product (no truncation). The `scaling` of
both summands is folded into the data and the result has `scaling = 1` (mirroring the
`CanonicalMPS` sum).
"""
Base.:+(hA::CanonicalMPO, hB::CanonicalMPO) =
	CanonicalMPO(_plus_data(_fold_scaling(hA), _fold_scaling(hB)).data)
"""
	Base.:+(hA::CanonicalMPO, hB::AbstractMPO) -> MPO
	Base.:+(hA::AbstractMPO, hB::CanonicalMPO) -> MPO

Exact (strict) sum; the `scaling` of the `CanonicalMPO` summand is folded into the data
of the plain-MPO result.
"""
Base.:+(hA::CanonicalMPO, hB::AbstractMPO) = _plus_data(_fold_scaling(hA), hB)
Base.:+(hA::AbstractMPO, hB::CanonicalMPO) = _plus_data(hA, _fold_scaling(hB))

"""
	Base.:+(hA::AbstractMPO, hB::AbstractMPO) -> MPO

Exact (strict) sum as a block-diagonal direct product (no truncation); the `scaling` of
`CanonicalMPO` summands is included (see the specialized methods).
"""
Base.:+(hA::AbstractMPO, hB::AbstractMPO) = _plus_data(hA, hB)
Base.:-(hA::AbstractMPO, hB::AbstractMPO) = hA + (-hB)

distance(hA::AbstractMPO, hB::AbstractMPO) = _distance(hA, hB)
distance2(hA::AbstractMPO, hB::AbstractMPO) = _distance2(hA, hB)

"""
	fidelity(hA::AbstractMPO, hB::AbstractMPO) -> Real

The Hilbert-Schmidt fidelity `|tr(conj(hA)·hB)| / (‖hA‖_HS·‖hB‖_HS)` of the operator
chains. Like [`distance`](@ref) it includes the `scaling` of `CanonicalMPO` inputs, but
the absolute value discards the overall phase: two chains differing by a global phase
(or external scale) have `fidelity = 1`, while their `distance` is generically nonzero.
Computed from raw (scale-free) transfer contractions — the `scaling` factors cancel
exactly (equivalent to the direct formula with `dot`/`norm`), so it stays finite for
arbitrarily large scalings.
"""
function fidelity(hA::AbstractMPO, hB::AbstractMPO)
	(length(hA) == length(hB)) || throw(ArgumentError("dimension mismatch"))
	return abs(_dot(hA, hB)) / sqrt(_dot(hA, hA) * _dot(hB, hB))
end

"""
	infidelity(hA::AbstractMPO, hB::AbstractMPO) -> Real

The complement of [`fidelity`](@ref): `1 - fidelity`.
"""
infidelity(hA::AbstractMPO, hB::AbstractMPO) = 1 - fidelity(hA, hB)

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
# disambiguation: the scale-free Hamiltonian mixes with a CanonicalMPO through the
# scaling-aware wrappers (the result keeps the canonical factor's scale)
Base.:*(hA::MPOHamiltonian, hB::CanonicalMPO) = _dense(hA) * hB
Base.:*(hA::CanonicalMPO, hB::MPOHamiltonian) = hA * _dense(hB)

Base.:+(hA::MPOHamiltonian, hB::AbstractMPO) = _dense(hA) + hB
Base.:+(hA::AbstractMPO, hB::MPOHamiltonian) = hA + _dense(hB)
Base.:+(hA::MPOHamiltonian, hB::MPOHamiltonian) = _dense(hA) + _dense(hB)
Base.:+(hA::MPOHamiltonian, hB::CanonicalMPO) = _dense(hA) + hB
Base.:+(hA::CanonicalMPO, hB::MPOHamiltonian) = hA + _dense(hB)

LinearAlgebra.lmul!(f::Number, h::MPOHamiltonian) = (isempty(h.data) || lmul!(f, h.data[1]); h)

"""
	todense(h::AbstractMPO; order=:msb) -> Matrix

Dense matrix of the operator chain. Per site the physical pair is ordered `(p_out,
p_in)`; the `order` keyword selects the endianness of the merged row/column indices:

- `order = :msb` — **big-endian / Kronecker order** (default): site 1 is the most
  significant (slowest) index on both the bra and the ket side, i.e. the matrix reads
  like the `kron` product of the local `(p_out, p_in)` factors site by site — the
  convention of most quantum-information texts.
- `order = :lsb` — **little-endian / column-major flattening**: site 1 is the least
  significant (fastest) index on both sides, the natural reading of
  `vec(reshape(M, ds, ds))` in Julia's column-major memory layout.

The `scaling` of a `CanonicalMPO` is included as `scaling(h)^L`, applied **per site
during the contraction** (no `scaling^L` power is materialized).
"""
function todense(h::AbstractMPO; order::Symbol=:msb)
	L = length(h)
	s = h isa CanonicalMPO ? scaling(h) : 1.0
	W = h[1]
	T = s * permutedims(W, (1, 2, 4, 3))                         # (aL, PO, PI, aR)
	for i in 2:L
		W = h[i]                                                 # (b, po, c, pi)
		@tensor T2[a, po, P, pin, Q, c] := T[a, P, Q, b] * W[b, po, c, pin]
		# merge (po, P) -> PO and (pi, Q) -> PI: the new site becomes the fastest index
		T = s * reshape(T2, size(T2, 1), size(T2, 2) * size(T2, 3), size(T2, 4) * size(T2, 5), size(T2, 6))
	end
	M = T[1, :, :, 1]
	order === :msb && return M
	order === :lsb || throw(ArgumentError("order must be :msb or :lsb"))
	# reverse the site order on the bra and ket sides
	ds = ophydims(h)
	perm = (ntuple(i -> L + 1 - i, L)..., ntuple(i -> 2L + 1 - i, L)...)
	return reshape(permutedims(reshape(M, (Tuple(ds)..., Tuple(ds)...)), perm), prod(ds), prod(ds))
end

"""
	tompo(M, phydims; order=:msb, trunc=DefaultOrthTruncation) -> CanonicalMPO

MPO representation of the dense operator matrix `M` on the physical dimensions
`phydims` — the inverse of [`todense(::AbstractMPO)`](@ref). The `order` keyword
selects the endianness of the merged row/column indices, matching
[`todense(::AbstractMPO)`](@ref):

- `order = :msb` — **big-endian / Kronecker order** (default): site 1 is the most
  significant (slowest) index on both the bra and the ket side.
- `order = :lsb` — **little-endian / column-major flattening**: site 1 is the least
  significant (fastest) index on both sides.

The chain is built from consecutive two-site SVDs on the interleaved (bra, ket) index
pairs (right-canonical form, the center on the first site; the bond spectra are
recorded), truncated with `trunc::TruncationScheme` (`DefaultOrthTruncation` by
default); with a bond-cap scheme the result is exact whenever every intermediate rank
stays within the cap. Any external scale of `M` is folded into the chain `scaling`.
"""
function tompo(M::AbstractMatrix, phydims::AbstractVector{Int};
			   order::Symbol=:msb, trunc::TruncationScheme=DefaultOrthTruncation)
	L = length(phydims)
	prod(phydims) == size(M, 1) == size(M, 2) ||
		throw(DimensionMismatch("matrix size $((size(M, 1), size(M, 2))) does not match prod(phydims) = $(prod(phydims)) on both sides"))
	order in (:msb, :lsb) || throw(ArgumentError("order must be :msb or :lsb"))
	TC = eltype(M)
	R = float(real(TC))
	# the dense matrix as a (po1..poL, pi1..piL) tensor with site 1 the slowest index
	# on both sides (the big-endian flattening puts site L slowest, so the :msb branch
	# reshapes in reverse site order and permutes back; :lsb matches directly);
	# then interleave to T[p1o, p1i, p2o, p2i, ..., pLo, pLi] (site 1 slowest)
	ds = Tuple(phydims)
	perm_rev = (ntuple(i -> L + 1 - i, L)..., ntuple(i -> 2L + 1 - i, L)...)
	Mt = order === :msb ?
		permutedims(reshape(M, (reverse(ds)..., reverse(ds)...)), perm_rev) :
		reshape(M, (ds..., ds...))
	perm = ntuple(i -> isodd(i) ? (i + 1) ÷ 2 : L + i ÷ 2, 2L)
	T = permutedims(Mt, perm)
	data = Vector{Array{TC, 4}}(undef, L)
	sarr = Vector{Union{Missing, Vector{R}}}(undef, L + 1)
	sarr[1] = ones(R, 1)
	sarr[L+1] = ones(R, 1)
	r = 1
	# right-to-left consecutive two-site SVDs: site i is split off and the right factor
	# is right-orthogonal (isometry from (po, aR, pi) to the singular index), so the
	# chain comes out right-canonical with the center (u·Diagonal(s)) on the first site
	for i in L:-1:2
		d = phydims[i]
		u, sv, v, _ = tsvd(reshape(T, :, r * d * d); trunc)        # cols (po_i, pi_i, aR)
		sarr[i] = collect(sv)
		rn = length(sv)
		# W_i[aL, po, aR, pi] from the right factor's rows (aL) × cols (po, pi, aR)
		data[i] = permutedims(reshape(v, rn, d, d, r), (1, 2, 4, 3))
		T = reshape(u * Diagonal(sv), :, rn)                       # (po1, pi1, ..., pi_{i-1}, aL)
		r = rn
	end
	data[1] = reshape(permutedims(reshape(T, phydims[1], phydims[1], r), (1, 3, 2)),
		1, phydims[1], r, phydims[1])                              # [1, po, aR, pi]
	# data-normalized gauge: the represented scale lives in `scaling` as scaling^L, and
	# the Schmidt values are rescaled to the normalized data (they were raw SVD values)
	nrm = norm(data[1])
	data[1] = data[1] / nrm
	for i in 2:L
		sarr[i] = sarr[i] ./ nrm
	end
	return CanonicalMPO(data, sarr; scaling=nrm^(1 / L))
end

"""
	todense(h::MPOHamiltonian) -> Matrix

Dense matrix of the Hamiltonian MPO: the block-sparse chain is expanded with
`tompotensors` and contracted; see [`todense(::AbstractMPO)`](@ref) for the index
convention.
"""
todense(h::MPOHamiltonian) = todense(_dense(h))
