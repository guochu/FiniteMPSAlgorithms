# entanglement measures of canonical chains (the expectation-value functions live in
# algorithms/expec.jl)

# Schmidt values at `bond` shared by CanonicalMPS and CanonicalMPO (canonicalize if needed)
function _bond_schmidt(x::Union{CanonicalMPS, CanonicalMPO}, bond::Integer)
	(1 <= bond <= length(x) - 1) || throw(BoundsError())
	svectors_uninitialized(x) && canonicalize!(x)
	s = x.s[bond+1]
	ismissing(s) && throw(ArgumentError("Schmidt values at bond $bond are not initialized"))
	return s
end

"""
	entanglement_entropy(x::Union{CanonicalMPS,CanonicalMPO}; bond=div(length(x), 2), α=1)

Rényi entropy `α` of the Schmidt spectrum at `bond` (`α = 1` gives the von Neumann entropy).
Requires the canonical form (initialized Schmidt values).
"""
function entanglement_entropy(x::Union{CanonicalMPS, CanonicalMPO}; bond::Integer=div(length(x), 2), α::Real=1)
	s = _bond_schmidt(x, bond)
	return renyi_entropy(s .^ 2; α)
end

"""
	entanglement_spectrum(x::Union{CanonicalMPS,CanonicalMPO}; bond=div(length(x), 2))

The Schmidt spectrum (squared Schmidt values) at `bond`.
"""
function entanglement_spectrum(x::Union{CanonicalMPS,CanonicalMPO}; bond::Integer=div(length(x), 2))
	s = _bond_schmidt(x, bond)
	return abs2.(s)
end

"""
	schmidt_values(x::Union{CanonicalMPS,CanonicalMPO}; bond=div(length(x), 2))

The Schmidt values at `bond` (equivalent to `x.s[bond+1]`).
"""
function schmidt_values(x::Union{CanonicalMPS,CanonicalMPO}; bond::Integer=div(length(x), 2))
	return _bond_schmidt(x, bond)
end

"""
	todense(ψ::CanonicalMPS; order=:msb) -> Vector

Dense state vector of the MPS. The `order` keyword selects the endianness of the merged
physical index:

- `order = :msb` — **big-endian / Kronecker order** (default): site 1 is the most
  significant (slowest) index and site `L` the fastest, i.e. the vector reads like
  `ψ = v₁ ⊗ v₂ ⊗ ⋯ ⊗ v_L`, the Kronecker convention of `kron`-ing the local basis
  vectors site by site (the convention of most quantum-information texts).
- `order = :lsb` — **little-endian / column-major flattening**: site 1 is the least
  significant (fastest) index — the natural reading of `vec(reshape(v, ds...))` in
  Julia's column-major memory layout, which is also what a left-to-right TT-SVD
  sweep produces natively.

The chain `scaling` is included as `scaling(ψ)^L`, applied **per site during the
contraction** (no `scaling^L` power is materialized); the Schmidt vectors are not used
(the site tensors carry the full state).
"""
function todense(ψ::CanonicalMPS; order::Symbol=:msb)
	L = length(ψ)
	A1 = ψ[1]
	T = scaling(ψ) * reshape(A1, size(A1, 2), size(A1, 3))          # (p1, a)
	for i in 2:L
		A = ψ[i]
		t3 = reshape(T, :, size(A, 1)) * reshape(A, size(A, 1), :) # (r, p·b)
		# append p_i as the fastest row dimension: rows stay (p1,...,p_i), p1 slowest
		T = reshape(permutedims(reshape(t3, size(T, 1), size(A, 2), size(A, 3)), (2, 1, 3)),
			size(A, 2) * size(T, 1), size(A, 3))
		T = scaling(ψ) * T
	end
	v = vec(T)
	order === :msb && return v
	order === :lsb || throw(ArgumentError("order must be :msb or :lsb"))
	# reverse the site order: site 1 becomes the fastest index
	vt = reshape(v, reverse(phydims(ψ))...)
	return vec(permutedims(vt, reverse(ntuple(i -> i, L))))
end

"""
	tomps(v, phydims; order=:msb, trunc=DefaultOrthTruncation) -> CanonicalMPS

Tensor-train (MPS) representation of the dense state vector `v` on the physical
dimensions `phydims`. The `order` keyword selects the endianness of the merged
physical index, matching [`todense`](@ref):

- `order = :msb` — **big-endian / Kronecker order** (default): site 1 is the most
  significant (slowest) index, i.e. the vector reads like the `kron` product of the
  local basis amplitudes site by site.
- `order = :lsb` — **little-endian / column-major flattening**: site 1 is the least
  significant (fastest) index, the natural reading of `vec(reshape(v, ds...))` in
  Julia's column-major memory layout.

The chain is built from consecutive right-to-left SVDs (right-canonical form, the
center on the first site; the bond spectra are recorded), truncated with
`trunc::TruncationScheme` (`DefaultOrthTruncation` by default); with a bond-cap scheme
the result is exact whenever every intermediate rank stays within the cap. Any
external norm of `v` is folded into the chain `scaling`.
"""
function tomps(v::AbstractVector, phydims::AbstractVector{Int}; order::Symbol=:msb,
			   trunc::TruncationScheme=DefaultOrthTruncation)
	L = length(phydims)
	prod(phydims) == length(v) ||
		throw(DimensionMismatch("vector length $(length(v)) does not match prod(phydims) = $(prod(phydims))"))
	order in (:msb, :lsb) || throw(ArgumentError("order must be :msb or :lsb"))
	# the dense amplitudes as an (p1, ..., pL) tensor with site 1 the slowest index:
	# in the :msb (big-endian) flattening site L is the fastest index, so the reshape
	# runs in reverse site order and is permuted back; :lsb matches directly
	vt = order === :msb ?
		permutedims(reshape(v, Tuple(reverse(phydims))), reverse(ntuple(i -> i, L))) :
		reshape(v, Tuple(phydims))
	TC = eltype(v)
	R = float(real(TC))
	data = Vector{Array{TC, 3}}(undef, L)
	sarr = Vector{Union{Missing, Vector{R}}}(undef, L + 1)
	sarr[1] = ones(R, 1)
	sarr[L+1] = ones(R, 1)
	r = 1
	# right-to-left consecutive SVDs: site i is split off and the right factor is
	# right-orthogonal (isometry from (p, aR) to the singular index), so the chain
	# comes out right-canonical with the center (u·Diagonal(s)) on the first site
	T = vt
	for i in L:-1:2
		d = phydims[i]
		u, sv, v, _ = tsvd(reshape(T, :, r * d); trunc)            # cols (p_i, aR)
		sarr[i] = collect(sv)
		rn = length(sv)
		data[i] = reshape(v, rn, d, r)                             # (aL, p, aR)
		T = reshape(u * Diagonal(sv), :, rn)                       # (p1..p_{i-1}, aL)
		r = rn
	end
	data[1] = reshape(T, 1, phydims[1], r)
	# data-normalized gauge: the represented norm lives in `scaling` as scaling^L, and
	# the Schmidt values are rescaled to the normalized data (they were raw SVD values)
	nrm = norm(data[1])
	data[1] = data[1] / nrm
	for i in 2:L
		sarr[i] = sarr[i] ./ nrm
	end
	return CanonicalMPS(data, sarr; scaling=nrm^(1 / L))
end
