# TEBD: quantum gates (AbstractGate / UnitaryGate / GeneralGate) + apply! + swap!
# Minimal TEBD building blocks; the time-evolution loop is driven by the caller.

"""
	AbstractGate{N,T}

Abstract supertype of quantum gates acting on `N` sites: `positions(g)::NTuple{N,Int}`
(ascending), `g.op::Array{T,M}` — the rank-`M = 2N` operator tensor in the index
convention `(i1', i2', …, iN', i1, …, iN)`, i.e. all bra (output) indices first and all
ket (input) indices second, each block ordered with site 1 the slowest index.
"""
abstract type AbstractGate{N, T} end

positions(g::AbstractGate) = g.positions
scalartype(::Type{<:AbstractGate{N, T}}) where {N, T} = T

"""
	shift(g, by)

Shift all support positions of gate `g` by `by` sites.
"""
shift(g::G, by::Integer) where {G<:AbstractGate} =
	typeof(g)(ntuple(i -> g.positions[i] + by, Val(length(g.positions))), g.op)

function Base.adjoint(g::G) where {G<:AbstractGate}
	N = length(g.positions)
	# U† : swap the ket/bra index blocks and conjugate; support positions unchanged
	perm = (ntuple(i -> N + i, Val(N))..., ntuple(i -> i, Val(N))...)
	return typeof(g)(g.positions, permutedims(conj(g.op), perm))
end

"""
	UnitaryGate(positions, op; atol=1e-10)

A unitary gate acting on the `N` ascending sites `positions`. The operator `op` is a
rank-`2N` tensor `(i1', i2', …, iN', i1, …, iN)` — all bra (output) indices first, all
ket (input) indices second, each block ordered with site 1 the slowest index.

The input is copied and materialized as a dense array of concrete element type, then
checked for unitarity (`op'*op ≈ I` within `atol`), throwing `ArgumentError` otherwise.
"""
struct UnitaryGate{N, T, M} <: AbstractGate{N, T}
	positions::NTuple{N, Int}
	op::Array{T, M}   # (i1', …, iN', i1, …, iN) with M == 2N (checked at construction)

	function UnitaryGate{N, T, M}(positions::NTuple{N, Int}, op::Array{<:Any, M}; atol::Real=1.0e-10) where {N, T, M}
		M == 2N || throw(ArgumentError("the operator tensor must have rank $(2N)"))
		issorted(collect(positions)) || throw(ArgumentError("positions must be ascending"))
		op = convert(Array{T, M}, op)
		m = tie(op, (N, N))
		isunitary(m; atol) || throw(ArgumentError("the gate operator is not unitary"))
		return new{N, T, M}(positions, op)
	end
end

function UnitaryGate(positions::NTuple{N, Int}, op::AbstractArray; atol::Real=1.0e-10) where {N}
	M = ndims(op)
	M == 2N || throw(ArgumentError("the operator tensor must have rank $(2N)"))
	T = scalartype(op)
	return UnitaryGate{N, T, M}(positions, Array{T, M}(op); atol)
end

"""
	GeneralGate(positions, op)

A general gate with the same data storage, index conventions and application as
[`UnitaryGate`](@ref) (rank-`2N` tensor `(i1', …, iN', i1, …, iN)`), but the input is
**not** checked for unitarity. The input is copied and materialized as a dense array of
concrete element type. Since a non-unitary gate does not preserve the canonical form
under the Hastings update, `apply!` re-canonicalizes the state with `canonicalize!`
afterwards.
"""
struct GeneralGate{N, T, M} <: AbstractGate{N, T}
	positions::NTuple{N, Int}
	op::Array{T, M}   # (i1', …, iN', i1, …, iN) with M == 2N (checked at construction)

	function GeneralGate{N, T, M}(positions::NTuple{N, Int}, op::Array{<:Any, M}) where {N, T, M}
		M == 2N || throw(ArgumentError("the operator tensor must have rank $(2N)"))
		issorted(collect(positions)) || throw(ArgumentError("positions must be ascending"))
		return new{N, T, M}(positions, convert(Array{T, M}, op))
	end
end

function GeneralGate(positions::NTuple{N, Int}, op::AbstractArray) where {N}
	M = ndims(op)
	M == 2N || throw(ArgumentError("the operator tensor must have rank $(2N)"))
	T = scalartype(op)
	return GeneralGate{N, T, M}(positions, Array{T, M}(op))
end

# nearest-neighbor Hastings gate core (shared by UnitaryGate and GeneralGate): the
# Schmidt spectrum `psi.s[i]` on the left bond is contracted into the post-gate
# two-site tensor, which is then SVD-truncated. The physical state is preserved
# exactly for any gate; only the canonical form is guaranteed for unitary gates with
# small truncation error.
function _nn_gate_apply!(g::AbstractGate{2}, ψ::CanonicalMPS, i::Integer; trunc::TruncationScheme)
	(1 <= i <= length(ψ) - 1) || throw(BoundsError())
	svectors_uninitialized(ψ) && canonicalize!(ψ)
	A = ψ[i]
	B = ψ[i+1]
	G = g.op
	# unweighted two-site block
	@tensor block[a, p, q, b] := A[a, p, c] * B[c, q, b]
	# post-gate block
	@tensor gated[a, p′, q′, b] := block[a, p, q, b] * G[p′, q′, p, q]
	# Hastings: fold the Schmidt spectrum on the left bond into the block, then SVD
	sv = Diagonal(ψ.s[i])
	@tensor weighted[a, p′, q′, b] := sv[a, 1] * gated[1, p′, q′, b]
	u, s, v, err = tsvd!(weighted, (1, 2), (3, 4); trunc)
	ψ[i+1] = v
	ψ.s[i+1] = s
	# back-project the unweighted post-gate block; the left site stays right-canonical
	@tensor Anew[a, p′, c] := gated[a, p′, q′, b] * conj(v[c, q′, b])
	ψ[i] = Anew
	return ψ
end

# Hastings content swap of the neighboring sites `b`, `b+1`: the two-site block is
# formed with the physical legs crossed (a SWAP gate), so the site contents exchange
# while the canonical form is preserved (the SWAP gate is unitary).
function _swap_content!(ψ::CanonicalMPS, b::Integer; trunc::TruncationScheme)
	(1 <= b <= length(ψ) - 1) || throw(BoundsError())
	svectors_uninitialized(ψ) && canonicalize!(ψ)
	A = ψ[b]
	B = ψ[b+1]
	# two-site block with the physical legs crossed: the leg at site b carries the
	# physical content of B, the leg at site b+1 that of A
	@tensor block[a, q, p, c] := A[a, p, b] * B[b, q, c]
	# Hastings: fold the Schmidt spectrum on the left bond into the block, then SVD
	sv = Diagonal(ψ.s[b])
	@tensor weighted[a, q, p, c] := sv[a, 1] * block[1, q, p, c]
	u, s, v, err = tsvd!(weighted, (1, 2), (3, 4); trunc)
	ψ[b+1] = v
	ψ.s[b+1] = s
	# back-project the crossed block; the left site keeps the (crossed) physical leg of B
	@tensor Anew[a, q, c] := block[a, q, p, c'] * conj(v[c, p, c'])
	ψ[b] = Anew
	return ψ
end

"""
	apply!(g::UnitaryGate{2}, ψ::CanonicalMPS; trunc=DefaultTruncation) -> ψ

Apply a two-site unitary gate to `ψ` with the Hastings update (aligned with
TEMPO/GTEMPO). The two gate sites may be any ascending pair `(i, j)`: non-adjacent
gates are first moved next to each other with exact unitary content swaps (`swap!`),
applied at the neighboring bond, and moved back. Initializes the canonical form if
needed.

**Mathematics of the Hastings update.** In the right-canonical gauge the left
environment at bond `i` is `diag(psi.s[i]²)`. With `A = psi[i]`, `B = psi[i+1]` and the
post-gate two-site tensor `t[a, p', q', b] = Σ_{p,q,c} G[p', q', p, q] A[a, p, c] B[c,
q, b]`, the two-site wavefunction block in the left Schmidt basis is the *folded*
tensor `W = diag(psi.s[i]) · t`. Decompose it,

	W = u · diag(σ) · v,		u'·u = I,	v·v' = I,

(with truncation, `σ` keeps the leading singular values and `err` is the discarded
weight), and assign

	psi[i+1] = v,	psi.s[i+1] = σ,		psi[i] = t · v†		(v† = conj(v)).

Three properties make this exact up to the discarded weights, without ever inverting a
spectrum:

* *state*: `psi[i] · psi[i+1] = t · (v†·v) = t`, because `v†·v` is the projector onto
  the row space of `W`, and the rows of `t` lie in it;
* *gauge*: for unitary `G`, `psi[i]` is right-orthogonal — summing `conj(psi[i])·psi[i]`
  over `(p', b)` gives `t†·t = block†·(G†·G)·block = block†·block = I`, since
  `block = A·B` is built from right-orthogonal factors;
* *spectrum*: the new left environment after site `i` is
  `v · (t† · diag(psi.s[i]²) · t) · v† = v · (W†·W) · v† = diag(σ²)`.

If the input is right-canonical and the truncation error is small, the right-canonical
form is preserved without any `canonicalize!`.
"""
function apply!(g::UnitaryGate{2}, ψ::CanonicalMPS; trunc::TruncationScheme=DefaultTruncation)
	i, j = g.positions
	(1 <= i < j <= length(ψ)) || throw(BoundsError())
	for b in j-1:-1:i+1
		swap!(ψ, b; trunc)
	end
	_nn_gate_apply!(g, ψ, i; trunc)
	for b in i+1:j-1
		swap!(ψ, b; trunc)
	end
	return ψ
end

"""
	apply!(g::GeneralGate{2}, ψ::CanonicalMPS; trunc=DefaultTruncation) -> ψ

Apply a possibly non-unitary two-site gate with the same Hastings update as
[`UnitaryGate`](@ref) (identical fold–SVD–back-project step; the gate sites may be any
ascending pair, moved next to each other with content swaps). A non-unitary `G` breaks
the right-orthogonality of the left factor (the *gauge* property of the Hastings
update), so the state is re-canonicalized with `canonicalize!` afterwards. Initializes
the canonical form if needed.
"""
function apply!(g::GeneralGate{2}, ψ::CanonicalMPS; trunc::TruncationScheme=DefaultTruncation)
	i, j = g.positions
	(1 <= i < j <= length(ψ)) || throw(BoundsError())
	for b in j-1:-1:i+1
		swap!(ψ, b; trunc)
	end
	_nn_gate_apply!(g, ψ, i; trunc)
	for b in i+1:j-1
		swap!(ψ, b; trunc)
	end
	canonicalize!(ψ)
	return ψ
end

"""
	swap!(ψ::CanonicalMPS, i::Integer; trunc=DefaultTruncation) -> ψ

Exchange the physical content of the neighboring sites `i` and `i+1` of `ψ` using the
Hastings SWAP gate (aligned with TEMPO/GTEMPO): the two-site block is formed with the
physical legs crossed, the Schmidt spectrum `psi.s[i]` on the left bond is contracted
into it, and the crossed block is re-decomposed with truncation. The right factor is
written directly to `psi[i+1]`, the renewed spectrum to `psi.s[i+1]`, and the left
site is recovered by back-projecting the crossed block. If the input is
right-canonical and the truncation error is small, the right-canonical form is
preserved without any `canonicalize!` (the SWAP gate is unitary). Sequences of
`swap!` (`permutation2swaps`) generate arbitrary permutations of the site contents.
Initializes the canonical form if needed.
"""
function swap!(ψ::CanonicalMPS, i::Integer; trunc::TruncationScheme=DefaultTruncation)
	(1 <= i <= length(ψ) - 1) || throw(BoundsError())
	return _swap_content!(ψ, i; trunc)
end

"""
	swap!(ρ::CanonicalMPO, i::Integer; trunc=DefaultTruncation) -> ρ

Exchange the physical content `(p_out, p_in)` of the neighboring sites `i` and `i+1`
of the operator chain `ρ` — the operator-space analog of the MPS [`swap!`](@ref)
(Hastings update, aligned with TEMPO/GTEMPO): the two-site block is formed with the
physical leg pairs crossed, the Schmidt spectrum `rho.s[i]` on the left bond is
contracted into it, and the crossed block is re-decomposed with truncation. The right
factor is written directly to `rho[i+1]`, the renewed spectrum to `rho.s[i+1]`, and
the left site is recovered by back-projecting the crossed block. If the input is
right-canonical and the truncation error is small, the right-canonical form is
preserved without any `canonicalize!`. Sequences of `swap!` (`permutation2swaps`)
generate arbitrary permutations of the site contents. Initializes the canonical form
if needed.
"""
function swap!(ρ::CanonicalMPO, i::Integer; trunc::TruncationScheme=DefaultTruncation)
	(1 <= i <= length(ρ) - 1) || throw(BoundsError())
	svectors_uninitialized(ρ) && canonicalize!(ρ)
	W = ρ[i]
	V = ρ[i+1]
	# two-site block with the physical leg pairs crossed: the legs at site i carry the
	# (p_out, p_in) content of V, the legs at site i+1 that of W
	@tensor block[a, qo, qi, po, pin, b] := W[a, po, m, pin] * V[m, qo, b, qi]
	# Hastings: fold the Schmidt spectrum on the left bond into the block, then SVD
	sv = Diagonal(ρ.s[i])
	@tensor weighted[a, qo, qi, po, pin, b] := sv[a, 1] * block[1, qo, qi, po, pin, b]
	u, s, v, err = tsvd!(weighted, (1, 2, 3), (4, 5, 6); trunc)
	ρ[i+1] = permutedims(v, (1, 2, 4, 3))          # (aL, po, aR, pin)
	ρ.s[i+1] = s
	# back-project the crossed block; the left site keeps the (crossed) legs of V
	@tensor Wnew[a, qo, m, qi] := block[a, qo, qi, po, pin, b] * conj(v[m, po, pin, b])
	ρ[i] = Wnew
	return ρ
end
