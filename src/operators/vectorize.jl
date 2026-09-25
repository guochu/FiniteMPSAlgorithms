# vectorize / devectorize: operator ↔ state conversion on the vectorized space, and the
# superoperators of left/right MPO multiplication.
#
# Convention: the vectorized representation of an operator merges the physical legs
# (p_out, p_in) of every site into one physical index of dimension d_out·d_in (column
# major, p_out the fastest), turning the operator into a state on a chain of the same
# length. The superoperator of `X ↦ h·X` / `X ↦ X·h` is the induced operator on this
# space; it is again an MPO with the same bond dimensions as `h`, so an operator product
# can be evaluated as an MPO·MPS product:
#   h1·h2 = devectorize(superoperator(h1; side=:left)  * vectorize(h2))
#         = devectorize(superoperator(h2; side=:right) * vectorize(h1))

"""
	vectorize(ρ::AbstractMPO) -> CanonicalMPS

The vectorized state of the operator chain `ρ`: every site tensor `W[aL, p_out, aR,
p_in]` becomes `A[aL, (p_out, p_in), aR]` with the physical legs merged column major
(`p_out` the fastest), so the operator is represented as a state on a chain of the same
length with per-site physical dimension `d_out·d_in`. Bond dimensions and the `scaling`
of a `CanonicalMPO` are carried over; the Schmidt values are reset.
"""
function vectorize(ρ::AbstractMPO)
	L = length(ρ)
	T = scalartype(ρ)
	data = Vector{Array{T,3}}(undef, L)
	for i in 1:L
		W = ρ[i]
		# V[aL, p_out, p_in, aR], then merge (p_out, p_in) column major
		data[i] = reshape(permutedims(W, (1, 2, 4, 3)), size(W, 1), ophydim(W) * iphydim(W), size(W, 3))
	end
	ρ isa CanonicalMPO && return CanonicalMPS(data; scaling=scaling(ρ))
	return CanonicalMPS(data)
end

# block-sparse operands are expanded into the dense MPO layer
vectorize(h::MPOHamiltonian) = vectorize(MPO(tompotensors(h)))

"""
	devectorize(ψ::AbstractMPS) -> CanonicalMPO

The inverse of [`vectorize`](@ref): every site tensor `A[aL, m, aR]` is split back into
`W[aL, p_out, aR, p_in]` (the physical dimension must be a perfect square with matching
output/input dimensions). Bond dimensions and the `scaling` of a `CanonicalMPS` are
carried over; the Schmidt values are reset.
"""
function devectorize(ψ::AbstractMPS)
	L = length(ψ)
	T = scalartype(ψ)
	data = Vector{Array{T,4}}(undef, L)
	for i in 1:L
		A = ψ[i]
		d = phydim(A)
		dh = isqrt(d)
		(dh^2 == d) ||
			throw(DimensionMismatch("the physical dimension $d of site $i is not a perfect square"))
		# split m into (p_out, p_in) column major, then p_in to the last axis
		data[i] = permutedims(reshape(A, size(A, 1), dh, dh, size(A, 3)), (1, 2, 4, 3))
	end
	ψ isa CanonicalMPS && return CanonicalMPO(data; scaling=scaling(ψ))
	return CanonicalMPO(data)
end

"""
	Base.transpose(h::MPO) -> MPO
	Base.transpose(h::CanonicalMPO) -> CanonicalMPO

The operator transpose: the physical legs of every site tensor are exchanged
(`W[aL, po, aR, pin] -> Wᵗ[aL, pin, aR, po]`), which transposes the represented
operator matrix. A `CanonicalMPO` keeps its `scaling`.
"""
Base.transpose(h::MPO) = MPO([permutedims(W, (1, 4, 3, 2)) for W in h.data])
Base.transpose(h::CanonicalMPO) =
	CanonicalMPO([permutedims(W, (1, 4, 3, 2)) for W in h.data]; scaling=scaling(h))

"""
	Base.kron(a::AbstractMPO, b::AbstractMPO) -> MPO

The Kronecker product of two operator chains of equal length: site `i` of the result
carries the local Kronecker product of the site operators on the fused physical space —
from the site tensors `A[aL, po, aR, pin]` and `B[bL, q, bR, qin]` the result tensor is

	S[(aL,bL), (po,q), (aR,bR), (pin,qin)] = A[aL, po, aR, pin] · B[bL, q, bR, qin]

(the legs of `a` are the fastest on every fused index, matching the `vectorize`
convention); the bond dimensions multiply. The `scaling` of `CanonicalMPO` inputs is
NOT folded (data-level convention). The two-sided superoperators are the special cases

	superoperator(h, :left)  == kron(h, identitympo(T, iphydims(h)))
	superoperator(h, :right) == kron(identitympo(T, ophydims(h)), transpose(h))
"""
function Base.kron(a::AbstractMPO, b::AbstractMPO)
	L = length(a)
	(L == length(b)) || throw(DimensionMismatch("kron requires chains of equal length, "
		* "got $L and $(length(b))"))
	T = promote_type(scalartype(a), scalartype(b))
	data = Vector{Array{T,4}}(undef, L)
	for i in 1:L
		A, B = a[i], b[i]
		@tensor S[al, bl, po, q, ar, br, pin, qin] := A[al, po, ar, pin] * B[bl, q, br, qin]
		# merge (al, bl), (po, q), (ar, br), (pin, qin) column major (a-leg the fastest)
		data[i] = reshape(S, size(A, 1) * size(B, 1), ophydim(A) * ophydim(B),
						  size(A, 3) * size(B, 3), iphydim(A) * iphydim(B))
	end
	return MPO(data)
end

"""
	superoperator(h::MPO, side=:left) -> MPO
	superoperator(h::MPOHamiltonian, side=:left) -> MPO
	superoperator(h::CanonicalMPO, side=:left) -> CanonicalMPO

The superoperator induced by left/right multiplication with `h`: the map `X ↦ h·X`
(`side = :left`) or `X ↦ X·h` (`side = :right`) on the vectorized space, built from the
MPO Kronecker product with the identity chain,

	superoperator(h, :left)  = kron(h, I)          # h acts on the output (po) legs
	superoperator(h, :right) = kron(I, transpose(h)) # hᵀ acts on the input (pi) legs

so the result is a chain on the doubled physical space (per-site dimension `d_out·d_in`)
with the same bond dimensions as `h`. Applying it to [`vectorize(X)`](@ref) and
devectorizing the result gives the operator product (up to the compression of the
chosen evaluation route). The represented superoperator is linear in the represented
`h`: a `CanonicalMPO` input carries its `scaling` into the result's `scaling` field,
while plain `MPO` / `MPOHamiltonian` inputs have no external scale. The `side` may also
be given as the keyword `side = :left`.
"""
function superoperator(h::MPO, side::Symbol=:left)
	(side === :left || side === :right) ||
		throw(ArgumentError("side must be :left or :right, got $side"))
	if side === :left
		return kron(h, MPO(scalartype(h), iphydims(h)))
	end
	return kron(MPO(scalartype(h), ophydims(h)), transpose(h))
end
superoperator(h::CanonicalMPO, side::Symbol=:left) =
	CanonicalMPO(superoperator(MPO(h), side).data; scaling=scaling(h))
superoperator(h::MPOHamiltonian, side::Symbol=:left) =
	superoperator(MPO(tompotensors(h)), side)
superoperator(h::AbstractMPO; side::Symbol=:left) = superoperator(h, side)
