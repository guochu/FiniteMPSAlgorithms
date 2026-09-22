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
	superoperator(h::AbstractMPO; side=:left) -> MPO

The superoperator induced by left/right multiplication with the data-level operator of
`h`: the map `X ↦ h·X` (`side = :left`) or `X ↦ X·h` (`side = :right`) on the vectorized
space — an `MPO` on the doubled physical space (per-site dimension `d_out·d_in`) with the
same bond dimensions as `h`. Site tensor layouts (δ the identity on the physical legs):

- `side = :left`:  `𝒲[aL, (po,pi), aR, (q,qi)] = W[aL, po, aR, q] · δ(pi, qi)`
- `side = :right`: `𝒲[aL, (po,pi), aR, (q,qi)] = δ(po, q) · W[aL, qi, aR, pi]`

Applying it to [`vectorize(X)`](@ref) and devectorizing the result gives the operator
product (up to the compression of the chosen evaluation route). The external `scaling`
of a `CanonicalMPO` input is not represented: like a plain `MPO`, the superoperator
carries the data-level scale only.
"""
function superoperator(h::AbstractMPO; side::Symbol=:left)
	(side === :left || side === :right) ||
		throw(ArgumentError("side must be :left or :right, got $side"))
	L = length(h)
	T = scalartype(h)
	data = Vector{Array{T,4}}(undef, L)
	for i in 1:L
		W = h[i]
		dl, d, dr = size(W, 1), phydim(W), size(W, 3)
		delta = Matrix{T}(I, d, d)
		if side === :left
			# S[aL, po, pi, aR, q, qi] = W[aL, po, aR, q] · δ(pi, qi)
			S = reshape(W, dl, d, 1, dr, d, 1) .* reshape(delta, 1, 1, d, 1, 1, d)
		else
			# S[aL, po, pi, aR, q, qi] = δ(po, q) · W[aL, qi, aR, pi]
			S = reshape(permutedims(W, (1, 4, 3, 2)), dl, 1, d, dr, 1, d) .*
				reshape(delta, 1, d, 1, 1, d, 1)
		end
		# merge (po, pi) and (q, qi) column major (p_out the fastest on both sides)
		data[i] = reshape(S, dl, d^2, dr, d^2)
	end
	return MPO(data)
end

# block-sparse operands are expanded into the dense MPO layer
superoperator(h::MPOHamiltonian; side::Symbol=:left) = superoperator(MPO(tompotensors(h)); side)
