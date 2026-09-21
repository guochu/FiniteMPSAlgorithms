# local effective Hamiltonians

"""
	ac_prime(x, W, hleft, hright)

Single-site effective Hamiltonian action: `y = hleft · x · W · hright`, where `hleft/hright`
are the (bra, MPO, ket) environments at both sides of the site. The ket (MPS) bonds —
the large dimensions — are contracted first, against the small physical legs and MPO
bonds of `W`; the MPO bonds are folded in afterwards through the environments.
"""
function ac_prime(x::MPSTensor, W::MPOTensor, hleft::AbstractArray{T,3}, hright::AbstractArray{T,3}) where {T}
	# fold the right ket bond into the right environment (large bond × small MPO bond)
	@tensor m1[l, p, wR, c] := x[l, p, r] * hright[c, wR, r]
	# contract the physical legs with the MPO tensor (small dimensions)
	@tensor m2[l, wL, p2, c] := m1[l, p, wR, c] * W[wL, p2, wR, p]
	# fold the left ket bond through the left environment
	@tensor y[-1, -2, -3] := hleft[-1, wL, l] * m2[l, wL, -2, -3]
	return y
end

# block-sparse MPO site tensor: sum the local action over non-zero operator blocks
function ac_prime(x::MPSTensor, W::AbstractSparseMPOTensor,
				  hleft::AbstractArray{T,3}, hright::AbstractArray{T,3}) where {T}
	y = zeros(T, size(hleft, 1), phydim(W), size(hright, 1))
	for (wL, wR) in keys(W)
		O = W[wL, wR]
		hl = hleft[:, wL, :]
		hr = hright[:, wR, :]
		@tensor yn[a, p, c] := hl[a, bL] * x[bL, q, bR] * O[p, q] * hr[c, bR]
		y .+= yn
	end
	return y
end

"""
	c_prime(x, hleft, hright)

Bond-space effective Hamiltonian action (the second TDVP1 projector):
`y[a, c] = Σ hleft[a, w, b] x[b, e] hright[c, w, e]`.
"""
function c_prime(x::AbstractMatrix, hleft::AbstractArray{T,3}, hright::AbstractArray{T,3}) where {T}
	@tensor y[-1, -2] := hleft[-1, 1, 2] * x[2, 3] * hright[-2, 1, 3]
	return y
end

"""
	CentralHeff

Single-site effective Hamiltonian of DMRG1: the MPO site tensor `W` with its
environments stored as-is (`left`/`right` are NOT pre-contracted with `W`; the MPS
bond dimensions are the large ones, so `ac_prime` contracts them first).
"""
struct CentralHeff{W<:Union{MPOTensor,AbstractSparseMPOTensor},T}
	W::W
	left::Array{T,3}
	right::Array{T,3}
end

"""
	Heff(W, hleft, hright) -> CentralHeff

Bundle the MPO site tensor with its environments (no contraction).
"""
Heff(W::Union{MPOTensor,AbstractSparseMPOTensor}, hleft::AbstractArray{T,3}, hright::AbstractArray{T,3}) where {T} =
	CentralHeff(W, hleft, hright)

ac_prime(x::MPSTensor, heff::CentralHeff) = ac_prime(x, heff.W, heff.left, heff.right)
