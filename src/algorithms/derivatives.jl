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

# Schur (Jordan upper-triangular) MPO site tensors: act on the block structure directly,
# without densifying. The two implied identity corners (1,1) and (m,n) act trivially on
# the vacuum/closing channels; the C/D/B blocks contribute one group contraction each
# and the interior A blocks one (the array storage keeps the structurally-zero
# lower-triangular A entries, which simply contribute nothing). The right ket bond is
# folded once — exactly as in the dense path — and no `(l, wL, p2, c)` intermediate of
# the full logical shape is ever materialized.
function ac_prime(x::MPSTensor, W::SchurMPOTensor,
				  hleft::AbstractArray{T,3}, hright::AbstractArray{T,3}) where {T}
	# fold the right ket bond into the right environment (large bond × small MPO bond)
	@tensor m1[l, p, wR, c] := x[l, p, r] * hright[c, wR, r]
	Ty = promote_type(T, scalartype(x), scalartype(W))
	y = zeros(Ty, size(hleft, 1), phydim(W), size(hright, 1))
	m, n = space_l(W), space_r(W)
	# channel-sliced views: the Schur channel offsets (interior channel j of the right
	# environment is slot j+1 of m1 / slot j of the C·A arrays, and slot i+1 of hleft /
	# slot i of the B·A arrays) are aligned before contracting; the vacuum/closing
	# channels are sliced out as well (literal indices would be free labels in @tensor)
	m1vac = @view m1[:, :, 1, :]            # the vacuum channel of the right environment
	m1clo = @view m1[:, :, n, :]            # the closing channel of the right environment
	m1int = @view m1[:, :, 2:n-1, :]        # the interior channels of the right environment
	hlvac = @view hleft[:, 1, :]            # the vacuum channel of the left environment
	hlclo = @view hleft[:, m, :]            # the closing channel of the left environment
	hl = @view hleft[:, 2:m-1, :]           # the interior channels of the left environment
	# the implied identity corners (1,1) and (m,n): the vacuum/closing channels pass x
	@tensor y[a, p2, c] += hlvac[a, l] * m1vac[l, p2, c]
	@tensor y[a, p2, c] += hlclo[a, l] * m1clo[l, p2, c]
	# first row: the C blocks (vacuum -> interior channels) and the D corner
	(n > 2) && @tensor y[a, p2, c] += hlvac[a, l] * m1int[l, p, j, c] * W.C[p2, j, p]
	@tensor y[a, p2, c] += hlvac[a, l] * m1clo[l, p, c] * W.D[p2, p]
	# last column: the B blocks (interior channels -> closing)
	(m > 2) && @tensor y[a, p2, c] += hl[a, i, l] * m1clo[l, p, c] * W.B[i, p2, p]
	# interior: the A blocks (upper-triangular channel transitions)
	((m > 2) && (n > 2)) &&
		@tensor y[a, p2, c] += hl[a, i, l] * m1int[l, p, j, c] * W.A[i, p2, j, p]
	return y
end

"""
	c_prime(x, hleft, hright)

Bond-space effective Hamiltonian action (the second TDVP1 projector):
`y[a, c] = Σ hleft[a, w, b] x[b, e] hright[c, w, e]`.
"""
function c_prime(x::AbstractMatrix, hleft::AbstractArray{T,3}, hright::AbstractArray{T,3}) where {T}
	# two explicit steps: fold `x` into the right environment first, then the left one.
	# Both steps are BLAS-shaped; letting the nary `@tensor` pick the order contracts the
	# two environments through their (small) middle legs instead, which materializes a
	# χ_L×χ_R×χ_L'×χ_R' intermediate — measured ~400× slower at χ=50, w=4.
	@tensor tmp[x1, h, o] := x[x1, k] * hright[o, h, k]
	@tensor y[-1, -2] := hleft[-1, h, x1] * tmp[x1, h, -2]
	return y
end

"""
	CentralHeff

Single-site effective Hamiltonian of DMRG1: the MPO site tensor `W` (dense or Schur
form, stored as-is) with its environments (`left`/`right` are NOT pre-contracted with
`W`; the MPS bond dimensions are the large ones, so `ac_prime` contracts them first).
A Schur site tensor is applied block-wise (see `ac_prime`), preserving the sparsity of
the upper-triangular channel structure.
"""
struct CentralHeff{W<:Union{MPOTensor,SchurMPOTensor},T}
	W::W
	left::Array{T,3}
	right::Array{T,3}
end

"""
	Heff(W, hleft, hright) -> CentralHeff

Bundle the MPO site tensor with its environments (no contraction, no conversion): the
site tensor keeps its own form (dense or Schur), and the repeated `eigsolve` actions
dispatch on it.
"""
Heff(W::Union{MPOTensor,SchurMPOTensor}, hleft::AbstractArray{T,3},
	 hright::AbstractArray{T,3}) where {T} = CentralHeff(W, hleft, hright)

ac_prime(x::MPSTensor, heff::CentralHeff) = ac_prime(x, heff.W, heff.left, heff.right)
