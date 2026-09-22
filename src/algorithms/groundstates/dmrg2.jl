# Two-site DMRG (DMRG2): the two-site update optimizes the two center tensors of
# sites (s, s+1) jointly against the two-site effective Hamiltonian and truncates the
# result by SVD to `alg.D`. The SVD itself moves the orthogonality center (singular
# values absorbed into the sweep direction), so no QR gauge moves are performed and the
# not-yet-swept tensors keep their canonical form.

# ---------- two-site effective Hamiltonian ----------

"""
	TwoSiteHeff{W1,W2,T}

Two-site effective Hamiltonian: the MPO tensors of sites `s` and `s+1` together with
their left and right environments.
"""
struct TwoSiteHeff{W1<:Union{MPOTensor,AbstractSparseMPOTensor},
				   W2<:Union{MPOTensor,AbstractSparseMPOTensor},T}
	W1::W1
	W2::W2
	left::Array{T,3}
	right::Array{T,3}
end

function ac2_prime(x::AbstractArray{T,4}, heff::TwoSiteHeff) where {T}
	# x[aL, p1, p2, aR]; W1[aLw, p1o, b, p1i], W2[b, p2o, aRw, p2i]
	W1, W2 = heff.W1, heff.W2
	hl, hr = heff.left, heff.right
	@tensor Wcomb[aLw, p1o, p2o, aRw, p1i, p2i] := W1[aLw, p1o, b, p1i] * W2[b, p2o, aRw, p2i]
	@tensor m1[aL, q1, q2, wR, c] := x[aL, q1, q2, r] * hr[c, wR, r]
	@tensor m2[aL, wL, p1o, p2o, c] := m1[aL, q1, q2, wR, c] * Wcomb[wL, p1o, p2o, wR, q1, q2]
	@tensor y[aL, p1o, p2o, c] := hl[aL, wL, b] * m2[b, wL, p1o, p2o, c]
	return y
end

# sparse MPO tensors: materialize the two-site operator block by block
function ac2_prime(x::AbstractArray{T,4},
				   heff::TwoSiteHeff{<:AbstractSparseMPOTensor,<:AbstractSparseMPOTensor}) where {T}
	W1, W2 = heff.W1, heff.W2
	hl, hr = heff.left, heff.right
	d1, d2 = phydim(W1), phydim(W2)
	y = zeros(T, size(hl, 1), d1, d2, size(hr, 1))
	# sum over the combined MPO bond (wL from W1, wR from W2) and the connecting bond b
	for (wL, b) in keys(W1)
		O1 = W1[wL, b]   # d1×d1 operator (or scalar)
		O1m = O1 isa Number ? O1 * I : O1
		for (b2, wR) in keys(W2)
			b2 == b || continue
			O2 = W2[b2, wR]  # d2×d2 operator (or scalar)
			O2m = O2 isa Number ? O2 * I : O2
			hl_s = hl[:, wL, :]
			hr_s = hr[:, wR, :]
			@tensor yn[aL, p1, p2, c] := hl_s[aL, bl] * x[bl, q1, q2, br] *
										  O1m[p1, q1] * O2m[p2, q2] * hr_s[c, br]
			y .+= yn
		end
	end
	return y
end

# ---------- algorithm type ----------

"""
	DMRG2(; maxiter=Defaults.maxiter, tol=Defaults.tol, D=Defaults.D, verbosity=0)

Parameters of the two-site DMRG ground-state search. At each update the two center
tensors are optimized jointly and truncated to bond `alg.D`; `alg.D` is only used
when the algorithm generates the initial ansatz itself.
"""
@kwdef struct DMRG2 <: IterativeMPSAlgorithm
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	D::Int = Defaults.D
	verbosity::Int = 0
end

# ---------- sweeps ----------

"""
	leftsweep!(env::DMRGCache, alg::DMRG2) -> kvals

Left-to-right two-site DMRG sweep: the pair (s, s+1) is optimized jointly and the SVD
absorbs the singular values into site s+1 (the new orthogonality center); the swept
tensors stay left-canonical, the not-yet-swept tensors stay right-canonical.
"""
function leftsweep!(env::DMRGCache, alg::DMRG2)
	L = length(env.ket)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		kvals[s], _ = _dmrg2_local_update!(env, s, alg; move_right=true)
		# rebuild the left environment with the updated site s
		updateleft!(env, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

"""
	rightsweep!(env::DMRGCache, alg::DMRG2) -> kvals

Right-to-left two-site DMRG sweep (symmetric to `leftsweep!`): singular values are
absorbed into the left site of each pair, keeping the already-swept tensors
right-canonical.
"""
function rightsweep!(env::DMRGCache, alg::DMRG2)
	L = length(env.ket)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		kvals[k], _ = _dmrg2_local_update!(env, s - 1, alg; move_right=false)
		k += 1
		# rebuild the right environment with the updated site s
		updateright!(env, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(env::DMRGCache, alg::DMRG2) = vcat(leftsweep!(env, alg), rightsweep!(env, alg))

function _dmrg2_local_update!(env::DMRGCache, s::Integer, alg::DMRG2; move_right::Bool=true)
	@assert s < length(env.ket)
	W1, W2 = env.H[s], env.H[s+1]
	heff = TwoSiteHeff(W1, W2, env.hstorage[s], env.hstorage[s+2])

	# initial guess: the two-site tensor formed from the current site tensors
	@tensor guess[aL, p1, p2, aR] := env.ket[s][aL, p1, b] * env.ket[s+1][b, p2, aR]

	vals, vecs, info = eigsolve(y -> ac2_prime(y, heff), guess, 1, :SR;
								ishermitian=true, tol=max(alg.tol, 1e-12),
								krylovdim=30, maxiter=200, eager=true)
	Ψ = vecs[1]
	E = real(vals[1])

	# SVD truncate the optimal two-site tensor.
	# `move_right=true`  (left sweep):  absorb singular values into the right site (s+1).
	# `move_right=false` (right sweep): absorb singular values into the left  site (s).
	u, sv, v, _ = tsvd!(Ψ, (1, 2), (3, 4); trunc=truncdim(D=alg.D))
	if move_right
		env.ket[s] = u
		sm = Diagonal(sv)
		@tensor vnew[nb, p2, aR] := sm[nb, j] * v[j, p2, aR]
		env.ket[s+1] = vnew
	else
		env.ket[s+1] = v
		sm = Diagonal(sv)
		@tensor unew[aL, p1, nb] := u[aL, p1, j] * sm[j, nb]
		env.ket[s] = unew
	end

	return E, info
end

# ---------- driver ----------

"""
	ground_state!(ψ::CanonicalMPS, h::AbstractMPO, alg::DMRG2) -> khist

Two-site DMRG ground-state search starting from the user-provided ansatz `ψ`
(modified in place; `alg.D` is ignored when an explicit ansatz is given). Returns
`khist`, the per-sweep local-energy history.
"""
function ground_state!(ψ::CanonicalMPS, h::AbstractMPO, alg::DMRG2)
	env = DMRGCache(h, ψ)
	khist = iterative_compute!(env, alg)
	setscaling!(ψ, 1.0)
	lmul!(1 / norm(ψ), ψ)
	return khist
end

"""
	ground_state(h::MPOHamiltonian, alg::DMRG2) -> (E, ψ)

Ground-state energy and state of `h` found by two-site DMRG, starting from a random MPS
of bond `alg.D`.
"""
function ground_state(h::MPOHamiltonian, alg::DMRG2)
	ψ = randommps(scalartype(h), ophydims(h); D=alg.D)
	khist = ground_state!(ψ, h, alg)
	(alg.verbosity > 0) && println("DMRG2 converged (delta = $(_iterative_delta(khist)))")
	return expectation(h, ψ), ψ
end