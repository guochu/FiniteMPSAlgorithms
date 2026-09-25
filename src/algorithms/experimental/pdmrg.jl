# Positive DMRG (p-DMRG): variational ground state of the free energy
#   F(ρ) = tr(H ρ) - T S(ρ),    ρ = exp(-β H) / Z,   β = 1/T
# for a positive matrix product ansatz (PMPA). The PMPA represents a mixed state as
#   ρ = V ρ̃ V†,
# with V an isometry built from left-canonical tensors A_l (l<c) and right-canonical
# tensors B_l (l>c), and the orthogonal center M[aL,p,aR,τ] of rank R, so that
#   ρ̃ = Σ_τ M[:,:,:,τ] ⊗ conj(M[:,:,:,τ])
# is positive semidefinite by construction. At each sweep site the local free energy
#   F̃(ρ̃) = tr(H_eff ρ̃) - T S(ρ̃)
# is solved exactly: ρ̃ = exp(-β H_eff) / Z, then M is recovered from its eigen-
# decomposition keeping the R largest eigenvalues. The center is moved by an SVD that
# absorbs the neighbouring canonical tensor.
#
# Boundary stability: at the chain edges the local space aL·p·aR is far smaller than R,
# so a single-site H_eff has rank < R and cannot carry the rank-R purification center
# (the entropy is truncated and the variational optimum overcools). The sweeps therefore
# merge the center with neighbouring boundary sites until the fused local space exceeds
# R (`_merged_left_move!` / `_merged_right_move!`), thermalize the fused center exactly,
# and split it back with center-movement SVDs.
#
# Reference: C. Guo, "Density Matrix Renormalization Group Algorithm For Mixed Quantum
# States", Phys. Rev. B 105, 195152 (2022) (arXiv:2203.16350).

# ---------- algorithm type ----------

"""
	PDMRG(; maxiter=Defaults.maxiter, tol=Defaults.tol, trunc=DefaultTruncation, R=20, verbosity=0)

Parameters of the positive DMRG thermal-state search. `trunc` is the truncation scheme
applied to the center-movement SVDs (its bond cap bounds the MPS bond dimension of the
isometric part of the PMPA) and `R` the rank of the orthogonal center (the maximal
Schmidt number of the represented mixed state).
"""
@kwdef struct PDMRG{TR<:TruncationScheme} <: TwoSiteUpdate
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	trunc::TR = DefaultTruncation
	R::Int = 20
	verbosity::Int = 0
end

# ---------- Positive Matrix Product Ansatz ----------

"""
	PositiveMPA{T}

Positive matrix product ansatz for a mixed quantum state. `data` stores the left-
canonical tensors `A_l` (sites `l < center`) and right-canonical tensors `B_l`
(sites `l > center`); `mcenter` is the rank-4 center tensor `M[aL, p, aR, τ]` with
`τ` the purification index of size `R`. The represented density operator is
`ρ = Σ_τ V·M[:,:,:,τ] ⊗ conj(M[:,:,:,τ])·V†` with `V` the isometry of `data`.
The mutable state (the center position and its tensor) lives in `Ref` fields, so the
ansatz itself is immutable: sweeps update `rho.center[]` / `rho.mcenter[]` in place.
"""
struct PositiveMPA{T<:Number}
	data::Vector{MPSTensor{T}}
	mcenter::Base.RefValue{Array{T,4}}
	center::Base.RefValue{Int}
	PositiveMPA(data::Vector{MPSTensor{T}}, mcenter::Array{T,4}, center::Integer) where {T<:Number} =
		new{T}(data, Ref(mcenter), Ref(center))
end

Base.length(x::PositiveMPA) = length(x.data)
Base.getindex(x::PositiveMPA, i::Integer) = (i == x.center[]) ? x.mcenter[] : x.data[i]
scalartype(::PositiveMPA{T}) where {T} = T
bonddims(x::PositiveMPA) = [size(x[i], 3) for i in 1:length(x)-1]
space_l(x::PositiveMPA) = size(x[1], 1)
space_r(x::PositiveMPA) = size(x[length(x)], 3)
phydim(x::PositiveMPA, i::Integer) = size(x[i], 2)

"""
	randompmpa(T, ds; D, R) -> PositiveMPA

Random PMPA of physical dimensions `ds`, bond cap `D` and purification rank `R`, in
right-canonical form with center at site 1.
"""
function randompmpa(::Type{T}, ds::Vector{Int}; D::Int, R::Int) where {T<:Number}
	mps = randommps(T, ds; D=D)
	rightorth!(mps)
	center = 1
	mcenter = randn(T, size(mps[center])..., R)
	return PositiveMPA(mps.data, mcenter, center)
end

"""
	tr(x::PositiveMPA) -> Real

Trace of the represented density operator (1 for a properly normalized state).
"""
function LinearAlgebra.tr(x::PositiveMPA)
	m = x.mcenter[]
	@tensor tmp[a1,p1,c1,a2,p2,c2] := m[a1,p1,c1,t] * conj(m[a2,p2,c2,t])
	N = size(tmp, 1) * size(tmp, 2) * size(tmp, 3)
	return tr(reshape(tmp, N, N))
end

# ---------- environment cache ----------

"""
	ThermalDMRGCache{H,V,T}

Environment stack for p-DMRG: `H` the MPO, `rho` the PMPA, `hstorage` the
`⟨A/B | H | A/B⟩` environments built from the isometric part of the PMPA.
"""
struct ThermalDMRGCache{H<:AbstractMPO,V<:PositiveMPA,T}
	H::H
	rho::V
	hstorage::Vector{Array{T,3}}
end

function ThermalDMRGCache(h::AbstractMPO, rho::PositiveMPA)
	L = length(rho)
	T = promote_type(scalartype(h), scalartype(rho))
	hs = Vector{Array{T,3}}(undef, L + 1)
	hs[1] = _pmpa_l_LL(rho, h)
	hs[L+1] = _pmpa_r_RR(rho, h)
	for s in L:-1:2
		hs[s] = _updateright(hs[s+1], rho[s], h[s], rho[s])
	end
	return ThermalDMRGCache(h, rho, hs)
end

# left/right boundary environments for the PMPA (mirror the MPS ones)
function _pmpa_l_LL(rho::PositiveMPA, h::AbstractMPO)
	T = promote_type(scalartype(rho), scalartype(h))
	return ones(T, space_l(rho), space_l(h), space_l(rho))
end
function _pmpa_r_RR(rho::PositiveMPA, h::AbstractMPO)
	T = promote_type(scalartype(rho), scalartype(h))
	return ones(T, space_r(rho), space_r(h), space_r(rho))
end
function _pmpa_l_LL(rho::PositiveMPA, h::MPOHamiltonian)
	T = promote_type(scalartype(rho), scalartype(h))
	v = zeros(T, space_l(rho), space_l(h), space_l(rho))
	v[:, _leftrow(h), :] .= one(T)
	return v
end
function _pmpa_r_RR(rho::PositiveMPA, h::MPOHamiltonian)
	T = promote_type(scalartype(rho), scalartype(h))
	v = zeros(T, space_r(rho), space_r(h), space_r(rho))
	v[:, _rightcol(h), :] .= one(T)
	return v
end

function updateleft!(env::ThermalDMRGCache, site::Integer)
	env.hstorage[site+1] = _updateleft(env.hstorage[site], env.rho[site], env.H[site], env.rho[site])
	return env
end
function updateright!(env::ThermalDMRGCache, site::Integer)
	env.hstorage[site] = _updateright(env.hstorage[site+1], env.rho[site], env.H[site], env.rho[site])
	return env
end

# ---------- local thermal-state solver ----------

"""
	thermalize_center(env, site, β, alg) -> (Mnew, free_energy)

Solve the local free-energy minimization at the orthogonal center `site`:
`ρ̃ = exp(-β H_eff) / Z`, then recover the center tensor `M` from the eigen-decomposition
of `ρ̃` keeping the `alg.R` largest eigenvalues. Returns the new center and the local
free energy.
"""
function thermalize_center(env::ThermalDMRGCache, site::Integer, β::Real, alg::PDMRG)
	rho = env.rho
	@assert rho.center[] == site
	dims = size(rho.mcenter[])[1:3]
	x0 = env.rho.mcenter[][:, :, :, 1]
	return _thermalize(env.H[site], env.hstorage[site], env.hstorage[site+1], dims, β, alg, x0)
end

# dispatch between the dense and Krylov local solvers
function _thermalize(W, hl, hr, dims::NTuple{3,Int}, β::Real, alg::PDMRG, x0)
	L = prod(dims)
	if L <= 4 * alg.R
		return _thermalize_dense(W, hl, hr, dims, β, alg)
	else
		return _thermalize_krylov(W, hl, hr, dims, β, alg, x0)
	end
end

# dense path: build the full effective Hamiltonian and exponentiate
function _thermalize_dense(W, hl, hr, dims::NTuple{3,Int}, β::Real, alg::PDMRG)
	heff = _local_hamiltonian_matrix(W, hl, hr, dims)
	# rho = exp(-β H_eff) / Z
	rho = Matrix(exp(-β * Hermitian(heff)))
	rho ./= tr(rho)
	eigvals, eigvecs = eigen(Hermitian(rho))
	M, kept = _eigvecs_to_center(eigvals, eigvecs, alg, dims)
	return M, _local_free_energy(W, hl, hr, M, kept, β)
end

# Krylov path: at low β the thermal state is dominated by the lowest eigenstates of H_eff
function _thermalize_krylov(W, hl, hr, dims::NTuple{3,Int}, β::Real, alg::PDMRG, x0)
	L = prod(dims)
	n = min(alg.R, L)
	vals, vecs, info = eigsolve(y -> ac_prime(y, W, hl, hr), x0, n, :SR;
								ishermitian=true, tol=1e-10, krylovdim=2n+2)
	# Boltzmann weights of the kept lowest eigenstates
	eigvals = exp.(-β .* real(vals))
	eigvals ./= sum(eigvals)
	Dm = length(eigvals)
	T = eltype(x0)
	M = zeros(T, dims..., Dm)
	for i in 1:Dm
		M[:, :, :, i] = vecs[i]
	end
	M = reshape(reshape(M, L, Dm) * Diagonal(sqrt.(eigvals)), dims..., Dm)
	return M, _local_free_energy(W, hl, hr, M, eigvals, β)
end

function _local_hamiltonian_matrix(W, hl, hr, dims::NTuple{3,Int})
	# build the dense effective Hamiltonian as a matrix in the local tensor basis
	T = eltype(hl)
	L = prod(dims)
	H = zeros(T, L, L)
	for k in 1:L
		v = zeros(T, dims...)
		v[k] = 1
		H[:, k] .= vec(ac_prime(v, W, hl, hr))
	end
	return H
end

function _eigvecs_to_center(eigvals, eigvecs, alg::PDMRG, dims::NTuple{3,Int})
	# keep the alg.R largest (positive) eigenvalues, renormalized
	pos = findfirst(>(0), eigvals)
	pos === nothing && (pos = length(eigvals))
	keep = min(alg.R, length(eigvals) - pos + 1)
	idx = (length(eigvals) - keep + 1):length(eigvals)
	vals = eigvals[idx]
	vals ./= sum(vals)
	vecs = eigvecs[:, idx]
	M = reshape(vecs * Diagonal(sqrt.(vals)), dims..., keep)
	return M, vals
end

function _local_free_energy(W, hl, hr, M, eigvals, β::Real)
	# β F = β E - S, evaluated consistently on the (truncated) new center:
	# E = Σ_τ ⟨M_τ|H_eff|M_τ⟩,  S = -Σ p log p over the kept eigenvalues.
	S = -sum(p -> p > 0 ? p * log(p) : zero(p), eigvals)
	E = zero(real(eltype(eigvals)))
	for τ in 1:size(M, 4)
		x = M[:, :, :, τ]
		y = ac_prime(x, W, hl, hr)
		E += real(dot(x, y))
	end
	return β * E - S   # β F = β<E> - S(ρ̃)  (the trace of ρ̃ is 1)
end

# ---------- merged boundary updates ----------

# The rank of the local effective Hamiltonian is bounded by the local space dimension
# aL·p·aR. Near the chain boundaries that dimension is far smaller than the purification
# rank `alg.R`, so a single-site thermalization cannot carry a rank-R center (the entropy
# is truncated and the variational optimum is systematically overcooled). The fix: merge
# the center with neighbouring boundary sites until the *fused* local space exceeds
# `alg.R`, thermalize the fused center exactly, then split it back into single-site
# tensors by the standard center-movement SVDs.

# contract the MPO block H[s..s+k] into one fused rank-4 MPO tensor
function _fuse_mpo_block(H::AbstractMPO, s::Int, k::Int)
	# per-site plain conversion that keeps the FULL bond structure: the boundary
	# channel selection already lives in the environments (hstorage), so the block
	# must stay compatible with their bond dimensions
	Wp = H isa MPOHamiltonian ? [tompotensor(H[j]) for j in s:s+k] : [H[j] for j in s:s+k]
	W = Wp[1]
	for j in 2:length(Wp)
		W2 = @tensor W2[x0, q1, q2, x2, m1, m2] := W[x0, q1, x1, m1] * Wp[j][x1, q2, x2, m2]
		s0, Q1, Q2, s3, M1, M2 = size(W2)
		W = reshape(W2, s0, Q1 * Q2, s3, M1 * M2)
	end
	return W
end

# number of extra sites to merge so the fused local space carries rank alg.R
function _merge_span(env::ThermalDMRGCache, s::Int, dir::Int, alg::PDMRG)
	L = length(env.rho)
	pds = ophydims(env.H)
	dims = size(env.rho.mcenter[])[1:3]
	fused = prod(dims)
	k = 0
	while fused <= alg.R && (dir > 0 ? s + k < L : s - k > 1)
		k += 1
		fused = dir > 0 ?
			dims[1] * prod(pds[s:s+k]) * size(env.rho.data[s+k], 3) :
			size(env.rho.data[s-k], 1) * prod(pds[s-k:s]) * dims[3]
	end
	return k
end

# left-sweep move with the center merged into the block s..s+k: thermalize the fused
# center, then split it left-to-right (the center lands at s+k)
function _merged_left_move!(env::ThermalDMRGCache, s::Int, β::Real, alg::PDMRG)
	L = length(env.rho)
	pds = ophydims(env.H)
	k = _merge_span(env, s, +1, alg)

	# merge the center with the k tensors to its right, fusing the physical legs
	Θ = env.rho.mcenter[]
	for j in (s+1):(s+k)
		B = env.rho.data[j]
		Θ2 = @tensor Θ2[a, pc, pj, b, τ] := Θ[a, pc, al, τ] * B[al, pj, b]
		Θ = reshape(Θ2, size(Θ2, 1), size(Θ2, 2) * size(Θ2, 3), size(Θ2, 4), size(Θ2, 5))
	end
	aR = size(Θ, 3)

	# exact thermalization of the fused center
	W = _fuse_mpo_block(env.H, s, k)
	m, f = _thermalize(W, env.hstorage[s], env.hstorage[s+k+1],
					   (size(Θ, 1), size(Θ, 2), aR), β, alg, Θ[:, :, :, 1])

	# split left-to-right: left-canonical tensors at s..s+k-1, center at s+k
	aLcur = size(m, 1)
	cur = reshape(m, aLcur * size(m, 2), size(m, 3) * size(m, 4))
	for j in s:(s+k-1)
		cur3 = reshape(cur, aLcur, pds[j], :)
		u, sv, v, err = tsvd!(cur3, (1, 2), (3,); trunc=alg.trunc)
		env.rho.data[j] = u
		aLcur = size(u, 3)
		sm = Diagonal(sv)
		cur = reshape(sm * Matrix(v), aLcur, :)
	end
	τ = size(m, 4)
	env.rho.mcenter[] = reshape(cur, aLcur, pds[s+k], aR, τ)
	env.rho.center[] = s + k
	for j in s:(s+k-1)
		updateleft!(env, j)
	end
	return f
end

# right-sweep move with the center merged into the block s-k..s: thermalize the fused
# center, then split it right-to-left (the center lands at s-k)
function _merged_right_move!(env::ThermalDMRGCache, s::Int, β::Real, alg::PDMRG)
	pds = ophydims(env.H)
	k = _merge_span(env, s, -1, alg)

	# merge the center with the k tensors to its left, fusing the physical legs
	# (:lsb order, site s-k fastest — consistent with `_fuse_mpo_block` and the split)
	Θ = env.rho.mcenter[]
	for j in (s-1):-1:(s-k)
		A = env.rho.data[j]
		Θ2 = @tensor Θ2[a, pj, pc, b, τ] := A[a, pj, al] * Θ[al, pc, b, τ]
		Θ = reshape(Θ2, size(Θ2, 1), size(Θ2, 2) * size(Θ2, 3), size(Θ2, 4), size(Θ2, 5))
	end
	aL = size(Θ, 1)

	# exact thermalization of the fused center
	W = _fuse_mpo_block(env.H, s - k, k)
	m, f = _thermalize(W, env.hstorage[s-k], env.hstorage[s+1],
					   (aL, size(Θ, 2), size(Θ, 3)), β, alg, Θ[:, :, :, 1])

	# split right-to-left: right-canonical tensors at s-k+1..s, center at s-k
	# (:lsb order — site j is the slowest leg of the fused remainder)
	aL, P, aR, τ = size(m)
	nc = aR
	cur = m
	for j in s:-1:(s-k+1)
		pj = pds[j]
		P = div(P, pj)
		cur5 = reshape(cur, aL, P, pj, nc, τ)
		u, sv, v, err = tsvd!(cur5, (1, 2, 5), (3, 4); trunc=alg.trunc)
		env.rho.data[j] = v                    # B[nc', pj, nc] (right-canonical)
		sm = Diagonal(sv)
		# `sm` is absorbed into `u`, whose rank leg is the last one: reshape to
		# (aL, P, τ, r) first, then move τ back to the last leg
		cur = permutedims(reshape(reshape(u, aL * P * τ, :) * sm, aL, P, τ, :), (1, 2, 4, 3))
		nc = size(cur, 3)
	end
	env.rho.mcenter[] = reshape(cur, aL, pds[s-k], nc, τ)
	env.rho.center[] = s - k
	for j in s:-1:(s-k+1)
		updateright!(env, j)
	end
	return f
end

# ---------- sweeps ----------

"""
	leftsweep!(env::ThermalDMRGCache, alg::PDMRG; β) -> fvals

Left-to-right p-DMRG sweep. At each site the local thermal state is computed, then the
center is moved one site to the right by an SVD of `M · B`; the cache's PMPA is updated
in place through its `Ref` fields. When the local space at the center is too small to
carry the rank-`alg.R` purification (boundary sites), the center is merged with
neighbouring sites (`_merged_left_move!`) so the local effective Hamiltonian can
support rank `R` again.
"""
function leftsweep!(env::ThermalDMRGCache, alg::PDMRG; β::Real)
	L = length(env.rho)
	fvals = Float64[]
	s = 1
	merged_to_end = false
	while s < L
		dims = size(env.rho.mcenter[])[1:3]
		if prod(dims) >= alg.R
			push!(fvals, _left_move!(env, s, β, alg))
			s += 1
		else
			push!(fvals, _merged_left_move!(env, s, β, alg))
			s = env.rho.center[]
			merged_to_end = (s == L)
		end
		(alg.verbosity > 2) && _logupdate(stdout, "l2r", s, fvals[end])
	end
	if !merged_to_end
		m, f = thermalize_center(env, L, β, alg)
		env.rho.mcenter[] = m
		env.rho.center[] = L
		push!(fvals, f)
		(alg.verbosity > 2) && _logupdate(stdout, "l2r", L, f)
	end
	return fvals
end

function _left_move!(env::ThermalDMRGCache, s::Int, β::Real, alg::PDMRG)
	m, f = thermalize_center(env, s, β, alg)
	env.rho.mcenter[] = m
	B = env.rho[s+1]
	# M[aL, ps, ac, τ] · B[ac, ps1, aR]  ->  Ψ[aL, ps, ps1, aR, τ]
	@tensor Ψ[aL, ps, ps1, aR, τ] := env.rho.mcenter[][aL, ps, ac, τ] * B[ac, ps1, aR]
	# SVD: left group (aL, ps) gives A[aL, ps, new]; right group (ps1, aR, τ) goes to center
	u, sv, v, err = tsvd!(Ψ, (1, 2), (3, 4, 5); trunc=alg.trunc)
	env.rho.data[s] = u                         # A[aL, ps, new] (left-canonical)
	# v[new, ps1, aR, τ] -> Mnew[new, ps1, aR, τ] after absorbing singular values
	sm = Diagonal(sv)
	@tensor Mnew[nb, ps1, aR, τ] := sm[nb, j] * v[j, ps1, aR, τ]
	env.rho.mcenter[] = Mnew
	env.rho.center[] = s + 1
	updateleft!(env, s)
	return f
end

"""
	rightsweep!(env::ThermalDMRGCache, alg::PDMRG; β) -> fvals

Right-to-left p-DMRG sweep (symmetric to `leftsweep!`, with merged boundary updates at
the left edge).
"""
function rightsweep!(env::ThermalDMRGCache, alg::PDMRG; β::Real)
	L = length(env.rho)
	fvals = Float64[]
	s = L
	merged_to_end = false
	while s > 1
		dims = size(env.rho.mcenter[])[1:3]
		if prod(dims) >= alg.R
			push!(fvals, _right_move!(env, s, β, alg))
			s -= 1
		else
			push!(fvals, _merged_right_move!(env, s, β, alg))
			s = env.rho.center[]
			merged_to_end = (s == 1)
		end
		(alg.verbosity > 2) && _logupdate(stdout, "r2l", s, fvals[end])
	end
	if !merged_to_end
		m, f = thermalize_center(env, 1, β, alg)
		env.rho.mcenter[] = m
		env.rho.center[] = 1
		push!(fvals, f)
		(alg.verbosity > 2) && _logupdate(stdout, "r2l", 1, f)
	end
	return fvals
end

function _right_move!(env::ThermalDMRGCache, s::Int, β::Real, alg::PDMRG)
	m, f = thermalize_center(env, s, β, alg)
	env.rho.mcenter[] = m
	A = env.rho[s-1]
	# A[aL, ps1, ac] · M[ac, ps, aR, τ]  ->  Ψ[aL, ps1, ps, aR, τ]
	@tensor Ψ[aL, ps1, ps, aR, τ] := A[aL, ps1, ac] * env.rho.mcenter[][ac, ps, aR, τ]
	# SVD: left group (aL, ps1, τ) goes to center; right group (ps, aR) gives B[ps, aR, new]
	u, sv, v, err = tsvd!(Ψ, (1, 2, 5), (3, 4); trunc=alg.trunc)
	env.rho.data[s] = v                         # B[ps, aR, new] (right-canonical)
	# u[aL, ps1, τ, new] -> Mnew[aL, ps1, new, τ] after absorbing singular values
	sm = Diagonal(sv)
	@tensor Mnew[aL, ps1, nb, τ] := u[aL, ps1, τ, j] * sm[j, nb]
	env.rho.mcenter[] = Mnew
	env.rho.center[] = s - 1
	updateright!(env, s)
	return f
end

sweep!(env::ThermalDMRGCache, alg::PDMRG; β::Real) =
	vcat(leftsweep!(env, alg; β), rightsweep!(env, alg; β))

# ---------- driver ----------

"""
	thermalstate!(rho::PositiveMPA, h::AbstractMPO, β::Real, alg::PDMRG=PDMRG()) -> rho

Variational search for the equilibrium state `ρ = exp(-β h) / Z` in the PMPA `rho`.
The canonical `data` tensors are updated in place; because the ansatz is immutable, the
updated PMPA is **returned** and should be reassigned: `rho = thermalstate!(rho, h, β, alg)`.
The iteration loop is the generic `iterative_compute!` with `β` forwarded to the sweeps.
"""
function thermalstate!(rho::PositiveMPA, h::AbstractMPO, β::Real, alg::PDMRG=PDMRG())
	env = ThermalDMRGCache(h, rho)
	iterative_compute!(env, alg; β)
	return env.rho
end

"""
	thermalstate(h::AbstractMPO, β::Real, alg::PDMRG=PDMRG()) -> (F, rho)

Free energy `F = -log(Z)/β` and the PMPA equilibrium state of `h` at inverse temperature
`β`, starting from a random PMPA with the bond cap carried by `alg.trunc` and rank `alg.R`.
"""
function thermalstate(h::AbstractMPO, β::Real, alg::PDMRG=PDMRG())
	rho = randompmpa(scalartype(h), ophydims(h); D=_guess_bond(alg.trunc), R=alg.R)
	rho = thermalstate!(rho, h, β, alg)
	F = freeenergy(h, rho, β)
	return F, rho
end

"""
	freeenergy(h::AbstractMPO, rho::PositiveMPA, β::Real) -> Real

Full-chain free energy `F = tr(H ρ) - T S(ρ)` of the PMPA state `rho` at inverse
temperature `β` (`T = 1/β`), with the entropy from the center's local spectrum (the
spectrum of `ρ` equals that of its center block).
"""
function freeenergy(h::AbstractMPO, rho::PositiveMPA, β::Real)
	env = ThermalDMRGCache(h, rho)
	s = rho.center[]
	M = rho.mcenter[]
	s1, s2, s3, R = size(M)
	T = eltype(M)
	E = zero(real(T))
	for τ in 1:R
		x = M[:, :, :, τ]
		y = ac_prime(x, env.H[s], env.hstorage[s], env.hstorage[s+1])
		E += real(dot(x, y))
	end
	# the center block of ρ̃: ρ̃[a, b] = Σ_τ M_τ[a] · conj(M_τ[b]) (hermitian)
	rholoc = zeros(T, s1 * s2 * s3, s1 * s2 * s3)
	for τ in 1:R
		vv = vec(M[:, :, :, τ])
		rholoc .+= vv .* vv'
	end
	pv = eigen(Hermitian(rholoc)).values
	S = -sum(p -> p > 1e-14 ? p * log(p) : 0.0, real.(pv))
	return E - S / β
end
