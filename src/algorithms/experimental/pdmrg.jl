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
# Reference: C. Guo, "Density Matrix Renormalization Group Algorithm For Mixed Quantum
# States", Phys. Rev. B 105, 195152 (2022) (arXiv:2203.16350).

# ---------- algorithm type ----------

"""
	PDMRG(; maxiter=Defaults.maxiter, tol=Defaults.tol, trunc=truncdim(D=Defaults.D), R=20, verbosity=0)

Parameters of the positive DMRG thermal-state search. `trunc` is the truncation scheme
applied to the center-movement SVDs (its bond cap bounds the MPS bond dimension of the
isometric part of the PMPA) and `R` the rank of the orthogonal center (the maximal
Schmidt number of the represented mixed state).
"""
@kwdef struct PDMRG{TR<:TruncationScheme} <: TwoSiteUpdate
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	trunc::TR = truncdim(D=Defaults.D)
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
	s1, s2, s3 = size(rho.mcenter[])[1:3]
	L = s1 * s2 * s3
	if L <= 4 * alg.R
		return _thermalize_dense(env, site, β, alg, s1, s2, s3)
	else
		return _thermalize_krylov(env, site, β, alg, s1, s2, s3)
	end
end

# dense path: build the full effective Hamiltonian and exponentiate
function _thermalize_dense(env, site, β, alg, s1, s2, s3)
	heff = _local_hamiltonian_matrix(env, site, s1, s2, s3)
	# rho = exp(-β H_eff) / Z
	rho = Matrix(exp(-β * Hermitian(heff)))
	rho ./= tr(rho)
	eigvals, eigvecs = eigen(Hermitian(rho))
	M, kept = _eigvecs_to_center(eigvals, eigvecs, alg, s1, s2, s3)
	return M, _local_free_energy(env, site, M, kept, β)
end

# Krylov path: at low β the thermal state is dominated by the lowest eigenstates of H_eff
function _thermalize_krylov(env, site, β, alg, s1, s2, s3)
	L = s1 * s2 * s3
	x0 = env.rho.mcenter[][:, :, :, 1]
	n = min(alg.R, L)
	vals, vecs, info = eigsolve(y -> ac_prime(y, env.H[site], env.hstorage[site], env.hstorage[site+1]),
								x0, n, :SR; ishermitian=true, tol=1e-10, krylovdim=2n+2)
	# Boltzmann weights of the kept lowest eigenstates
	eigvals = exp.(-β .* real(vals))
	eigvals ./= sum(eigvals)
	Dm = length(eigvals)
	T = eltype(env.rho.mcenter[])
	M = zeros(T, s1, s2, s3, Dm)
	for i in 1:Dm
		M[:, :, :, i] = real.(vecs[i])
	end
	M = reshape(reshape(M, L, Dm) * Diagonal(sqrt.(eigvals)), s1, s2, s3, Dm)
	return M, _local_free_energy(env, site, M, eigvals, β)
end

function _local_hamiltonian_matrix(env, site, s1, s2, s3)
	# build the dense effective Hamiltonian as a matrix in the local tensor basis
	W = env.H[site]
	hl = env.hstorage[site]
	hr = env.hstorage[site+1]
	T = eltype(hl)
	L = s1 * s2 * s3
	H = zeros(T, L, L)
	for k in 1:L
		v = zeros(T, s1, s2, s3)
		v[k] = 1
		H[:, k] .= vec(ac_prime(v, W, hl, hr))
	end
	return H
end

function _eigvecs_to_center(eigvals, eigvecs, alg::PDMRG, s1, s2, s3)
	# keep the alg.R largest (positive) eigenvalues, renormalized
	pos = findfirst(>(0), eigvals)
	pos === nothing && (pos = length(eigvals))
	keep = min(alg.R, length(eigvals) - pos + 1)
	idx = (length(eigvals) - keep + 1):length(eigvals)
	vals = eigvals[idx]
	vals ./= sum(vals)
	vecs = eigvecs[:, idx]
	M = reshape(vecs * Diagonal(sqrt.(vals)), s1, s2, s3, keep)
	return M, vals
end

function _local_free_energy(env, site, M, eigvals, β::Real)
	# β F = β E - S, evaluated consistently on the (truncated) new center:
	# E = Σ_τ ⟨M_τ|H_eff|M_τ⟩,  S = -Σ p log p over the kept eigenvalues.
	S = -sum(p -> p > 0 ? p * log(p) : zero(p), eigvals)
	E = zero(real(eltype(eigvals)))
	W = env.H[site]
	hl = env.hstorage[site]
	hr = env.hstorage[site+1]
	for τ in 1:size(M, 4)
		x = M[:, :, :, τ]
		y = ac_prime(x, W, hl, hr)
		E += real(dot(x, y))
	end
	return β * E - S   # β F = β<E> - S(ρ̃)  (the trace of ρ̃ is 1)
end

# ---------- sweeps ----------

"""
	leftsweep!(env::ThermalDMRGCache, alg::PDMRG; β) -> fvals

Left-to-right p-DMRG sweep. At each site the local thermal state is computed, then the
center is moved one site to the right by an SVD of `M · B`; the cache's PMPA is updated
in place through its `Ref` fields.
"""
function leftsweep!(env::ThermalDMRGCache, alg::PDMRG; β::Real)
	L = length(env.rho)
	fvals = zeros(Float64, L)
	for s in 1:L-1
		fvals[s] = _left_move!(env, s, β, alg)
	end
	m, f = thermalize_center(env, L, β, alg)
	env.rho.mcenter[] = m
	env.rho.center[] = L
	fvals[L] = f
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

Right-to-left p-DMRG sweep (symmetric to `leftsweep!`).
"""
function rightsweep!(env::ThermalDMRGCache, alg::PDMRG; β::Real)
	L = length(env.rho)
	fvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		fvals[k] = _right_move!(env, s, β, alg)
		k += 1
	end
	m, f = thermalize_center(env, 1, β, alg)
	env.rho.mcenter[] = m
	env.rho.center[] = 1
	fvals[L] = f
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
	E = 0.0
	for τ in 1:R
		x = M[:, :, :, τ]
		y = ac_prime(x, env.H[s], env.hstorage[s], env.hstorage[s+1])
		E += real(dot(x, y))
	end
	rholoc = zeros(Float64, s1 * s2 * s3, s1 * s2 * s3)
	for τ in 1:R
		vv = vec(M[:, :, :, τ])
		rholoc .+= vv .* vv'
	end
	pv = eigen(Hermitian(rholoc)).values
	S = -sum(p -> p > 1e-14 ? p * log(p) : 0.0, pv)
	return E - S / β
end
