# DMRG2 engine for the arithmetics (mult / add / compress / hadamard / linsolve):
# two-site variational sweeps. At each pair (s, s+1) the local target — the same
# environment contraction the single-site ALS uses — is formed for both sites jointly
# and re-split by a truncating SVD (`alg.trunc`), so the bond dimension adapts during
# the sweeps (growth where the environment demands it, truncation where the scheme
# caps it). The convergence measure is the norm of the two-site optimal target,
# monotone in processing time, exactly as in the single-site sweeps.

# ---------- generic two-site re-split ----------

# re-split the optimal two-site block of a rank-3 bra (MPS case): block legs
# (bL, p1, p2, bR); the singular values are absorbed into the sweep direction and
# normalized — the two-site ALS determines the direction only (the scale direction of
# the un-normalized iteration is divergent under truncation); the physical scale is
# restored once by the driver from the problem's KKT eigenvalue
function _als2_update!(bra, s::Integer, t2::AbstractArray{T,4}, alg::DMRG2;
					   move_right::Bool) where {T}
	u, sv, v, _ = tsvd!(t2, (1, 2), (3, 4); trunc=alg.trunc)
	n = norm(sv)
	n == 0 || (sv ./= n)
	if move_right
		bra[s] = u
		sm = Diagonal(sv)
		@tensor vnew[b, p2, c] := sm[b, j] * v[j, p2, c]
		bra[s+1] = vnew
	else
		bra[s+1] = v
		sm = Diagonal(sv)
		@tensor unew[a, p1, b] := u[a, p1, j] * sm[j, b]
		bra[s] = unew
	end
	return bra
end

# rank-4 bra (density-matrix / MPO case): block legs (bL, po1, po2, pi1, pi2, bR); the
# re-split keeps the (p_out, p_in) pair of each site on one factor
function _als2_update!(bra, s::Integer, t2::AbstractArray{T,6}, alg::DMRG2;
					   move_right::Bool) where {T}
	u, sv, v, _ = tsvd!(t2, (1, 2, 4), (3, 5, 6); trunc=alg.trunc)
	# u: (bL, po1, pi1, md); v: (md, po2, pi2, bR)
	n = norm(sv)
	n == 0 || (sv ./= n)
	sm = Diagonal(sv)
	if move_right
		bra[s] = permutedims(u, (1, 2, 4, 3))            # (bL, po1, aR, pi1)
		vn = @tensor vn[b, p2, q2, c] := sm[b, j] * v[j, p2, q2, c]
		bra[s+1] = permutedims(vn, (1, 2, 4, 3))         # (aL, po2, aR, pi2)
	else
		bra[s+1] = permutedims(v, (1, 2, 4, 3))          # (aL, po2, aR, pi2)
		@tensor unew[a, p1, b, q1] := u[a, p1, q1, j] * sm[j, b]
		bra[s] = unew                                    # (bL, po1, aR, pi1)
	end
	return bra
end

# ---------- MultCache ----------

# the two-site optimal block: left environment · ket pair · MPO pair · right environment
function _reduce_site2(m::MultCache, s::Integer)
	cL, cR = m.hstorage[s], m.hstorage[s+2]
	W2 = @tensor W[u, q1, q2, v, r1, r2] := m.H[s][u, q1, c, r1] * m.H[s+1][c, q2, v, r2]
	if m.ket[s] isa MPSTensor
		k2 = @tensor k[a, q1, q2, b] := m.ket[s][a, q1, c] * m.ket[s+1][c, q2, b]
		return @tensor t[-1, -2, -3, -4] := cL[-1, 1, 2] * k2[2, q1, q2, 3] *
										   W2[1, -2, -3, 4, q1, q2] * cR[-4, 4, 3]
	end
	# explicit binary contraction steps (the n-ary tree is unreliable at 6 free legs).
	# W2 legs: (wL, po1, po2, wR, pi1, pi2); the ket's OUT legs contract W2's IN legs
	# (the H·ket matrix product, matching `_updateleft3`), the bra's free pair is
	# (W2.po, ket.pi)
	k2 = @tensor k[a, o1, o2, b, i1, i2] := m.ket[s][a, o1, c, i1] * m.ket[s+1][c, o2, b, i2]
	hh = @tensor hh[b, w1, o1, o2, bR, i1, i2] := cL[b, w1, kL] * k2[kL, o1, o2, bR, i1, i2]
	hw = @tensor hw[b, p1, p2, w2, bR, i1, i2] := hh[b, w1, o1, o2, bR, i1, i2] *
												 W2[w1, p1, p2, w2, o1, o2]
	return @tensor t[b, -2, -3, -4, -5, -6] := hw[b, -2, -3, w2, bR, -4, -5] *
											   cR[-6, w2, bR]
end

function leftsweep!(m::MultCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		t2 = _reduce_site2(m, s)
		kvals[s] = norm(t2)
		_als2_update!(m.bra, s, t2, alg; move_right=true)
		# hstorage[s+1] = left env over sites 1..s — exactly the next pair's left env;
		# the right envs hstorage[s+2..] must stay untouched
		_env_updateleft!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(m::MultCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		t2 = _reduce_site2(m, s - 1)
		kvals[k] = norm(t2)
		k += 1
		_als2_update!(m.bra, s - 1, t2, alg; move_right=false)
		_env_updateright!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(m::MultCache, alg::DMRG2) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# ---------- AddCache / CompressCache (OverlapCache environments) ----------

# the two-site optimal overlap block ⟨bra|ket⟩ over the pair
function _reduce_two_site(c::OverlapCache, s::Integer)
	if c.ket[s] isa MPOTensor
		k2 = @tensor k[a, o1, o2, b, i1, i2] := c.ket[s][a, o1, d, i1] * c.ket[s+1][d, o2, b, i2]
		return @tensor t[-1, -2, -3, -4, -5, -6] := c.cstorage[s][-1, 1] *
			k2[1, -2, -3, 2, -4, -5] * c.cstorage[s+2][-6, 2]
	end
	k2 = @tensor k[a, p1, p2, b] := c.ket[s][a, p1, d] * c.ket[s+1][d, p2, b]
	return @tensor t[-1, -2, -3, -4] := c.cstorage[s][-1, 1] * k2[1, -2, -3, 2] *
									   c.cstorage[s+2][-4, 2]
end

function _reduce_two_site(m::AddCache, s::Integer)
	acc = missing
	for c in m.hstorage
		t = _reduce_two_site(c, s)
		acc = acc isa Missing ? t : acc + t
	end
	acc isa Missing && throw(ArgumentError("empty AddCache"))
	return acc
end

function leftsweep!(m::AddCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		t2 = _reduce_two_site(m, s)
		kvals[s] = norm(t2)
		_als2_update!(m.bra, s, t2, alg; move_right=true)
		for c in m.hstorage
			updateleft!(c, s)
		end
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(m::AddCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		t2 = _reduce_two_site(m, s - 1)
		kvals[k] = norm(t2)
		k += 1
		_als2_update!(m.bra, s - 1, t2, alg; move_right=false)
		for c in m.hstorage
			updateright!(c, s)
		end
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(m::AddCache, alg::DMRG2) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

function leftsweep!(c::OverlapCache, alg::DMRG2)
	L = length(c.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		t2 = _reduce_two_site(c, s)
		kvals[s] = norm(t2)
		_als2_update!(c.bra, s, t2, alg; move_right=true)
		updateleft!(c, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(c::OverlapCache, alg::DMRG2)
	L = length(c.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		t2 = _reduce_two_site(c, s - 1)
		kvals[k] = norm(t2)
		k += 1
		_als2_update!(c.bra, s - 1, t2, alg; move_right=false)
		updateright!(c, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(c::OverlapCache, alg::DMRG2) = vcat(leftsweep!(c, alg), rightsweep!(c, alg))

# ---------- HadamardCache ----------

# the two-site optimal pointwise-product block: the fused pairs of both sites
function _reduce_hadamard_site2(m::HadamardCache, s::Integer)
	A2 = @tensor a[aL, p1, p2, aR] := m.ketx[s][aL, p1, b] * m.ketx[s+1][b, p2, aR]
	B2 = @tensor b[bL, q1, q2, bR] := m.kety[s][bL, q1, c] * m.kety[s+1][c, q2, bR]
	# the pointwise product shares ONE physical value per site: broadcast, not contract
	KB2 = reshape(A2, size(A2, 1), 1, size(A2, 2), size(A2, 3), size(A2, 4), 1) .*
		  reshape(B2, 1, size(B2, 1), size(B2, 2), size(B2, 3), 1, size(B2, 4))
	# KB2: (aL, bL, p1, p2, aR, bR)
	return @tensor t[-1, -2, -3, -4] := m.hstorage[s][-1, 1, 2] * KB2[1, 2, -2, -3, 3, 4] *
									   m.hstorage[s+2][-4, 3, 4]
end

function leftsweep!(m::HadamardCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		t2 = _reduce_hadamard_site2(m, s)
		kvals[s] = norm(t2)
		_als2_update!(m.bra, s, t2, alg; move_right=true)
		_env_updateleft!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(m::HadamardCache, alg::DMRG2)
	L = length(m.bra)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		t2 = _reduce_hadamard_site2(m, s - 1)
		kvals[k] = norm(t2)
		k += 1
		_als2_update!(m.bra, s - 1, t2, alg; move_right=false)
		_env_updateright!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(m::HadamardCache, alg::DMRG2) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# ---------- LinsolveCache ----------

# the two-site MPO block of the operator
function _two_site_mpo(m::LinsolveCache, s::Integer)
	return @tensor W[w1, o1, o2, w2, i1, i2] := m.mpo[s][w1, o1, c, i1] *
											   m.mpo[s+1][c, o2, w2, i2]
end

# ⟨x|A†A|z⟩ over the pair: z2[zL, p1, p2, zR]
# hL legs (xL, A†-bond, A-bond, zL); hR legs (xR, A†-bond, A-bond, zR)
function _h2_apply(z2::AbstractArray{T,4}, W2, hL::AbstractArray{T,4}, hR::AbstractArray{T,4}) where {T}
	# (A2 z2) at the ket layer: sum z2's physicals against the MPO block inputs
	Az = @tensor az[w1, o1, o2, w2, zL, zR] := z2[zL, i1, i2, zR] *
											   W2[w1, o1, o2, w2, i1, i2]
	# wrap with A†2 and the environments
	@tensor y[-1, -2, -3, -4] := hL[-1, cL, aL, zL] *
								 conj(W2[cL, o1, o2, cR, -2, -3]) *
								 Az[aL, o1, o2, w2, zL, zR] * hR[-4, cR, w2, zR]
	return y
end

# ⟨x|A†|y⟩ over the pair: the two-site right-hand side. As in the single-site
# `_b_target`, y's physicals contract W2's PO legs and the free legs are W2's PI legs
# (A†y lives on A's input space).
function _b2_target(W2, y2::AbstractArray{T,4}, bL::AbstractArray{T,3}, bR::AbstractArray{T,3}) where {T}
	@tensor t[-1, -2, -3, -4] := bL[-1, cL, kL] * conj(W2[cL, p1, p2, cR, -2, -3]) *
								 y2[kL, p1, p2, kR] * bR[-4, cR, kR]
	return t
end

function _site_solve2(m::LinsolveCache, s::Integer)
	W2 = _two_site_mpo(m, s)
	y2 = @tensor yy[a, p1, p2, b] := m.bra[s][a, p1, c] * m.bra[s+1][c, p2, b]
	t = _b2_target(W2, y2, m.bstorage[s], m.bstorage[s+2])
	shape = (size(m.hstorage[s], 1), size(W2, 5), size(W2, 6), size(m.hstorage[s+2], 1))
	n = prod(shape)
	Hmat = zeros(promote_type(scalartype(m.bra), scalartype(W2)), n, n)
	E = zeros(scalartype(Hmat), shape)
	for i in 1:n
		fill!(E, 0)
		E[i] = 1
		Hmat[:, i] .= vec(_h2_apply(E, W2, m.hstorage[s], m.hstorage[s+2]))
	end
	return reshape(Hmat \ vec(t), shape)
end

function leftsweep!(m::LinsolveCache, alg::DMRG2)
	L = length(m.ket)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		z2 = _site_solve2(m, s)
		kvals[s] = norm(z2)
		_als2_update!(m.ket, s, z2, alg; move_right=true)
		_h_left!(m, s)
		_b_left!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

function rightsweep!(m::LinsolveCache, alg::DMRG2)
	L = length(m.ket)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		z2 = _site_solve2(m, s - 1)
		kvals[k] = norm(z2)
		k += 1
		_als2_update!(m.ket, s - 1, z2, alg; move_right=false)
		_h_right!(m, s)
		_b_right!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(m::LinsolveCache, alg::DMRG2) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# the normal-equation eigenvalue of the converged direction: ⟨x|A†y⟩ / ⟨x|A†A|x⟩
function _linsolve_kkt(m::LinsolveCache)
	T = scalartype(m.bra)
	h = ones(T, 1, 1, 1, 1)
	b = ones(T, 1, 1, 1)
	for s in 1:length(m.ket)
		h = _h_updateleft(h, m.ket[s], m.mpo[s], m.ket[s])
		b = _b_updateleft(b, m.ket[s], m.mpo[s], m.bra[s])
	end
	return b[1, 1] / h[1, 1, 1, 1]
end

# ---------- exported interface (DMRG2) ----------
# the initial guesses are drawn with the bond cap carried by `alg.trunc`
# (`_truncation_bond`, falling back to `Defaults.D`); the two-site sweeps adapt the
# bond dimension during the iterations, so the guess may be smaller than the final bond
#
# the normalized two-site sweeps converge the DIRECTION of the solution; the physical
# scale is restored once per driver from the problem's KKT eigenvalue
#   β = (linear form of the target along the converged direction) / ⟨dir|dir⟩,
# folded back into the data (`lmul!`), and the operand external scales are attached
# through the `scaling` field afterwards

# ⟨bra|H|ket⟩ of a MultCache at the data level
function _overlap3(m::MultCache)
	T = scalartype(m.bra)
	v = ones(T, 1, 1, 1)
	for s in eachindex(m.bra)
		v = _env_updateleft(v, m.bra[s], m.H[s], m.ket[s])
	end
	return v[1, 1, 1]
end

# ⟨χ|ψA ⊙ ψB⟩ of a HadamardCache at the data level
function _hadamard_overlap(m::HadamardCache)
	T = scalartype(m.bra)
	v = ones(T, 1, 1, 1)
	for s in eachindex(m.bra)
		v = _updateleft(v, m.bra[s], m.ketx[s], m.kety[s])
	end
	return v[1, 1, 1]
end

# the KKT eigenvalue of a normalized ALS direction: (linear form) / ⟨dir|dir⟩
_kkt_ratio(form::Number, dirnorm2::Number) = form / dirnorm2

_dmrg2_guessD(alg::DMRG2) = something(_truncation_bond(alg.trunc), Defaults.D)

function mult!(out, h, x, alg::DMRG2)
	cache = MultCache(h, x, out)
	iterative_compute!(cache, alg)
	# attach the external operand scales first, then restore the physical scale from the
	# problem's KKT eigenvalue (H ket = λ bra at convergence): `lmul!` folds the factor
	# into the per-site scaling, so it must come after `setscaling!`
	s = _opscaling(h) * _opscaling(x)
	s == 1 || setscaling!(out, s)
	lmul!(_kkt_ratio(_overlap3(cache), _dot(out, out)), out)
	return out
end

function mult(h::AbstractMPO, x, alg::DMRG2)
	(length(h) == length(x)) || throw(ArgumentError("dimension mismatch"))
	return mult!(svdguess_mult(h, x, _dmrg2_guessD(alg)), h, x, alg)
end
mult(hA::MPOHamiltonian, x, alg::DMRG2) = mult(MPO(tompotensors(hA)), x, alg)
mult(h::AbstractMPO, hB::MPOHamiltonian, alg::DMRG2) = mult(h, MPO(tompotensors(hB)), alg)
mult(hA::MPOHamiltonian, hB::MPOHamiltonian, alg::DMRG2) =
	mult(MPO(tompotensors(hA)), MPO(tompotensors(hB)), alg)

function add!(out, chains, alg::DMRG2)
	cache = AddCache(out, chains)
	iterative_compute!(cache, alg)
	# Σ kets = β·bra at convergence: β = Σ_n ⟨bra|ket_n⟩ / ⟨bra|bra⟩.
	# `setscaling!` first, `lmul!` second (see `mult!` — `lmul!` folds the factor into
	# the per-site scaling and must not be overwritten)
	setscaling!(out, scaling(chains[1]))
	form = sum(_dot(out, c) for c in chains)
	lmul!(_kkt_ratio(form, _dot(out, out)), out)
	return out
end

function add(ψs::Vector{<:CanonicalMPS}, alg::DMRG2)
	isempty(ψs) && throw(ArgumentError("empty input"))
	return add!(svdguess_add(ψs, _dmrg2_guessD(alg)), ψs, alg)
end
function add(ρs::Vector{<:CanonicalMPO}, alg::DMRG2)
	isempty(ρs) && throw(ArgumentError("empty input"))
	return add!(svdguess_add(ρs, _dmrg2_guessD(alg)), ρs, alg)
end
add(ψA::CanonicalMPS, ψB::CanonicalMPS, alg::DMRG2) = add([ψA, ψB], alg)
add(ρA::CanonicalMPO, ρB::CanonicalMPO, alg::DMRG2) = add([ρA, ρB], alg)

function compress!(out, x, alg::DMRG2)
	cache = OverlapCache(out, x)
	iterative_compute!(cache, alg)
	# x = β·out at convergence: β = ⟨out|x⟩ / ⟨out|out⟩ (same ordering as in `mult!`)
	x isa Union{CanonicalMPS, CanonicalMPO} && setscaling!(out, scaling(x))
	lmul!(_kkt_ratio(_dot(out, x), _dot(out, out)), out)
	return out
end

function compress(x, alg::DMRG2)
	out = svdguess_compress(x, _dmrg2_guessD(alg))
	compress!(out, x, alg)
	return out
end
compress(h::MPOHamiltonian, alg::DMRG2) = compress(MPO(tompotensors(h)), alg)

function hadamard!(χ, ψA, ψB, alg::DMRG2)
	cache = HadamardCache(ψA, ψB, χ)
	iterative_compute!(cache, alg)
	# ψA ⊙ ψB = β·χ at convergence: β = ⟨χ|ψA ⊙ ψB⟩ / ⟨χ|χ⟩ (same ordering as in `mult!`)
	setscaling!(χ, scaling(ψA) * scaling(ψB))
	lmul!(_kkt_ratio(_hadamard_overlap(cache), _dot(χ, χ)), χ)
	return χ
end

function hadamard(ψA::CanonicalMPS, ψB::CanonicalMPS, alg::DMRG2)
	_validate_hadamard(ψA, ψB)
	return hadamard!(svdguess_hadamard(ψA, ψB, _dmrg2_guessD(alg)), ψA, ψB, alg)
end

function linsolve!(x, A, y, alg::DMRG2)
	m = LinsolveCache(A, y, x)
	prev = Inf
	for _ in 1:alg.maxiter
		sweep!(m, alg)
		r = _residual_norm(m)
		abs(r - prev) < alg.tol && break
		prev = r
	end
	# the sweeps read raw data only: the solution is expressed in the right-hand side's
	# per-site scaling convention. A·x* = β·y' with the normal-equation eigenvalue
	# β = ⟨x|A†y⟩ / ⟨x|A†A|x⟩ restores the physical scale of the normalized direction
	# (same ordering as in `mult!`: `setscaling!` first, then the folding `lmul!`)
	setscaling!(m.ket, scaling(y))
	lmul!(_linsolve_kkt(m), m.ket)
	return x
end

function linsolve(A::AbstractMPO, y::CanonicalMPS, alg::DMRG2)
	_validate_linsolve(A, y)
	T = promote_type(scalartype(A), scalartype(y))
	x = randommps(T, ophydims(A); D=_dmrg2_guessD(alg), normalize=false)
	return linsolve!(x, A, y, alg)
end
linsolve(A::MPOHamiltonian, y::CanonicalMPS, alg::DMRG2) =
	linsolve(MPO(tompotensors(A)), y, alg)
