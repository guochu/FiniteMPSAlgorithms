# DMRG2 engine for the arithmetics (mult / add / compress / hadamard / linsolve):
# two-site variational sweeps. At each pair (s, s+1) the local target — the same
# environment contraction the single-site ALS uses — is formed for both sites jointly
# and re-split by a truncating SVD (`alg.trunc`), so the bond dimension adapts during
# the sweeps (growth where the environment demands it, truncation where the scheme
# caps it). The raw singular values are absorbed into the sweep direction WITHOUT
# normalization (as in the single-site ALS gauge moves): the data carries the scale of
# the local targets through the sweeps, and their normalized spectra are recorded as
# the bond Schmidt values. Each driver ends by attaching the operand external scales
# and folding the center tensor's norm (the total data norm) into the `scaling` field —
# the output is canonical (isometric sites, initialized Schmidt values) and its
# represented value equals the strict operation up to the sweep truncations. The
# convergence measure is the norm of the two-site optimal target, monotone in
# processing time, exactly as in the single-site sweeps.

# ---------- generic two-site re-split ----------

# re-split the optimal two-site block of a rank-3 bra (MPS case): block legs
# (bL, p1, p2, bR); the raw singular values are absorbed into the sweep direction
# without normalization (the data keeps the scale of the local target, as in the
# single-site ALS gauge moves), and their normalized spectrum is recorded as the bond's
# Schmidt values
function _als2_update!(bra, s::Integer, t2::AbstractArray{T,4}, alg::DMRG2;
					   move_right::Bool) where {T}
	u, sv, v, _ = tsvd!(t2, (1, 2), (3, 4); trunc=alg.trunc)
	n = norm(sv)
	n == 0 || (bra.s[s+1] = sv ./ n)
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
	n == 0 || (bra.s[s+1] = sv ./ n)
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

# ⟨x|A†A|z⟩ over the pair: z2[zL, p1, p2, zR]; W1/W2 are the MPO site tensors (used
# as-is — the pair block is never materialized)
# hL legs (xL, A†-bond, A-bond, zL); hR legs (xR, A†-bond, A-bond, zR)
# contraction order: the large MPS bonds (zL, zR — far larger than the MPO bonds and
# physical dimensions) are folded into the environments first; every intermediate stays
# at O(x·z·D_Aᵏ·dᵏ) instead of materializing zL·zR-scaled blocks
function _h2_apply(z2::AbstractArray{T,4}, W1, W2,
				   hL::AbstractArray{T,4}, hR::AbstractArray{T,4}) where {T}
	# fold z2 into the right environment over the large ket bond zR
	t1 = @tensor t1[zL, i1, i2, xR, cR, w2] := hR[xR, cR, w2, zR] * z2[zL, i1, i2, zR]
	# apply the MPO block site by site (small physical and A-bond legs)
	t2 = @tensor t2[zL, i1, xR, cR, c, o2] := t1[zL, i1, i2, xR, cR, w2] *
											  W2[c, o2, w2, i2]
	t3 = @tensor t3[zL, xR, cR, w1, o1, o2] := t2[zL, i1, xR, cR, c, o2] *
											   W1[w1, o1, c, i1]
	# wrap with A†2: its input legs become the free physical legs of the target
	t4 = @tensor t4[zL, xR, w1, c, o1, p2] := conj(W2[c, o2, cR, p2]) *
											   t3[zL, xR, cR, w1, o1, o2]
	t5 = @tensor t5[zL, xR, w1, cL, p1, p2] := conj(W1[cL, o1, c, p1]) *
											   t4[zL, xR, w1, c, o1, p2]
	# close with the left environment over the large ket bond zL and the small A-bonds
	return @tensor y[xL, p1, p2, xR] := hL[xL, cL, w1, zL] * t5[zL, xR, w1, cL, p1, p2]
end

# ⟨x|A†|y⟩ over the pair: the two-site right-hand side. As in the single-site
# `_b_target`, y's physicals contract the MPO PO legs and the free legs are the MPO PI
# legs (A†y lives on A's input space). Large MPS bonds folded into the environments
# first (same order as `_h2_apply`).
function _b2_target(W1, W2, y2::AbstractArray{T,4}, bL::AbstractArray{T,3}, bR::AbstractArray{T,3}) where {T}
	t1 = @tensor t1[kL, o1, o2, xR, cR] := bR[xR, cR, kR] * y2[kL, o1, o2, kR]
	t2 = @tensor t2[kL, o1, xR, c, q2] := conj(W2[c, o2, cR, q2]) * t1[kL, o1, o2, xR, cR]
	t3 = @tensor t3[kL, xR, cL, q1, q2] := conj(W1[cL, o1, c, q1]) * t2[kL, o1, xR, c, q2]
	return @tensor t[xL, q1, q2, xR] := bL[xL, cL, kL] * t3[kL, xR, cL, q1, q2]
end

function _site_solve2(m::LinsolveCache, s::Integer)
	W1, W2 = m.mpo[s], m.mpo[s+1]
	y2 = @tensor yy[a, p1, p2, b] := m.bra[s][a, p1, c] * m.bra[s+1][c, p2, b]
	t = _b2_target(W1, W2, y2, m.bstorage[s], m.bstorage[s+2])
	shape = (size(m.hstorage[s], 1), size(W1, 4), size(W2, 4), size(m.hstorage[s+2], 1))
	z0 = size(m.ket[s]) == shape ? m.ket[s] : zero(t)
	# the local normal equation is solved iteratively (KrylovKit) on the linear
	# operator action — the dense H matrix is never formed
	z, _ = KrylovKit.linsolve(y -> _h2_apply(y, W1, W2, m.hstorage[s], m.hstorage[s+2]),
							  t, z0; ishermitian=true, tol=Defaults.tol, krylovdim=25, maxiter=100)
	return z, t
end

# the exact global residual² ‖A·x − y‖², evaluated from the local decomposition at the
# pair with the updated tensor z — no global contraction is needed
function _site_loss(m::LinsolveCache, s::Integer, z::AbstractArray{T,4}, t::AbstractArray{T,4}) where {T}
	Hz = _h2_apply(z, m.mpo[s], m.mpo[s+1], m.hstorage[s], m.hstorage[s+2])
	return real(dot(z, Hz)) - 2 * real(dot(z, t)) + m.ynorm2
end

function leftsweep!(m::LinsolveCache, alg::DMRG2)
	L = length(m.ket)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		z2, t = _site_solve2(m, s)
		kvals[s] = _site_loss(m, s, z2, t)
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
		z2, t = _site_solve2(m, s - 1)
		kvals[k] = _site_loss(m, s - 1, z2, t)
		k += 1
		_als2_update!(m.ket, s - 1, z2, alg; move_right=false)
		_h_right!(m, s)
		_b_right!(m, s)
	end
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(m::LinsolveCache, alg::DMRG2) = vcat(leftsweep!(m, alg), rightsweep!(m, alg))

# ---------- exported interface (DMRG2) ----------
# the initial guesses are drawn with the bond cap carried by `alg.trunc`
# (`_guess_bond`); the two-site sweeps adapt the bond dimension during the iterations,
# so the guess may be smaller than the final bond
#
# the sweeps keep the natural data scale (as in DMRG1) and record the bond spectra:
# each driver attaches the operand external scales through the `scaling` field and
# folds the center norm into it — the output is `iscanonical` and its represented value
# equals the strict operation up to the sweep truncations

function mult!(out, h, x, alg::DMRG2)
	cache = MultCache(h, x, out)
	iterative_compute!(cache, alg)
	# attach the external operand scales and fold the center tensor's norm (the total
	# data norm — the swept chain is isometric on the other sites) into the `scaling`
	# field; the bond spectra recorded during the sweeps initialize the Schmidt values
	s = _opscaling(h) * _opscaling(x)
	s == 1 || setscaling!(out, s)
	_renormalize!(out, out[1], false)
	return out
end

function mult(h::AbstractMPO, x, alg::DMRG2)
	(length(h) == length(x)) || throw(ArgumentError("dimension mismatch"))
	return mult!(svdguess_mult(h, x, _guess_bond(alg.trunc)), h, x, alg)
end
mult(hA::MPOHamiltonian, x, alg::DMRG2) = mult(MPO(tompotensors(hA)), x, alg)
mult(h::AbstractMPO, hB::MPOHamiltonian, alg::DMRG2) = mult(h, MPO(tompotensors(hB)), alg)
mult(hA::MPOHamiltonian, hB::MPOHamiltonian, alg::DMRG2) =
	mult(MPO(tompotensors(hA)), MPO(tompotensors(hB)), alg)

# the KKT eigenvalue of the converged ALS direction: (linear form) / ⟨dir|dir⟩
_kkt_ratio(form::Number, dirnorm2::Number) = form / dirnorm2

function add!(out, chains, alg::DMRG2)
	cache = AddCache(out, chains)
	iterative_compute!(cache, alg)
	# Σ kets = β·bra at convergence: β = Σ_n ⟨bra|ket_n⟩ / ⟨bra|bra⟩. Unlike the other
	# DMRG2 drivers, `add` restores the physical scale of the (cancellation-prone) sum
	# from the problem's KKT eigenvalue; `setscaling!` first, `lmul!` second (the
	# folding `lmul!` must not be overwritten by a later `setscaling!`)
	setscaling!(out, scaling(chains[1]))
	form = sum(_dot(out, c) for c in chains)
	lmul!(_kkt_ratio(form, _dot(out, out)), out)
	return out
end

function add(ψs::Vector{<:CanonicalMPS}, alg::DMRG2)
	isempty(ψs) && throw(ArgumentError("empty input"))
	return add!(svdguess_add(ψs, _guess_bond(alg.trunc)), ψs, alg)
end
function add(ρs::Vector{<:CanonicalMPO}, alg::DMRG2)
	isempty(ρs) && throw(ArgumentError("empty input"))
	return add!(svdguess_add(ρs, _guess_bond(alg.trunc)), ρs, alg)
end
add(ψA::CanonicalMPS, ψB::CanonicalMPS, alg::DMRG2) = add([ψA, ψB], alg)
add(ρA::CanonicalMPO, ρB::CanonicalMPO, alg::DMRG2) = add([ρA, ρB], alg)

function compress!(out, x, alg::DMRG2)
	cache = OverlapCache(out, x)
	iterative_compute!(cache, alg)
	# attach the external operand scale and fold the center norm into `scaling` (see `mult!`)
	x isa Union{CanonicalMPS, CanonicalMPO} && setscaling!(out, scaling(x))
	_renormalize!(out, out[1], false)
	return out
end

function compress(x, alg::DMRG2)
	out = svdguess_compress(x, _guess_bond(alg.trunc))
	compress!(out, x, alg)
	return out
end
compress(h::MPOHamiltonian, alg::DMRG2) = compress(MPO(tompotensors(h)), alg)

function hadamard!(χ, ψA, ψB, alg::DMRG2)
	cache = HadamardCache(ψA, ψB, χ)
	iterative_compute!(cache, alg)
	# attach the ⊙ convention scale scaling(ψA)·scaling(ψB) and fold the center norm
	# into `scaling` (see `mult!`)
	setscaling!(χ, scaling(ψA) * scaling(ψB))
	_renormalize!(χ, χ[1], false)
	return χ
end

function hadamard(ψA::CanonicalMPS, ψB::CanonicalMPS, alg::DMRG2)
	_validate_hadamard(ψA, ψB)
	return hadamard!(svdguess_hadamard(ψA, ψB, _guess_bond(alg.trunc)), ψA, ψB, alg)
end

function linsolve!(x, A, y, alg::DMRG2)
	m = LinsolveCache(A, y, x)
	iterative_compute!(m, alg)
	# the solution is expressed in the right-hand side's per-site scaling convention;
	# fold the center norm into `scaling` (see `mult!`)
	setscaling!(m.ket, scaling(y))
	_renormalize!(m.ket, m.ket[1], false)
	return x
end

function linsolve(A::AbstractMPO, y::CanonicalMPS, alg::DMRG2)
	_validate_linsolve(A, y)
	T = promote_type(scalartype(A), scalartype(y))
	x = randommps(T, ophydims(A); D=_guess_bond(alg.trunc), normalize=false)
	return linsolve!(x, A, y, alg)
end
linsolve(A::MPOHamiltonian, y::CanonicalMPS, alg::DMRG2) =
	linsolve(MPO(tompotensors(A)), y, alg)
