# Shared utilities of the two-site sweep engines (mult / add / compress / hadamard /
# linsolve): the DMRG2 (and generally `TwoSiteUpdate`) engine forms, at each pair
# (s, s+1), the local target — the same environment contraction the single-site ALS
# uses — for both sites jointly and re-splits it by a truncating SVD (`alg.trunc`), so
# the bond dimension adapts during the sweeps (growth where the environment demands it,
# truncation where the scheme caps it). The raw singular values are absorbed into the
# sweep direction WITHOUT normalization (as in the single-site ALS gauge moves): the
# data carries the scale of the local targets through the sweeps, and their normalized
# spectra are recorded as the bond Schmidt values. Each driver ends by attaching the
# operand external scales and folding the center tensor's norm (the total data norm)
# into the `scaling` field — the output is canonical (isometric sites, initialized
# Schmidt values) and its represented value equals the strict operation up to the sweep
# truncations. The convergence measure is the norm of the two-site optimal target,
# monotone in processing time, exactly as in the single-site sweeps.

# ---------- generic two-site re-split ----------

# re-split the optimal two-site block of a rank-3 bra (MPS case): block legs
# (bL, p1, p2, bR); the raw singular values are absorbed into the sweep direction
# without normalization (the data keeps the scale of the local target, as in the
# single-site ALS gauge moves), and their normalized spectrum is recorded as the bond's
# Schmidt values
function _als2_update!(bra, s::Integer, t2::AbstractArray{T,4}, alg::TwoSiteUpdate;
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
function _als2_update!(bra, s::Integer, t2::AbstractArray{T,6}, alg::TwoSiteUpdate;
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

# ---------- shared two-site overlap block (add / compress) ----------

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
