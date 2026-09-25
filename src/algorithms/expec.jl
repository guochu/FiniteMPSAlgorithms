# expectation values and their normalized counterparts.
#
# NOTE on the external scale: the represented state/operator includes the per-site
# `scaling` factor raised to the L-th power. `expectation` reports the represented value
# and therefore materializes scaling^L (or scaling^2L) factors — it is intended as a
# DEBUG tool for small systems and moderate scales, and may overflow otherwise.
# `expectationvalue` is computed from raw (scale-free) transfer contractions in which
# the same scaling power appears in numerator and denominator and cancels exactly, so
# it stays finite and accurate for arbitrarily large scalings and long chains.

"""
	expectation(ψA::CanonicalMPS, h::AbstractMPO, ψB::CanonicalMPS) -> Number

The matrix element `⟨ψA|h|ψB⟩` of the represented chains (no normalization; use
[`expectationvalue`](@ref) for the normalized value).

!!! warning "debug tool"
	The result includes the external scale, i.e. factors `scaling^L` per chain side.
	These powers overflow for large scalings or long chains, so `expectation` is meant
	for debugging on small systems; use [`expectationvalue`](@ref) instead.
"""
function expectation(ψA::CanonicalMPS, h::AbstractMPO, ψB::CanonicalMPS)
	(length(ψA) == length(h) == length(ψB)) || throw(ArgumentError("dimension mismatch"))
	sA = scaling(ψA)
	sB = scaling(ψB)
	sh = h isa CanonicalMPO ? scaling(h) : one(sA * sB)
	hold = l_LL(ψA, h, ψB)
	for i in 1:length(h)
		hold = (sA * sB * sh) * _updateleft(hold, ψA[i], h[i], ψB[i])
	end
	# contract with the right boundary vector: for block-sparse chains it selects the
	# closing column (a dense MPO chain has a single column, so this is its scalar too)
	r = r_RR(ψA, h, ψB)
	@tensor val = conj(r[a, w, c]) * hold[a, w, c]
	return val
end

"""
	expectation(h::AbstractMPO, ψ::CanonicalMPS) -> Number
	expectation(h::AbstractMPO, ρ::CanonicalMPO) -> Number

The (unnormalized) expectation value of the represented chains: `⟨ψ|h|ψ⟩` or `tr(h·ρ)`
(the mixed case via the crossed physical-index contraction).

!!! warning "debug tool"
	The result includes the external scale (`scaling^L` powers), which may overflow for
	large scalings or long chains. `expectation` is meant for debugging on small
	systems; use [`expectationvalue`](@ref) instead.
"""
expectation(h::AbstractMPO, ψ::CanonicalMPS) = expectation(ψ, h, ψ)

function expectation(h::AbstractMPO, ρ::CanonicalMPO)
	(length(h) == length(ρ)) || throw(ArgumentError("dimension mismatch"))
	s = scaling(ρ)
	sh = h isa CanonicalMPO ? scaling(h) : one(s)
	T = promote_type(scalartype(h), scalartype(ρ))
	c = ones(T, 1, 1)
	for i in eachindex(h)
		Wh = h[i]
		Wρ = ρ[i]
		@tensor c2[-1, -2] := c[1, 2] * Wh[1, 3, -1, 5] * Wρ[2, 5, -2, 3]
		c = sh * s * c2
	end
	return scalar(c)
end

"""
	expectation(op::OpTerm, ψ::CanonicalMPS) -> Number

The matrix element `⟨ψ|op|ψ⟩` (unnormalized; see [`expectationvalue`](@ref)). Exploits the
mixed canonical form of `ψ`: on both sides of the term's support the environments collapse
(identity on the right, `Diagonal(s²)` on the left), so only the support sites are
contracted (sites in between the term's operators pass the environment through unchanged).

!!! warning "debug tool"
	The result includes the external scale (`scaling^2L`), which may overflow for large
	scalings or long chains. `expectation` is meant for debugging on small systems; use
	[`expectationvalue`](@ref) instead.
"""
function expectation(op::OpTerm, ψ::CanonicalMPS)
	L = length(ψ)
	firstpos, lastpos = first(op.positions), last(op.positions)
	(1 <= firstpos && lastpos <= L) || throw(BoundsError())
	svectors_uninitialized(ψ) && canonicalize!(ψ)
	s2 = scaling(ψ)^2
	# right environment of the support: identity (right-canonical form); the per-site
	# s² (bra·ket) is folded into every contracted site
	e = r_RR(ψ, ψ)   # (1, 1)
	for i in L:-1:lastpos+1
		e = s2 * _updateright(e, ψ[i], ψ[i])
	end
	# walk down through the support, inserting the term's operators (bra side conjugated)
	for i in lastpos:-1:firstpos
		j = findfirst(==(i), op.positions)
		if j === nothing
			e = s2 * _updateright(e, ψ[i], ψ[i])
		else
			A = op.operators[j]
			@tensor e2[-1, -2] := conj(ψ[i][-1, -3, 1]) * A[-3, -4] * ψ[i][-2, -4, 2] * e[1, 2]
			e = s2 * e2
		end
	end
	# left reduced environment: Diagonal(s²) at bond firstpos-1; the per-site s² of the
	# sites left of the support is folded into the weight vector incrementally (the
	# total applied scale is scaling^(2L) — one power per side, L sites each)
	s = ψ.s[firstpos]
	ismissing(s) && throw(ArgumentError("Schmidt values left of site $firstpos are not initialized"))
	w = abs2.(s)
	for _ in 1:firstpos-1
		w .*= s2
	end
	return op.coeff * dot(w, diag(e))
end

# crossed-physical-index contraction of `op` into `ρ` between the pre-computed trace
# environments `left` and `right` (flat vectors, reshaped on entry); the per-site
# `scaling` of `ρ` is folded into every contracted support site
function _op_trace_raw(op::OpTerm, ρ::CanonicalMPO, left::AbstractVector,
					   right::AbstractVector)
	T = promote_type(scalartype(op), scalartype(ρ))
	d = phydim(ρ[1])
	di = Matrix{T}(I, d, d)   # identity on the physical legs (sites outside the support)
	s = scaling(ρ)
	firstpos, lastpos = first(op.positions), last(op.positions)
	e = reshape(left, 1, :)
	for i in firstpos:lastpos
		j = findfirst(==(i), op.positions)
		Wρ = ρ[i]
		if j === nothing
			@tensor e2[-1, -2] := e[-1, 2] * Wρ[2, 3, -2, 4] * di[3, 4]
		else
			O = op.operators[j]
			@tensor e2[-1, -2] := e[-1, 2] * O[3, 5] * Wρ[2, 5, -2, 3]
		end
		e = s * e2
	end
	r = reshape(right, 1, :)
	return @tensor val = e[1, 2] * r[1, 2]
end

"""
	TraceCache(ρ::CanonicalMPO)

Pre-computed physical-trace environments of the density-matrix chain `ρ` (the crossed
transfer contractions with an identity on the physical legs, built from both chain
ends; stored as flat vectors, `left[k]` / `right[k]` being the environment at bond `k-1`,
to the left / right of site `k`; the per-site `scaling` of `ρ` is folded in site by
site — no `scaling^L` power is ever materialized). Passing a cache to
`expectation(op, ρ, cache)` / `expectationvalue(op, ρ, cache)` avoids rebuilding the
identity transfer for every observable of a fixed `ρ`. The cache keeps a reference to
`ρ`; the callers pass `ρ` again and it is checked with `===` against the cached one.
"""
struct TraceCache{V<:CanonicalMPO, T<:Number}
	ρ::V
	left::Vector{Vector{T}}
	right::Vector{Vector{T}}
end
function TraceCache(ρ::CanonicalMPO)
	T = scalartype(ρ)
	d = phydim(ρ[1])
	di = Matrix{T}(I, d, d)
	s = scaling(ρ)
	L = length(ρ)
	left = Vector{Vector{T}}(undef, L + 1)
	left[1] = ones(T, size(ρ[1], 1))
	for i in 1:L
		c = reshape(left[i], 1, :)
		@tensor c2[-1, -2] := c[-1, 2] * ρ[i][2, 3, -2, 4] * di[3, 4]
		left[i+1] = vec(s * c2)
	end
	right = Vector{Vector{T}}(undef, L + 1)
	right[L+1] = ones(T, size(ρ[L], 3))
	for i in L:-1:1
		c = reshape(right[i+1], :, 1)
		@tensor c2[-1, -2] := ρ[i][-1, 3, 2, 4] * di[3, 4] * c[2, -2]
		right[i] = vec(s * c2)
	end
	return TraceCache{typeof(ρ), T}(ρ, left, right)
end

"""
	expectation(op::OpTerm, ρ::CanonicalMPO) -> Number
	expectation(op::OpTerm, ρ::CanonicalMPO, cache::TraceCache) -> Number

The (unnormalized) expectation value in the mixed state `ρ`: `tr(op·ρ)` including
`scaling`, contracted through the crossed physical indices. Sites outside the term's
support only propagate the `ρ` transfer (physical trace). Passing a [`TraceCache`](@ref)
reuses the pre-computed trace environments of `ρ` across many observables.

!!! warning "debug tool"
	The result includes the external scale (`scaling^L`), which may overflow for large
	scalings or long chains. `expectation` is meant for debugging on small systems; use
	[`expectationvalue`](@ref) instead.
"""
expectation(op::OpTerm, ρ::CanonicalMPO) = expectation(op, ρ, TraceCache(ρ))
function expectation(op::OpTerm, ρ::CanonicalMPO, cache::TraceCache)
	(ρ === cache.ρ) || throw(ArgumentError("the cache was built for a different ρ"))
	firstpos, lastpos = first(op.positions), last(op.positions)
	(1 <= firstpos && lastpos <= length(ρ)) || throw(BoundsError())
	# the cache environments and the support walk carry the per-site `scaling` already
	return op.coeff * _op_trace_raw(op, ρ, cache.left[firstpos], cache.right[lastpos+1])
end

# raw (scale-free) trace of a CanonicalMPO
function _trace_raw(ρ::CanonicalMPO)
	v = ones(scalartype(ρ), 1)
	for i in eachindex(ρ)
		v = _updatetraceleft(v, ρ[i])
	end
	return scalar(v)
end

"""
	expectationvalue(h::AbstractMPO, ψ::CanonicalMPS) -> Number
	expectationvalue(h::AbstractMPO, ρ::CanonicalMPO) -> Number
	expectationvalue(op::OpTerm, ψ::CanonicalMPS) -> Number
	expectationvalue(op::OpTerm, ρ::CanonicalMPO, [cache::TraceCache]) -> Number

The normalized expectation value: `⟨ψ|h|ψ⟩ / ⟨ψ|ψ⟩` (pure states), `tr(h·ρ) / tr(ρ)`
(mixed states), or the corresponding term ratios. Computed from raw (scale-free)
transfer contractions: the external `scaling` factors of the chains `ψ` / `ρ` appear
with the same power in numerator and denominator and cancel exactly, so
`expectationvalue` is stable even when the represented `scaling^L` powers would
overflow (unlike [`expectation`](@ref)); the `scaling^L` of a `CanonicalMPO` operator
`h` is part of the represented value `h` and is kept.
"""
function expectationvalue(h::AbstractMPO, ψ::CanonicalMPS)
	(length(h) == length(ψ)) || throw(ArgumentError("dimension mismatch"))
	sh = h isa CanonicalMPO ? scaling(h) : 1.0
	hold = l_LL(ψ, h, ψ)
	for i in 1:length(h)
		hold = sh * _updateleft(hold, ψ[i], h[i], ψ[i])
	end
	r = r_RR(ψ, h, ψ)
	@tensor num = conj(r[a, w, c]) * hold[a, w, c]
	return num / _dot(ψ, ψ)
end

function expectationvalue(h::AbstractMPO, ρ::CanonicalMPO)
	(length(h) == length(ρ)) || throw(ArgumentError("dimension mismatch"))
	sh = h isa CanonicalMPO ? scaling(h) : 1.0
	T = promote_type(scalartype(h), scalartype(ρ))
	c = ones(T, 1, 1)
	for i in eachindex(h)
		Wh = h[i]
		Wρ = ρ[i]
		@tensor c2[-1, -2] := c[1, 2] * Wh[1, 3, -1, 5] * Wρ[2, 5, -2, 3]
		c = sh * c2
	end
	return scalar(c) / _trace_raw(ρ)
end

function expectationvalue(op::OpTerm, ψ::CanonicalMPS)
	L = length(ψ)
	firstpos, lastpos = first(op.positions), last(op.positions)
	(1 <= firstpos && lastpos <= L) || throw(BoundsError())
	svectors_uninitialized(ψ) && canonicalize!(ψ)
	e = r_RR(ψ, ψ)   # (1, 1)
	for i in L:-1:lastpos+1
		e = _updateright(e, ψ[i], ψ[i])
	end
	for i in lastpos:-1:firstpos
		j = findfirst(==(i), op.positions)
		if j === nothing
			e = _updateright(e, ψ[i], ψ[i])
		else
			A = op.operators[j]
			@tensor e2[-1, -2] := conj(ψ[i][-1, -3, 1]) * A[-3, -4] * ψ[i][-2, -4, 2] * e[1, 2]
			e = e2
		end
	end
	s = ψ.s[firstpos]
	ismissing(s) && throw(ArgumentError("Schmidt values left of site $firstpos are not initialized"))
	return op.coeff * dot(abs2.(s), diag(e)) / _dot(ψ, ψ)
end

function expectationvalue(op::OpTerm, ρ::CanonicalMPO, cache::TraceCache = TraceCache(ρ))
	(ρ === cache.ρ) || throw(ArgumentError("the cache was built for a different ρ"))
	firstpos, lastpos = first(op.positions), last(op.positions)
	(1 <= firstpos && lastpos <= length(ρ)) || throw(BoundsError())
	num = op.coeff * _op_trace_raw(op, ρ, cache.left[firstpos], cache.right[lastpos+1])
	# the raw trace of ρ is the single entry of the completed left environment
	return num / only(cache.left[length(ρ)+1])
end
