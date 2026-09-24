# long-range operators with exponentially decaying couplings, in Schur (W-form) MPO
# representation (aligned with TEMPO's schurmpo exponential-decay operators).
#
#   ExpDecayOpTerm(a, m, b, α, λ)  represents the operator
#       O = α Σ_{i<j} λ^(j-i) · a_i m_(i+1) m_(i+2) ⋯ m_(j-1) b_j ,
#   i.e. the operator `a` acts at the start site `i`, `b` at the end site `j`, the
#   propagator `m` acts on every site in between, and each propagation step contributes
#   a decay factor `λ` (total λ^(j-i), so the term strength decays exponentially with
#   the distance j - i). The whole sum carries an overall strength `α`.
#
#   ExpDecayOpSum(a, m, b, αs, λs) is the sum of |αs| such terms, one per parameter
#   pair (αs[k], λs[k]).
#
# Both admit an exact W-form (SchurMPOTensor) representation with one internal channel
# per decay parameter: the channel is opened by `C = α·λ·a`, propagated as `A = λ·m`,
# and closed by `B = b`; each channel therefore accumulates exactly the factor
# α·λ^(j-i)·m^(j-i-1) between the two endpoints.

"""
	ExpDecayOpTerm(a, m, b, α, λ)

An exponentially decaying long-range operator term
`O = α Σ_{i<j} λ^(j-i) · a_i m_(i+1) ⋯ m_(j-1) b_j` (see the module docstring). `a`,
`m`, `b` are `d×d` matrices, `λ` the decay factor per lattice spacing and `α` the
overall strength. Converts directly to a [`SchurMPOTensor`](@ref).
"""
struct ExpDecayOpTerm{M<:AbstractMatrix,T<:Number}
	a::M
	m::M
	b::M
	α::T
	λ::T
	function ExpDecayOpTerm(a::AbstractMatrix, m::AbstractMatrix, b::AbstractMatrix,
							α::Number, λ::Number)
		d = size(a, 1)
		(size(a, 2) == size(m, 1) == size(m, 2) == size(b, 1) == size(b, 2) == d) ||
			throw(DimensionMismatch("a, m, b must be square matrices of equal size"))
		T = promote_type(typeof(α), typeof(λ), scalartype(a), scalartype(m), scalartype(b))
		return new{Matrix{T},T}(convert(Matrix{T}, a), convert(Matrix{T}, m),
			convert(Matrix{T}, b), convert(T, α), convert(T, λ))
	end
end

scalartype(::Type{<:ExpDecayOpTerm{M,T}}) where {M,T} = T
phydim(t::ExpDecayOpTerm) = size(t.a, 1)

"""
	ExpDecayOpSum(a, m, b, αs, λs)

The sum of the exponentially decaying terms [`ExpDecayOpTerm(a, m, b, αs[k], λs[k])`](@ref
ExpDecayOpTerm) over all parameter pairs (αs[k], λs[k]). Converts directly to a
[`SchurMPOTensor`](@ref) with one internal channel per pair.
"""
struct ExpDecayOpSum{M<:AbstractMatrix,T<:Number}
	a::M
	m::M
	b::M
	αs::Vector{T}
	λs::Vector{T}
	function ExpDecayOpSum(a::AbstractMatrix, m::AbstractMatrix, b::AbstractMatrix,
						   αs::AbstractVector, λs::AbstractVector)
		d = size(a, 1)
		(size(a, 2) == size(m, 1) == size(m, 2) == size(b, 1) == size(b, 2) == d) ||
			throw(DimensionMismatch("a, m, b must be square matrices of equal size"))
		(length(αs) == length(λs) && !isempty(αs)) ||
			throw(ArgumentError("αs and λs must be non-empty vectors of equal length"))
		T = Float64
		for v in (αs..., λs...)
			T = promote_type(T, typeof(v))
		end
		for M3 in (a, m, b)
			T = promote_type(T, scalartype(M3))
		end
		return new{Matrix{T},T}(convert(Matrix{T}, a), convert(Matrix{T}, m),
			convert(Matrix{T}, b), convert(Vector{T}, collect(αs)),
			convert(Vector{T}, collect(λs)))
	end
end

ExpDecayOpSum(t::ExpDecayOpTerm) = ExpDecayOpSum(t.a, t.m, t.b, [t.α], [t.λ])
scalartype(::Type{<:ExpDecayOpSum{M,T}}) where {M,T} = T
phydim(s::ExpDecayOpSum) = size(s.a, 1)

# the W-form site tensor of an ExpDecayOpSum: n internal channels (one per parameter
# pair), C[k] = αk·λk·a (start, carrying the first decay factor), A[k,k] = λk·m (each
# further propagation step), B[k] = b (close); a term spanning i < j therefore
# accumulates αk·λk^(j-i) · a ⊗ m^⊗(j-i-1) ⊗ b exactly. The optional on-site operator
# `hloc` accumulates in the D corner (every site). The boundary sites keep only the
# vacuum row (the starts) and the closing column (the closes) — the
# `MPOHamiltonian(::Vector{<:SchurMPOTensor})` constructor handles this through the
# row/column selection of `tompotensors`.
function SchurMPOTensor(s::ExpDecayOpSum{M,T}, hloc::AbstractMatrix) where {M,T}
	# (M = Matrix{T} by construction, so the Schur tensor is parameterized by T alone)
	d = phydim(s)
	n = length(s.αs)
	(size(hloc, 1) == size(hloc, 2) == d) ||
		throw(DimensionMismatch("hloc must be a $(d)×$(d) matrix"))
	logical = Matrix{Any}(undef, n + 2, n + 2)
	for I in eachindex(logical)
		logical[I] = 0.0
	end
	for k in 1:n
		logical[1, k+1] = s.αs[k] * s.λs[k] * s.a    # C: vacuum -> channel k
		logical[k+1, k+1] = s.λs[k] * s.m            # A: propagate channel k
		logical[k+1, n+2] = s.b                      # B: channel k -> closing
	end
	logical[1, n+2] = hloc                           # D: on-site term
	return SchurMPOTensor{T}(logical)
end
SchurMPOTensor(t::ExpDecayOpTerm, hloc::AbstractMatrix) = SchurMPOTensor(ExpDecayOpSum(t), hloc)

"""
	MPOHamiltonian(s::ExpDecayOpSum, L::Int, [hloc]) -> MPOHamiltonian
	MPOHamiltonian(t::ExpDecayOpTerm, L::Int, [hloc]) -> MPOHamiltonian

The exponentially decaying operator as an `L`-site Hamiltonian chain (assembled from the
per-site [`SchurMPOTensor`](@ref)s of `s`, with the optional on-site term `hloc` in
every `D` corner).
"""
MPOHamiltonian(s::ExpDecayOpSum, L::Int, hloc::AbstractMatrix = zeros(scalartype(s), phydim(s), phydim(s))) =
	MPOHamiltonian([SchurMPOTensor(s, hloc) for _ in 1:L])
MPOHamiltonian(t::ExpDecayOpTerm, L::Int, hloc::AbstractMatrix = zeros(scalartype(t), phydim(t), phydim(t))) =
	MPOHamiltonian(ExpDecayOpSum(t), L, hloc)

# debug constructor: expand the exact W-form representation into the (typically much
# larger) explicit OpSum of all two-site terms
function OpSum(s::ExpDecayOpSum, L::Int)
	d = phydim(s)
	terms = OpTerm[]
	for k in eachindex(s.αs), i in 1:L-1, j in i+1:L
		positions = collect(i:j)
		operators = [ifelse(t == i, s.a, ifelse(t == j, s.b, s.m)) for t in positions]
		push!(terms, OpTerm(s.αs[k] * s.λs[k]^(j - i), positions, operators))
	end
	return OpSum(fill(d, L), terms)
end
OpSum(t::ExpDecayOpTerm, L::Int) = OpSum(ExpDecayOpSum(t), L)
