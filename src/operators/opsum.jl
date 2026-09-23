# OpTerm: a product operator term with support on specific sites.
# OpSum: a validated collection of OpTerms on a shared lattice.

"""
	OpTerm(coeff, positions, operators)
	OpTerm(coeff, i₁ => O₁, i₂ => O₂, ...)

A product operator term `coeff * O₁ ⊗ O₂ ⋯` acting on the strictly increasing site
positions `positions`.
"""
struct OpTerm{N,O<:AbstractMatrix,T<:Number}
	coeff::T
	positions::NTuple{N,Int}
	operators::NTuple{N,O}
	function OpTerm(coeff::Number, positions::NTuple{N,Int}, operators::NTuple{N,<:AbstractMatrix}) where {N}
		(N > 0) || throw(ArgumentError("term must act on at least one site"))
		for i in 2:N
			(positions[i] > positions[i-1]) || throw(ArgumentError("positions must be strictly increasing"))
		end
		d = size(operators[1], 1)
		for op in operators
			(size(op, 1) == size(op, 2) == d) || throw(DimensionMismatch("all operators of a term must share the physical dimension"))
		end
		T = promote_type(typeof(coeff), scalartype(operators[1]))
		for op in operators
			T = promote_type(T, scalartype(op))
		end
		ops = map(op -> convert(Matrix{T}, op), operators)
		return new{N,Matrix{T},T}(convert(T, coeff), positions, ops)
	end
end

scalartype(::Type{<:OpTerm{N, O, T}}) where {N, O, T} = T

OpTerm(coeff::Number, positions::AbstractVector{Int}, operators::AbstractVector{<:AbstractMatrix}) =
	OpTerm(coeff, Tuple(positions), Tuple(operators))
OpTerm(coeff::Number, pairs::Pair{Int,<:AbstractMatrix}...) =
	OpTerm(coeff, Tuple(first.(pairs)), Tuple(last.(pairs)))

"""
	term(coeff, i₁ => O₁, ...)
	term(i₁ => O₁, ...)

Convenience constructor of [`OpTerm`](@ref) (coefficient defaults to 1).
"""
term(coeff::Number, pairs::Pair{Int,<:AbstractMatrix}...) = OpTerm(coeff, pairs...)
term(pairs::Pair{Int,<:AbstractMatrix}...) = OpTerm(one(Float64), pairs...)

"""
	OpSum(ds)
	OpSum(ds, terms)

A collection of [`OpTerm`](@ref)s on a lattice with physical dimensions `ds::Vector{Int}`.
The terms are validated against `ds` on construction / `push!` (positions in range,
every operator matching the local dimension), so an `OpSum` is self-consistent by
construction; it is the input of `MPOHamiltonian(::OpSum)`.
"""
struct OpSum
	data::Vector{OpTerm}
	ds::Vector{Int}
end
OpSum(ds::AbstractVector{Int}) = OpSum(OpTerm[], ds)
function OpSum(ds::AbstractVector{Int}, terms::AbstractVector{<:OpTerm})
	s = OpSum(ds)
	for t in terms
		push!(s, t)
	end
	return s
end
OpSum(ds::AbstractVector{Int}, terms::OpTerm...) = OpSum(ds, collect(terms))

Base.length(s::OpSum) = length(s.data)
Base.iterate(s::OpSum, state...) = iterate(s.data, state...)
Base.getindex(s::OpSum, i::Int) = s.data[i]

function Base.push!(s::OpSum, t::OpTerm)
	L = length(s.ds)
	for (pos, op) in zip(t.positions, t.operators)
		(1 <= pos <= L) ||
			throw(ArgumentError("term position $pos out of range for $(L) sites"))
		(size(op, 1) == s.ds[pos]) ||
			throw(DimensionMismatch("operator dimension $(size(op, 1)) does not match ds[$pos] = $(s.ds[pos])"))
	end
	push!(s.data, t)
	return s
end
