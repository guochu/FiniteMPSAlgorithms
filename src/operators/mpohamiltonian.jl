# MPOHamiltonian: a Hamiltonian stored as a chain of SchurMPOTensor site tensors
# (matrix-of-matrices form), aligned with MPSKit's MPOHamiltonian{<:JordanMPOTensor}
# and TEMPO's Schur-form MPOHamiltonian.

"""
	MPOHamiltonian{T<:Number}
	MPOHamiltonian(data::AbstractVector{<:SchurMPOTensor})
	MPOHamiltonian(data::Vector{<:Matrix})
	MPOHamiltonian(L::Int, terms::OpTerm...)

An MPO representation of a Hamiltonian whose site tensors are matrices of local `d×d`
operators (sparse matrix-of-matrices form). The first site tensor is the first row of
the chain, the last site the last column; the interior encodes started operator strings.

Each site tensor carries **its own local dimension** (`phydim` inferred from its stored
blocks), so a lattice with differing per-site dimensions is supported: build the terms on
an [`OpSum(ds)`](@ref) carrying the per-site dimensions, and every operator is validated
against its own `ds[pos]` (an [`OpTerm`](@ref) itself only requires square operators).
`MPOHamiltonian(L, terms...)` carries no lattice and therefore needs one common dimension.

Assemble from product terms with [`OpTerm`](@ref) or from a vector of block matrices;
convert to the dense 4-index chain with `MPO(h)` / [`tompotensors`](@ref).
This is the standard operator input of `ground_state`, `excited_state` and TDVP1.
"""
struct MPOHamiltonian{T<:Number} <: AbstractMPO{T}
	data::Vector{SchurMPOTensor{T}}

	function MPOHamiltonian(data::Vector{SchurMPOTensor{T}}) where {T<:Number}
		isempty(data) && throw(ArgumentError("empty MPOHamiltonian"))
		(size(data[1], 1) == size(data[end], 2)) ||
			throw(DimensionMismatch("boundary dimension mismatch: $(size(data[1],1)) != $(size(data[end],2))"))
		for i in 1:length(data)-1
			(size(data[i], 2) == size(data[i+1], 1)) ||
				throw(DimensionMismatch("chain dimension mismatch at bond $i"))
		end
		return new{T}(data)
	end
end

# from sparse site tensors
MPOHamiltonian(data::AbstractVector{<:SchurMPOTensor{T}}) where {T<:Number} =
	MPOHamiltonian(convert(Vector{SchurMPOTensor{T}}, data))
# from a vector of block matrices
MPOHamiltonian(data::Vector{<:Matrix}) = MPOHamiltonian([SchurMPOTensor(c) for c in data])

MPOHamiltonian(h::MPOHamiltonian) = h

MPO(h::MPOHamiltonian) = MPO(tompotensors(h))

Base.getindex(h::MPOHamiltonian, i::Int, j::Int, k::Int) = h[i][j, k]
Base.copy(h::MPOHamiltonian) = MPOHamiltonian(copy(h.data))
Base.complex(h::MPOHamiltonian{T}) where {T} =
	T <: Complex ? h : MPOHamiltonian([complex(h[i]) for i in 1:length(h)])

function Base.show(io::IO, h::MPOHamiltonian)
	print(io, "MPOHamiltonian{", scalartype(h), "} with ", length(h), " sites, block size = ",
		size(h[1], 1), "×", size(h[1], 2))
end

# finite-chain channel convention: the left boundary vector selects the vacuum channel
# (row 1), the right boundary the closing channel (last column), matching tompotensors'
# row/col selection.
_leftrow(::MPOHamiltonian) = 1
_rightcol(h::MPOHamiltonian) = size(h[end], 2)

function l_LL(ψA::AbstractMPS, h::MPOHamiltonian, ψB::AbstractMPS)
	T = promote_type(scalartype(ψA), scalartype(h), scalartype(ψB))
	v = zeros(T, space_l(ψA), space_l(h), space_l(ψB))
	v[:, _leftrow(h), :] .= one(T)
	return v
end
function r_RR(ψA::AbstractMPS, h::MPOHamiltonian, ψB::AbstractMPS)
	T = promote_type(scalartype(ψA), scalartype(h), scalartype(ψB))
	v = zeros(T, space_r(ψA), space_r(h), space_r(ψB))
	v[:, _rightcol(h), :] .= one(T)
	return v
end

# ---------- sparse -> dense conversion (ported from TEMPO def.jl) ----------

"""
	tompotensors(h::MPOHamiltonian)

Convert an `MPOHamiltonian` to the list of dense 4-index site tensors of a finite
[`MPO`](@ref): the vacuum row (channel 1) is kept at the first site, the closing column
(last channel) at the last site.
"""
tompotensors(h::MPOHamiltonian) = _tompotensors(h, 1, size(h[end], 2))

function _tompotensors(h::MPOHamiltonian, leftrow::Int, rightcol::Int)
	L = length(h)
	(L >= 2) || throw(ArgumentError("size of MPO must at least be 2"))
	T = scalartype(h)
	mpotensors = Vector{Array{T,4}}(undef, L)
	dj = phydim(h[1])
	tmp = zeros(T, 1, dj, size(h[1], 2), dj)
	for i in 1:size(h[1], 2)
		tmp[1, :, i, :] = h[1, leftrow, i]
	end
	mpotensors[1] = tmp
	for n in 2:L-1
		mpotensors[n] = tompotensor(h[n])
	end
	dj = phydim(h[L])
	tmp = zeros(T, size(h[L], 1), dj, 1, dj)
	for i in 1:size(h[L], 1)
		tmp[i, :, 1, :] = h[L, i, rightcol]
	end
	mpotensors[L] = tmp
	return mpotensors
end

# ---------- term-based Schur-form construction ----------

"""
	MPOHamiltonian(L::Int, terms::OpTerm...)
	MPOHamiltonian(ds::AbstractVector{<:Integer}, terms::OpTerm...)
	MPOHamiltonian(terms::OpSum)

Assemble the Schur-form (Jordan upper-triangular) MPO Hamiltonian from the product
terms `terms` (given individually on `L` sites, on a lattice `ds` with possibly differing
per-site dimensions, or as a validated [`OpSum`](@ref)). Each multi-site term gets its own
private chain of channels (one per partially-completed operator string): logical channel 1
is the vacuum identity and the last channel the closing identity; on-site terms accumulate
in the `D` corner block. Site tensors are built directly in the `A`/`B`/`C`/`D` block
storage: interior sites are square `n×n` Jordan blocks, the first site keeps the vacuum row
(`1×n`) and the last site the closing column (`n×1`).

The `L` form carries no lattice and therefore requires a **uniform** local dimension: every
operator of every term must share one `d` (otherwise it throws a `DimensionMismatch`). Use
`MPOHamiltonian(ds, terms...)` or `MPOHamiltonian(OpSum(ds))` for a lattice with differing
per-site dimensions, where every operator is validated against its own `ds[pos]`.
"""
MPOHamiltonian(terms::OpSum) = _mpohamiltonian_from_terms(terms.ds, terms.data)
MPOHamiltonian(L::Int, terms::OpTerm...) = _mpohamiltonian_from_terms(L, collect(terms))
MPOHamiltonian(ds::AbstractVector{<:Integer}, terms::OpTerm...) =
	_mpohamiltonian_from_terms(collect(Int, ds), collect(terms))

# `MPOHamiltonian(L, terms...)` carries no lattice: every operator must then share one
# dimension (use `MPOHamiltonian(ds, terms...)` or `OpSum(ds)` for a lattice with
# differing per-site dimensions)
function _mpohamiltonian_from_terms(L::Int, terms::AbstractVector{<:OpTerm})
	isempty(terms) && throw(ArgumentError("no terms given"))
	d = size(terms[1].operators[1], 1)
	for t in terms, op in t.operators
		(size(op, 1) == d) ||
			throw(DimensionMismatch("MPOHamiltonian(L, terms...) requires one local dimension; " *
									"build an OpSum(ds) instead for a lattice with differing " *
									"per-site dimensions"))
	end
	return _mpohamiltonian_from_terms(fill(d, L), terms)
end

function _mpohamiltonian_from_terms(ds::AbstractVector{Int}, terms::AbstractVector{<:OpTerm})
	L = length(ds)
	isempty(terms) && throw(ArgumentError("no terms given"))
	for t in terms
		all(1 .<= t.positions .<= L) || throw(ArgumentError("term positions out of range"))
		for (pos, op) in zip(t.positions, t.operators)
			(size(op, 1) == ds[pos]) ||
				throw(DimensionMismatch("operator dimension $(size(op, 1)) does not match the " *
										"lattice dimension ds[$pos] = $(ds[pos])"))
		end
	end
	# a single scalar type across all sites (terms may mix real and complex operators)
	T = Float64
	for t in terms
		T = promote_type(T, typeof(t.coeff))
		for op in t.operators
			T = promote_type(T, scalartype(op))
		end
	end

	# per-term channel allocation: term a with n_a operators occupies interior channels
	# bases[a] : bases[a]+n_a-2 (one per partially applied string); logical channels
	# 1 = vacuum and n = done are the implicit identity corners
	bases = Vector{Int}(undef, length(terms))
	nextch = 1
	for (a, t) in enumerate(terms)
		bases[a] = nextch + 1
		nextch += length(t.positions) - 1
	end
	n = nextch + 1

	tensors = Vector{SchurMPOTensor{T}}(undef, L)
	for s in 1:L
		# full logical shape at every site: MPOHamiltonian does not handle chain
		# boundaries — the vacuum row / closing column of the boundary sites are
		# selected downstream by tompotensors
		Os = fill!(Array{Union{Matrix{T}, T}, 2}(undef, n, n), zero(T))
		Os[1, 1] = Os[n, n] = one(T)   # the implied identity corners
		# terms may carry a narrower scalar type than the unified Hamiltonian type T
		asM = v -> convert(Matrix{T}, v)
		for (a, t) in enumerate(terms)
			nt = length(t.positions)
			j = findfirst(==(s), t.positions)
			if j === nothing
				# idle propagation of a started string across a gap: an identity on the
				# interior block diagonal
				for k in 1:nt-1
					if t.positions[k] < s < t.positions[k+1]
						li = bases[a] + k - 2
						Os[li+1, li+1] = _add_block(Os[li+1, li+1], isometry(T, ds[s]))
					end
				end
			elseif nt == 1
				# on-site term: vacuum -> closing corner
				Os[1, n] = _add_block(Os[1, n], asM(t.coeff * t.operators[1]))
			elseif j == 1
				# string start: vacuum -> interior
				li = bases[a] - 1
				Os[1, li+1] = _add_block(Os[1, li+1], asM(t.coeff * t.operators[1]))
			elseif j == nt
				# string end: interior -> closing
				li = bases[a] + nt - 3
				Os[li+1, n] = _add_block(Os[li+1, n], asM(t.operators[end]))
			else
				# interior transition of the string
				r = bases[a] + j - 3
				c = bases[a] + j - 2
				Os[r+1, c+1] = _add_block(Os[r+1, c+1], asM(t.operators[j]))
			end
		end
		tensors[s] = SchurMPOTensor{T}(Os, ds[s])
	end
	return MPOHamiltonian(tensors)
end

_add_block(old, v) = (isa(old, Number) ? old * isometry(scalartype(v), size(v, 1)) : old) + v

# backward-compatible alias (block-sparse MPO is the only MPO Hamiltonian form)
const SparseMPOHamiltonian = MPOHamiltonian
