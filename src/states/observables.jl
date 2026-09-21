# entanglement measures of canonical chains (the expectation-value functions live in
# algorithms/expec.jl)

# Schmidt values at `bond` shared by CanonicalMPS and CanonicalMPO (canonicalize if needed)
function _bond_schmidt(x::Union{CanonicalMPS, CanonicalMPO}, bond::Integer)
	(1 <= bond <= length(x) - 1) || throw(BoundsError())
	svectors_uninitialized(x) && canonicalize!(x)
	s = x.s[bond+1]
	ismissing(s) && throw(ArgumentError("Schmidt values at bond $bond are not initialized"))
	return s
end

"""
	entanglement_entropy(x::Union{CanonicalMPS,CanonicalMPO}; bond=div(length(x), 2), α=1)

Rényi entropy `α` of the Schmidt spectrum at `bond` (`α = 1` gives the von Neumann entropy).
Requires the canonical form (initialized Schmidt values).
"""
function entanglement_entropy(x::Union{CanonicalMPS, CanonicalMPO}; bond::Integer=div(length(x), 2), α::Real=1)
	s = _bond_schmidt(x, bond)
	return renyi_entropy(s .^ 2; α)
end

"""
	entanglement_spectrum(x::Union{CanonicalMPS,CanonicalMPO}; bond=div(length(x), 2))

The Schmidt spectrum (squared Schmidt values) at `bond`.
"""
function entanglement_spectrum(x::Union{CanonicalMPS,CanonicalMPO}; bond::Integer=div(length(x), 2))
	s = _bond_schmidt(x, bond)
	return abs2.(s)
end

"""
	schmidt_values(x::Union{CanonicalMPS,CanonicalMPO}; bond=div(length(x), 2))

The Schmidt values at `bond` (equivalent to `x.s[bond+1]`).
"""
function schmidt_values(x::Union{CanonicalMPS,CanonicalMPO}; bond::Integer=div(length(x), 2))
	return _bond_schmidt(x, bond)
end

"""
	todense(ψ::CanonicalMPS) -> Vector

Dense state vector of the MPS. The physical indices are merged with **site 1 the
slowest** index and site `L` the fastest (the Kronecker convention of `kron`-ing the
local basis vectors site by site, `vec(|p1 … pL⟩)`). The chain `scaling` is included as
`scaling(ψ)^L`; the Schmidt vectors are not used (the site tensors carry the full
state).
"""
function todense(ψ::CanonicalMPS)
	L = length(ψ)
	A1 = ψ[1]
	T = reshape(A1, size(A1, 2), size(A1, 3))                      # (p1, a)
	for i in 2:L
		A = ψ[i]
		t3 = reshape(T, :, size(A, 1)) * reshape(A, size(A, 1), :) # (r, p·b)
		# append p_i as the fastest row dimension: rows stay (p1,...,p_i), p1 slowest
		T = reshape(permutedims(reshape(t3, size(T, 1), size(A, 2), size(A, 3)), (2, 1, 3)),
			size(A, 2) * size(T, 1), size(A, 3))
	end
	return vec(T) * scaling(ψ)^L
end
