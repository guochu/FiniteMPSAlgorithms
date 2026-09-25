# ---------------- physical models on a finite chain ----------------
#
# Convenience builders of the standard one-dimensional Hamiltonians on a **finite open
# chain** of `L` sites, as `MPOHamiltonian`s (the sparse Schur form the algorithms use;
# `MPO(h)` / `todense(h)` give the dense routes). The models and sign conventions mirror
# InfiniteMPSAlgorithms' `models.jl`, which builds the same models on an infinite,
# translation-invariant chain:
#
#   heisenberg_hamiltonian  H = J Σ (SˣSˣ + SʸSʸ + Δ SᶻSᶻ) − h Σ Sᶻ
#   tfim_hamiltonian        H = −J Σ σˣσˣ − h Σ σᶻ
#   fermi_hubbard           H = −t Σσ (c†σ,ᵢcσ,ᵢ₊₁ + h.c.) + U Σ n↑n↓ − μ Σ n
#
# All of them are short-range models on the local space `ℂ²` (spins) or `ℂ⁴` (Hubbard);
# periodic boundaries and longer-range or disordered couplings are left to the caller, who
# can assemble the terms of an `OpSum` directly (see `MPOHamiltonian(::OpSum)`).

# ---- spin-1/2 operators ----

"""
	σx([T]), σy([T]), σz([T]) -> Matrix
	Sx([T]), Sy([T]), Sz([T]) -> Matrix

The Pauli matrices and the spin-1/2 operators (`S = σ/2`), as `2×2` matrices of element
type `T` (default `ComplexF64`). `σx`/`σz`/`Sx`/`Sz` keep `T`; `σy`/`Sy` are promoted to a
complex element type, since they are purely imaginary.
"""
σx(::Type{T}=ComplexF64) where {T<:Number} = Matrix{T}([0 1; 1 0])
σy(::Type{T}=ComplexF64) where {T<:Number} = Matrix{complex(float(T))}([0 -im; im 0])
σz(::Type{T}=ComplexF64) where {T<:Number} = Matrix{T}([1 0; 0 -1])
Sx(::Type{T}=ComplexF64) where {T<:Number} = σx(T) ./ 2
Sy(::Type{T}=ComplexF64) where {T<:Number} = σy(T) ./ 2
Sz(::Type{T}=ComplexF64) where {T<:Number} = σz(T) ./ 2

# ---- spin models ----

"""
	heisenberg_hamiltonian(L::Int; J=1.0, Δ=1.0, h=0.0, T=ComplexF64) -> MPOHamiltonian

Spin-1/2 XXZ chain with a longitudinal field on `L` sites with open boundaries,

	H = J Σᵢ (SˣᵢSˣᵢ₊₁ + SʸᵢSʸᵢ₊₁ + Δ SᶻᵢSᶻᵢ₊₁) − h Σᵢ Sᶻᵢ,

so `Δ = 1` is the isotropic Heisenberg (XXX) chain and `Δ = 0` the XX chain.
"""
function heisenberg_hamiltonian(L::Int; J::Real=1.0, Δ::Real=1.0, h::Real=0.0,
							   T::Type=ComplexF64)
	L >= 2 || throw(ArgumentError("L must be at least 2"))
	sx, sy, sz = Sx(T), Sy(T), Sz(T)
	terms = OpSum(fill(2, L))
	for i in 1:L-1
		push!(terms, OpTerm(J, i => sx, i + 1 => sx))
		push!(terms, OpTerm(J, i => sy, i + 1 => sy))
		push!(terms, OpTerm(J * Δ, i => sz, i + 1 => sz))
	end
	if !iszero(h)
		for i in 1:L
			push!(terms, OpTerm(-h, i => sz))
		end
	end
	return MPOHamiltonian(terms)
end

"""
	tfim_hamiltonian(L::Int; J=1.0, h=1.0, T=ComplexF64) -> MPOHamiltonian

Transverse-field Ising chain on `L` sites with open boundaries,

	H = −J Σᵢ σˣᵢσˣᵢ₊₁ − h Σᵢ σᶻᵢ.

`J = h = 1` is the critical point (the infinite-chain ground-state energy density is
`−4/π`); `h = 0` gives the classical Ising limit.
"""
function tfim_hamiltonian(L::Int; J::Real=1.0, h::Real=1.0, T::Type=ComplexF64)
	L >= 2 || throw(ArgumentError("L must be at least 2"))
	sx, sz = σx(T), σz(T)
	terms = OpSum(fill(2, L))
	for i in 1:L-1
		push!(terms, OpTerm(-J, i => sx, i + 1 => sx))
	end
	if !iszero(h)
		for i in 1:L
			push!(terms, OpTerm(-h, i => sz))
		end
	end
	return MPOHamiltonian(terms)
end

# ---- fermionic model ----

"""
	fermi_hubbard(L::Int; t=1.0, U=0.0, μ=0.0, T=ComplexF64) -> MPOHamiltonian

Fermi-Hubbard chain on `L` sites with open boundaries, local basis
`(|0⟩, |↑⟩, |↓⟩, |↑↓⟩)` (the up factor the major one),

	H = −t Σᵢ Σσ (c†σ,ᵢ cσ,ᵢ₊₁ + h.c.) + U Σᵢ n↑,ᵢn↓,ᵢ − μ Σᵢ nᵢ.

The fermions are Jordan-Wigner transformed with the modes ordered site by site and `↑`
before `↓` within a site, every mode carrying the parity string of the modes that precede
it. The string is split between the local operators (the intra-site part, e.g. the ↑
parity that a ↓ mode carries) and the sites to the left, and for a *nearest-neighbour*
hopping the two inter-site strings collapse onto the parity `P = (−1)^n` of the left site
only — there are no sites in between. One bond term per species is therefore

	c†σ,ᵢ cσ,ᵢ₊₁ = (d†σ,ᵢ Pᵢ) ⊗ dσ,ᵢ₊₁,

with `d` the local mode operators, i.e. a nearest-neighbour term on the 4-dimensional
local space with the dressed creator `d†↑P = σ⁺↑ ⊗ σᶻ↓` for the up fermions and
`d†↓P = I⊗σ⁺↓` for the down ones, and its hermitian conjugate for the reverse hop. On an
open chain no boundary string appears; a periodic Hubbard chain would need one (a global
parity term), so it is not provided here.
"""
function fermi_hubbard(L::Int; t::Real=1.0, U::Real=0.0, μ::Real=0.0,
					   T::Type=ComplexF64)
	L >= 2 || throw(ArgumentError("L must be at least 2"))
	I2 = Matrix{T}(I, 2, 2)
	sm = T[0 1; 0 0]                     # σ⁻ = |0⟩⟨↑|, basis (|0⟩, |↑⟩)
	sz = T[1 0; 0 -1]                    # σᶻ
	# local mode operators in the basis (|0⟩, |↑⟩, |↓⟩, |↑↓⟩), with the intra-site part of
	# the Jordan-Wigner string folded in (the ↑ mode precedes the ↓ mode on a site)
	d_up = kron(sm, I2)                  # ↑ annihilation
	d_dn = kron(sz, sm)                  # ↓ annihilation, carrying the ↑ parity
	P = kron(sz, sz)                     # on-site parity P = (−1)^n
	n_up, n_dn = d_up' * d_up, d_dn' * d_dn
	terms = OpSum(fill(4, L))
	for (dc, d) in ((d_up' * P, d_up), (d_dn' * P, d_dn))
		for i in 1:L-1
			push!(terms, OpTerm(-t, i => dc, i + 1 => d))       # c†σ,ᵢ cσ,ᵢ₊₁
			push!(terms, OpTerm(-t, i => dc', i + 1 => d'))     # its hermitian conjugate
		end
	end
	if !iszero(U)
		for i in 1:L
			push!(terms, OpTerm(U, i => n_up * n_dn))
		end
	end
	if !iszero(μ)
		for i in 1:L
			push!(terms, OpTerm(-μ, i => n_up + n_dn))
		end
	end
	return MPOHamiltonian(terms)
end
