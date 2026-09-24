# Clifford augmented DMRG (CA-DMRG): ground-state search where the MPS ansatz is
# augmented by a layer of two-site Clifford circuits. At each two-site update the local
# ground state Ψ is obtained, then every two-site Clifford circuit C is applied to Ψ;
# the circuit that minimizes the SVD truncation error (maximizes the retained weight) is
# kept, and the Hamiltonian MPO is transformed as C H C† on those two sites. Because the
# identity circuit is always a candidate, CA-DMRG is at least as accurate as standard
# two-site DMRG.
#
# Reference: L. Fu, H. Shang, J. Yang, C. Guo, "Clifford augmented density matrix
# renormalization group for ab initio quantum chemistry", Phys. Rev. B 112, 195111 (2025)
# (arXiv:2506.16026).

# ---------- algorithm type ----------

"""
	CADMRG(; maxiter=Defaults.maxiter, tol=Defaults.tol, trunc=DefaultTruncation, verbosity=0)

Parameters of the Clifford-augmented two-site DMRG ground-state search. `trunc` is the
truncation scheme applied to the SVD after each two-site update (its bond cap bounds
the bond dimension).
"""
@kwdef struct CADMRG{TR<:TruncationScheme} <: TwoSiteUpdate
	maxiter::Int = Defaults.maxiter
	tol::Float64 = Defaults.tol
	trunc::TR = DefaultTruncation
	verbosity::Int = 0
end

# ---------- two-qubit Clifford group ----------

const _H = ComplexF64[1 1; 1 -1] / sqrt(2)
const _S = ComplexF64[1 0; 0 im]
const _X = ComplexF64[0 1; 1 0]
const _Y = ComplexF64[0 -im; im 0]
const _Z = ComplexF64[1 0; 0 -1]
const _I2 = Matrix{ComplexF64}(I, 2, 2)
const _CNOT12 = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]   # control=1, target=2
const _CNOT21 = ComplexF64[1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]   # control=2, target=1

const _PAULIS_2Q = [kron(p, q) for p in (_I2, _X, _Y, _Z) for q in (_I2, _X, _Y, _Z)]
const _CLIFF_GENS = [kron(_X, _I2), kron(_Z, _I2), kron(_I2, _X), kron(_I2, _Z)]

# Identify the Pauli operator (up to global phase) that Q is closest to, and return its
# index in _PAULIS_2Q. Returns nothing if Q is not close to any Pauli.
function _identify_pauli!(Q, P_list)
	best_idx = 0
	best_diff = Inf
	for (idx, P) in enumerate(P_list)
		for s in (1.0, -1.0, im, -im)
			diff = 0.0
			@inbounds for k in 1:16
				d = abs(Q[k] - s * P[k])
				diff = ifelse(d > diff, d, diff)
			end
			if diff < best_diff
				best_diff = diff
				best_idx = idx
			end
		end
	end
	return best_diff < 1e-6 ? best_idx : nothing
end

# Compute a 16-bit integer key from the 4x4 binary symplectic matrix of C. Two Cliffords
# that differ only by a global phase share the same key, so this enumerates Sp(4,2) (720
# elements). The key is built from the images of X1, Z1, X2, Z2 under conjugation by C.
function _symplectic_key!(C, Q, tmp, Cdag)
	copyto!(Cdag, C')
	F = 0
	for i in 1:4
		mul!(tmp, C, _CLIFF_GENS[i])
		mul!(Q, tmp, Cdag)
		idx = _identify_pauli!(Q, _PAULIS_2Q)
		idx === nothing && return nothing
		p1 = (idx - 1) ÷ 4 + 1
		p2 = (idx - 1) % 4 + 1
		# I=(0,0), X=(1,0), Y=(1,1), Z=(0,1)
		xz = ((0, 0), (1, 0), (1, 1), (0, 1))
		x1, z1 = xz[p1]
		x2, z2 = xz[p2]
		F = F * 16 + (x1 << 3) | (z1 << 2) | (x2 << 1) | z2
	end
	return F
end

"""
	two_qubit_cliffords() -> Vector{Matrix{ComplexF64}}

All 720 projective two-qubit Clifford circuits (the symplectic group Sp(4,2)), generated
by BFS from the identity with generators H⊗I, I⊗H, S⊗I, I⊗S, CNOT(1→2), CNOT(2→1).
Each element acts on the computational basis ordering |00>, |01>, |10>, |11>.
Deduplication uses the exact binary symplectic matrix (action on the Pauli basis) as the
key, which is immune to floating-point drift.
"""
function two_qubit_cliffords()
	gen = [kron(_H, _I2), kron(_I2, _H), kron(_S, _I2), kron(_I2, _S), _CNOT12, _CNOT21]
	group = Dict{Int, Matrix{ComplexF64}}()
	Q = similar(gen[1])
	tmp = similar(gen[1])
	Cdag = similar(gen[1])
	id = Matrix{ComplexF64}(I, 4, 4)
	h_id = _symplectic_key!(id, Q, tmp, Cdag)
	group[h_id] = id
	queue = [id]
	while !isempty(queue)
		cur = popfirst!(queue)
		for g in gen
			nxt = g * cur
			h = _symplectic_key!(nxt, Q, tmp, Cdag)
			if h !== nothing && !haskey(group, h)
				group[h] = nxt
				push!(queue, nxt)
			end
		end
	end
	return collect(values(group))
end

const TWO_QUBIT_CLIFFORDS = two_qubit_cliffords()

# TwoSiteHeff / ac2_prime live in groundstates/dmrg2.jl (shared with DMRG2)

# ---------- Clifford-augmented state types ----------

const _PAULI_LABELS = (:I, :X, :Y, :Z)
const _PAULI_MAP = (I = [1.0 0; 0 1.0], X = [0.0 1; 1 0.0],
	Y = [0.0 -1.0im; 1.0im 0.0], Z = [1.0 0; 0 -1.0])

"""
	CliffordGate

A two-qubit Clifford circuit applied to sites `(site, site + 1)` of a CA-DMRG state.
`C::Matrix{ComplexF64}` is the 4×4 circuit matrix in the computational basis ordered
as |00>, |01>, |10>, |11> (row-major in the two physical indices).
"""
struct CliffordGate
	site::Int
	C::Matrix{ComplexF64}
end

"""
	CAMPS

A CA-DMRG result: the canonical MPS `ket` produced by the algorithm **together with the
recorded Clifford circuits** that were absorbed into the (transformed) MPO. The ket
lives in the Clifford-rotated picture, `|ket⟩ = U|ψ⟩` with `U` the product of all
recorded gates in application order, so observables must be evaluated with
`expectation(::PauliTerm, ::CAMPS)`, which conjugates the Pauli string back through the
circuits instead of using `ket` directly.
"""
struct CAMPS{T<:Number}
	ket::CanonicalMPS{T}
	gates::Vector{CliffordGate}
end

Base.length(c::CAMPS) = length(c.ket)

"""
	PauliTerm(coeff, ops)
	PauliTerm(ops)
	PauliTerm(coeff, ops::AbstractString)

A (weighted) tensor product of single-site Pauli operators, e.g.
`PauliTerm(-1.0, "ZZII")` or `PauliTerm(2.0, [:I, :X, :I, :I])`. `ops[i] ∈ (:I, :X, :Y, :Z)`
acts on site `i`.
"""
struct PauliTerm
	coeff::ComplexF64
	ops::Vector{Symbol}
	function PauliTerm(coeff::Number, ops::Vector{Symbol})
		all(o -> o in _PAULI_LABELS, ops) ||
			throw(ArgumentError("PauliTerm entries must be one of $_PAULI_LABELS"))
		return new(ComplexF64(coeff), ops)
	end
end

PauliTerm(coeff::Number, ops::AbstractString) = PauliTerm(coeff, Symbol.(collect(ops)))
PauliTerm(ops::AbstractString) = PauliTerm(1.0, ops)
PauliTerm(ops::Vector{Symbol}) = PauliTerm(1.0, ops)

Base.length(pt::PauliTerm) = length(pt.ops)

_pauli_matrix(s::Symbol) = _PAULI_MAP[s]

# ---------- cache ----------

"""
	CADMRGCache{M,V,T}

Environment stack for CA-DMRG: single-site environments (reused as in `DMRGCache`)
together with the **recorded Clifford circuits** — every two-site update pushes the
optimal `CliffordGate` it applied onto `gates` (in application order). The ket lives in
the Clifford-rotated picture; the gates are what `expectation(::PauliTerm, ::CAMPS)`
conjugates observables back through.
"""
struct CADMRGCache{M<:AbstractMPO,V<:CanonicalMPS,T}
	H::M
	ket::V
	hstorage::Vector{Array{T,3}}
	gates::Vector{CliffordGate}
end

function CADMRGCache(h::AbstractMPO, ψ::CanonicalMPS)
	L = length(ψ)
	T = promote_type(scalartype(h), scalartype(ψ))
	unset_svectors!(ψ)
	hs = Vector{Array{T,3}}(undef, L + 1)
	hs[1] = l_LL(ψ, h, ψ)
	hs[L+1] = r_RR(ψ, h, ψ)
	for s in L:-1:2
		hs[s] = _updateright(hs[s+1], ψ[s], h[s], ψ[s])
	end
	return CADMRGCache(h, ψ, hs, CliffordGate[])
end

updateleft!(env::CADMRGCache, site::Integer) =
	(env.hstorage[site+1] = _updateleft(env.hstorage[site], env.ket[site], env.H[site], env.ket[site]); env)
updateright!(env::CADMRGCache, site::Integer) =
	(env.hstorage[site] = _updateright(env.hstorage[site+1], env.ket[site], env.H[site], env.ket[site]); env)

# Convert a (possibly sparse) MPO tensor to a dense 4-index array W[wL, p_out, wR, p_in].
function _dense_mpotensor(W::AbstractSparseMPOTensor)
	d = phydim(W)
	wL = size(W, 1); wR = size(W, 2)
	T = scalartype(W)
	Wd = zeros(T, wL, d, wR, d)
	for (i, j) in keys(W)
		O = W[i, j]
		Om = O isa Number ? O * Matrix{T}(I, d, d) : convert(Matrix{T}, O)
		Wd[i, :, j, :] = Om
	end
	return Wd
end
_dense_mpotensor(W::MPOTensor) = W

# Convert an entire MPO to dense MPO form (CA-DMRG rewrites MPO tensors in place).
# Clifford circuits are complex, so the result is always complex.
function _dense_mpo(h::AbstractMPO)
	TC = complex(scalartype(h))
	data = [convert(Array{TC,4}, _dense_mpotensor(h[s])) for s in 1:length(h)]
	return MPO{TC}(data)
end
# Schur-form Hamiltonians keep the full logical shape at every site (no boundary
# collapse), so the dense conversion must select the boundary channels first
_dense_mpo(h::MPOHamiltonian{<:SchurMPOTensor}) = _dense_mpo(MPO(tompotensors(h)))

# ---------- local two-site update with Clifford search ----------

function _cadmrg_local_update!(env::CADMRGCache, s::Integer, alg::CADMRG; move_right::Bool=true)
	L = length(env.ket)
	@assert s < L
	W1, W2 = env.H[s], env.H[s+1]
	heff = TwoSiteHeff(W1, W2, env.hstorage[s], env.hstorage[s+2])
	# initial guess: the two-site tensor formed from the current site tensors
	@tensor guess[aL, p1, p2, aR] := env.ket[s][aL, p1, b] * env.ket[s+1][b, p2, aR]

	vals, vecs, info = eigsolve(y -> ac2_prime(y, heff), guess, 1, :SR;
								ishermitian=true, tol=max(alg.tol, 1e-12),
								krylovdim=30, maxiter=200, eager=true)
	Ψ = vecs[1]
	E = real(vals[1])

	# search over all two-qubit Clifford circuits for the one minimizing SVD truncation
	d = phydim(env.ket[s])
	@assert d == 2 && phydim(env.ket[s+1]) == 2 "CA-DMRG assumes qubit (d=2) sites"

	aL = size(Ψ, 1); aR = size(Ψ, 4)
	# The two-qubit Clifford matrices act on the computational basis ordered as
	# |00>, |01>, |10>, |11> (row-major in (p1, p2)). Julia's column-major reshape of
	# Ψ[aL, p1, p2, aR] would instead order (p2, p1), so we permute to align.
	Ψp = permutedims(Ψ, (1, 3, 2, 4))      # [aL, p2, p1, aR] -> vec gives (p1-1)*2+p2
	Ψmat = reshape(Ψp, aL, 4, aR)           # merge physical indices in std basis order
	best_err = Inf
	best_Ψ = Ψ
	best_C = Matrix{ComplexF64}(I, 4, 4)
	for C in TWO_QUBIT_CLIFFORDS
		# Ψcmat[aL, po, aR] = Σ_p C[po, p] Ψmat[aL, p, aR], with the (aL, aR) bond
		# indices preserved as a batch (a matmul A*Cᵀ here would wrongly contract aR!)
		Ψcmat = similar(Ψmat)
		@tensor Ψcmat[x, po, y] := C[po, p] * Ψmat[x, p, y]
		Ψcp = reshape(Ψcmat, aL, 2, 2, aR)  # [aL, p2o, p1o, aR]
		Ψc = permutedims(Ψcp, (1, 3, 2, 4)) # back to [aL, p1o, p2o, aR]
		_, _, _, err = tsvd(Ψc, (1, 2), (3, 4); trunc=alg.trunc)
		if err < best_err
			best_err = err
			best_Ψ = Ψc
			best_C = C
		end
	end

	# SVD truncate the best Clifford-transformed tensor.
	# `move_right=true`  (left sweep):  absorb singular values into the right site (s+1).
	# `move_right=false` (right sweep): absorb singular values into the left  site (s).
	u, sv, v, _ = tsvd!(best_Ψ, (1, 2), (3, 4); trunc=alg.trunc)
	if move_right
		env.ket[s] = u
		sm = Diagonal(sv)
		@tensor vnew[nb, p2, aR] := sm[nb, j] * v[j, p2, aR]
		env.ket[s+1] = vnew
	else
		env.ket[s+1] = v
		sm = Diagonal(sv)
		@tensor unew[aL, p1, nb] := u[aL, p1, j] * sm[j, nb]
		env.ket[s] = unew
	end
	# the applied circuit is part of the state description: record it on the cache
	push!(env.gates, CliffordGate(s, best_C))

	# transform the Hamiltonian MPO on the two sites: W -> C W C†
	if !isapprox(best_C, Matrix{ComplexF64}(I, 4, 4); atol=1e-10)
		@tensor Wcomb[aLw, p1o, p2o, aRw, p1i, p2i] := W1[aLw, p1o, b, p1i] * W2[b, p2o, aRw, p2i]
		# align both output (p1o,p2o) and input (p1i,p2i) pairs to Clifford basis order
		# (row-major in (p1,p2)); column-major reshape needs (p2,p1) ordering.
		Wc_in = permutedims(Wcomb, (1, 3, 2, 4, 6, 5))  # [aLw, p2o, p1o, aRw, p2i, p1i]
		w1 = size(Wc_in, 1); w2 = size(Wc_in, 4)
		Wc = reshape(Wc_in, w1, 4, w2, 4)   # (aLw, po, aRw, pi) with po,pi in std basis order
		@tensor Wct[aLw, po, aRw, pin] := best_C[po, qo] * Wc[aLw, qo, aRw, ri] * conj(best_C[pin, ri])
		# split back to individual physical indices, restoring (p1, p2) order
		Wt4 = reshape(Wct, w1, 2, 2, w2, 2, 2)  # [aLw, p2o, p1o, aRw, p2i, p1i]
		Wt = permutedims(Wt4, (1, 3, 2, 4, 6, 5))  # [aLw, p1o, p2o, aRw, p1i, p2i]
		# re-split Wt into two MPO tensors by SVD on the MPO bond.
		# Wt[aLw, p1o, p2o, aRw, p1i, p2i]; split between (p1o,p1i) and (p2o,p2i).
		Wmat = permutedims(Wt, (1, 2, 5, 3, 6, 4))   # (aLw, p1o, p1i, p2o, p2i, aRw)
		ml = size(Wmat, 1) * size(Wmat, 2) * size(Wmat, 3)
		Wmat = reshape(Wmat, ml, :)
		uW, sW, vW, _ = tsvd(Wmat; trunc=truncdim(D=size(W1, 3) + size(W2, 1) - 1))
		sqrt_s = sqrt.(sW)
		nb = length(sW)
		# left tensor: reshape row-index (aLw, p1o, p1i) then move bond dim to slot 3
		W1t = reshape(uW * Diagonal(sqrt_s), size(Wt, 1), 2, 2, nb)  # [aLw, p1o, p1i, nb]
		env.H[s] = permutedims(W1t, (1, 2, 4, 3))                     # [aLw, p1o, nb, p1i]
		# right tensor: reshape col-index (p2o, p2i, aRw) then move aRw to slot 3
		W2t = reshape(Diagonal(sqrt_s) * vW, nb, 2, 2, size(Wt, 4))   # [nb, p2o, p2i, aRw]
		env.H[s+1] = permutedims(W2t, (1, 2, 4, 3))                    # [nb, p2o, aRw, p2i]
	end

	return E, info
end

# ---------- sweeps ----------

"""
	leftsweep!(env::CADMRGCache, alg::CADMRG) -> kvals

Left-to-right CA-DMRG sweep: two-site local minimization (sites s, s+1) with Clifford
augmentation. The SVD itself moves the orthogonality center (singular values absorbed
into site s+1); no QR gauge moves are performed, so the not-yet-swept tensors stay
right-canonical and the precomputed right environments remain valid. Every applied
Clifford circuit is recorded on `env.gates`.
"""
function leftsweep!(env::CADMRGCache, alg::CADMRG)
	L = length(env.ket)
	kvals = zeros(Float64, L)
	for s in 1:L-1
		kvals[s], _ = _cadmrg_local_update!(env, s, alg; move_right=true)
		# rebuild the left environment with the (possibly transformed) H[s]
		updateleft!(env, s)
	end
	# last slot carries the final two-site energy (no separate single-site update: the
	# transformed MPO gauge makes a single-site Heff inconsistent with the two-site one).
	kvals[L] = kvals[L-1]
	return kvals
end

"""
	rightsweep!(env::CADMRGCache, alg::CADMRG) -> kvals

Right-to-left CA-DMRG sweep (symmetric to `leftsweep!`): singular values are absorbed
into the left site of each pair, keeping the already-swept tensors right-canonical.
Every applied Clifford circuit is recorded on `env.gates`.
"""
function rightsweep!(env::CADMRGCache, alg::CADMRG)
	L = length(env.ket)
	kvals = zeros(Float64, L)
	k = 1
	for s in L:-1:2
		kvals[k], _ = _cadmrg_local_update!(env, s - 1, alg; move_right=false)
		k += 1
		# rebuild the right environment with the (possibly transformed) H[s]
		updateright!(env, s)
	end
	# last slot carries the final two-site energy (no separate single-site update).
	kvals[L] = kvals[L-1]
	return kvals
end

sweep!(env::CADMRGCache, alg::CADMRG) = vcat(leftsweep!(env, alg), rightsweep!(env, alg))

# ---------- driver ----------

"""
	ground_state!(ψ::CanonicalMPS, h::AbstractMPO, alg::CADMRG) -> khist

Clifford-augmented ground-state search starting from the user-provided ansatz `ψ`
(modified in place; `alg.trunc` only affects the sweep truncations when an explicit
ansatz is given). The MPO `h`
is converted to dense form and rewritten in place (transformed by the optimal Clifford
circuits). Returns `khist`, the per-sweep local-energy history. To obtain the applied
Clifford circuits together with the state, use `ground_state(h, alg)` which returns a
[`CAMPS`](@ref).
"""
function ground_state!(ψ::CanonicalMPS, h::AbstractMPO, alg::CADMRG)
	hd = _dense_mpo(h)   # CA-DMRG rewrites MPO tensors in place; use dense form
	env = CADMRGCache(hd, ψ)
	khist = iterative_compute!(env, alg)
	setscaling!(ψ, 1.0)
	lmul!(1 / norm(ψ), ψ)
	return khist
end

"""
	ground_state(h::MPOHamiltonian, alg::CADMRG) -> CAMPS

Ground state of `h` found by CA-DMRG, starting from a random MPS with the bond cap
carried by `alg.trunc`,
returned as a [`CAMPS`](@ref): the canonical MPS in the Clifford-rotated picture
together with every applied two-qubit Clifford circuit (in application order).
Observables on the original Hamiltonian picture are evaluated with
`expectation(::PauliTerm, ::CAMPS)`; e.g. the ground-state energy is the sum of the
Hamiltonian's Pauli-term expectations.
"""
function ground_state(h::MPOHamiltonian, alg::CADMRG)
	TC = complex(scalartype(h))
	ψ = randommps(TC, ophydims(h); D=_guess_bond(alg.trunc))
	hd = _dense_mpo(h)   # CA-DMRG rewrites MPO tensors in place; use dense form
	env = CADMRGCache(hd, ψ)
	khist = iterative_compute!(env, alg)
	(alg.verbosity > 0) && println("CA-DMRG converged (delta = $(_iterative_delta(khist)))")
	setscaling!(ψ, 1.0)
	lmul!(1 / norm(ψ), ψ)
	return CAMPS(ψ, env.gates)
end

# ---------- PauliTerm expectation value ----------

"""
	expectation(pt::PauliTerm, camp::CAMPS) -> ComplexF64

Expectation value `⟨ψ|pt|ψ⟩` of a Pauli string on the physical state represented by a
[`CAMPS`](@ref). The ket stored in a CAMPS lives in the Clifford-rotated picture
(`|ket⟩ = U|ψ⟩`), so the Pauli string is conjugated **through the recorded circuits** —
`U† P U` up to phase, computed gate by gate: a Clifford maps a two-qubit Pauli string
back to a single Pauli string with a phase, so each gate only touches the two sites it
acts on (a 4×4 conjugation plus a Pauli decomposition). The conjugated string is then
contracted with the MPS directly; no circuit is ever applied to the state itself.
"""
function expectation(pt::PauliTerm, camp::CAMPS)
	L = length(camp)
	length(pt.ops) == L ||
		throw(DimensionMismatch("PauliTerm acts on $(length(pt.ops)) sites but the CAMPS has $L"))
	paulis = copy(pt.ops)
	coeff = pt.coeff
	# conjugate the Pauli string through every recorded gate, in application order:
	# after gate k the ket carries U_k = C_k⋯C_1, so P → C_k P C_k†.
	for gate in camp.gates
		s = gate.site
		Pl = kron(_pauli_matrix(paulis[s]), _pauli_matrix(paulis[s+1]))
		Pl = gate.C * Pl * gate.C'
		# Clifford normalizes the Pauli group: exactly one single-site product survives
		found = false
		for (sa, Ma) in pairs(_PAULI_MAP), (sb, Mb) in pairs(_PAULI_MAP)
			t = tr(kron(Ma, Mb)' * Pl) / 4
			if abs(t) > 1e-8
				paulis[s] = sa
				paulis[s+1] = sb
				coeff *= t
				found = true
				break
			end
		end
		found || throw(ArgumentError("Clifford conjugation did not produce a Pauli string"))
	end
	# contract the (conjugated) Pauli string with the MPS
	h = ones(ComplexF64, 1, 1, 1)
	for s in 1:L
		W = zeros(ComplexF64, 1, 2, 1, 2)
		W[1, :, 1, :] .= _pauli_matrix(paulis[s])
		h = _updateleft(h, camp.ket[s], W, camp.ket[s])
	end
	return coeff * h[1, 1, 1]
end
