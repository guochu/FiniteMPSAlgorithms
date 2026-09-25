@testset "models (finite 1d Hamiltonians)" begin
	# ---- spin-1/2 operators ----
	@test σx() == ComplexF64[0 1; 1 0]
	@test σy() == ComplexF64[0 -im; im 0]
	@test σz() == ComplexF64[1 0; 0 -1]
	@test σx(Float64) == Float64[0 1; 1 0]
	@test σz(Float64) == Float64[1 0; 0 -1]
	@test σy(Float64) == ComplexF64[0 -im; im 0]        # promoted to a complex type
	@test σx() * σy() ≈ im * σz()
	@test σy() * σz() ≈ im * σx()
	@test Sx() == σx() / 2 && Sy() == σy() / 2 && Sz() == σz() / 2
	@test Sz(Float64) == Float64[0.5 0; 0 -0.5]

	# ---- dense references (site 1 slowest, matching todense) ----
	I2 = Matrix{ComplexF64}(I, 2, 2)
	op_at(op, i, L, Id) =
		reshape(kron(ntuple(k -> k == i ? op : Id, L)...), size(Id, 1)^L, size(Id, 1)^L)

	# --- TFIM: H = −J Σ σˣσˣ − h Σ σᶻ ---
	L = 5
	J, h = 0.7, 1.3
	sx = ComplexF64[0 1; 1 0]
	sy = ComplexF64[0 -im; im 0]
	sz = ComplexF64[1 0; 0 -1]
	Ht = tfim_hamiltonian(L; J=J, h=h)
	@test Ht isa MPOHamiltonian
	Href = zeros(ComplexF64, 2^L, 2^L)
	for i in 1:L-1
		Href .+= (-J) .* op_at(sx, i, L, I2) * op_at(sx, i + 1, L, I2)
	end
	for i in 1:L
		Href .+= (-h) .* op_at(sz, i, L, I2)
	end
	@test todense(Ht) ≈ Href atol = 1e-13
	@test ishermitian(todense(Ht))
	# fully polarised product state (local index 1 has σᶻ = +1): only the field survives
	ψ0 = prodmps(ComplexF64, fill(2, L), fill(1, L))
	@test real(expectationvalue(Ht, ψ0)) ≈ -h * L atol = 1e-12

	# --- Heisenberg XXZ + field: H = J Σ (SˣSˣ + SʸSʸ + Δ SᶻSᶻ) − h Σ Sᶻ ---
	for (Δ, hf) in ((1.0, 0.0), (0.0, 0.0), (0.6, 0.35))
		Hh = heisenberg_hamiltonian(L; J=J, Δ=Δ, h=hf)
		Href = zeros(ComplexF64, 2^L, 2^L)
		for i in 1:L-1
			Href .+= J .* op_at(sx / 2, i, L, I2) * op_at(sx / 2, i + 1, L, I2)
			Href .+= J .* op_at(sy / 2, i, L, I2) * op_at(sy / 2, i + 1, L, I2)
			Href .+= (J * Δ) .* op_at(sz / 2, i, L, I2) * op_at(sz / 2, i + 1, L, I2)
		end
		for i in 1:L
			Href .+= (-hf) .* op_at(sz / 2, i, L, I2)
		end
		@test todense(Hh) ≈ Href atol = 1e-13
		@test ishermitian(todense(Hh))
	end

	# ---- Fermi-Hubbard vs an explicit Jordan-Wigner construction ----
	L = 4
	tt, U, μ = 1.0, 2.0, 0.3
	I4 = Matrix{ComplexF64}(I, 4, 4)
	# local mode operators, basis (|0⟩,|↑⟩,|↓⟩,|↑↓⟩) with the ↑ mode first per site; the
	# intra-site part of the string is folded in (the ↓ mode carries the ↑ parity)
	sm = ComplexF64[0 1; 0 0]                 # σ⁻ = |0⟩⟨↑|
	sz = ComplexF64[1 0; 0 -1]
	d_up = kron(sm, I2)                       # ↑ annihilation
	d_dn = kron(sz, sm)                       # ↓ annihilation
	P = kron(sz, sz)                          # on-site parity (−1)^n
	@test d_up * d_up' + d_up' * d_up ≈ I4
	@test d_dn * d_dn' + d_dn' * d_dn ≈ I4
	@test d_up * d_dn' + d_dn' * d_up ≈ zero(I4)
	# full fermionic operators: every mode carries the parity string of the preceding ones
	cop(σ, i) = begin
		M = op_at(σ === :up ? d_up : d_dn, i, L, I4)
		for k in 1:i-1
			M = op_at(P, k, L, I4) * M
		end
		M
	end
	cdag(σ, i) = cop(σ, i)'
	# canonical anticommutation relations (also across species and sites)
	for (σ, i, τ, j) in ((:up, 1, :up, 1), (:dn, 3, :dn, 3), (:up, 2, :up, 3),
						 (:up, 2, :dn, 2), (:dn, 1, :up, 4), (:up, 2, :dn, 3))
		δ = (i == j && σ == τ) ? one(ComplexF64) : zero(ComplexF64)
		@test norm(cdag(σ, i) * cop(τ, j) + cop(τ, j) * cdag(σ, i) - δ * I(4^L)) < 1e-12
		@test norm(cop(σ, i) * cop(τ, j) + cop(τ, j) * cop(σ, i)) < 1e-12
	end
	Href = zeros(ComplexF64, 4^L, 4^L)
	for i in 1:L-1, σ in (:up, :dn)
		Href .+= -tt .* (cdag(σ, i) * cop(σ, i + 1) + cdag(σ, i + 1) * cop(σ, i))
	end
	for i in 1:L
		Href .+= U .* (cdag(:up, i) * cop(:up, i)) * (cdag(:dn, i) * cop(:dn, i))
		Href .+= -μ .* (cdag(:up, i) * cop(:up, i) + cdag(:dn, i) * cop(:dn, i))
	end
	Hf = fermi_hubbard(L; t=tt, U=U, μ=μ)
	@test Hf isa MPOHamiltonian
	@test size(todense(Hf)) == (4^L, 4^L)
	@test todense(Hf) ≈ Href atol = 1e-12
	@test ishermitian(todense(Hf))
	# hopping only (U = μ = 0)
	Href0 = zeros(ComplexF64, 4^L, 4^L)
	for i in 1:L-1, σ in (:up, :dn)
		Href0 .+= -tt .* (cdag(σ, i) * cop(σ, i + 1) + cdag(σ, i + 1) * cop(σ, i))
	end
	@test todense(fermi_hubbard(L; t=tt, U=0.0, μ=0.0)) ≈ Href0 atol = 1e-12

	# ---- the models drive the algorithms: DMRG2 vs exact diagonalization ----
	Hcrit = tfim_hamiltonian(10; J=1.0, h=1.0)
	E, _ = ground_state(Hcrit, DMRG2(maxiter=20, verbosity=0))
	@test E ≈ eigmin(Hermitian(todense(Hcrit))) rtol = 1e-8
	Hhub = fermi_hubbard(4; t=1.0, U=2.0)
	Ef, _ = ground_state(Hhub, DMRG2(maxiter=50, verbosity=0))
	@test Ef ≈ eigmin(Hermitian(todense(Hhub))) rtol = 1e-6

	# ---- a real element type is supported throughout ----
	@test todense(tfim_hamiltonian(4; J=1.0, h=1.3, T=Float64)) ≈
		  todense(tfim_hamiltonian(4; J=1.0, h=1.3))
	@test todense(heisenberg_hamiltonian(4; J=1.0, Δ=0.5, T=Float64)) ≈
		  todense(heisenberg_hamiltonian(4; J=1.0, Δ=0.5))
	@test todense(fermi_hubbard(3; t=1.0, U=2.0, T=Float64)) ≈
		  todense(fermi_hubbard(3; t=1.0, U=2.0))
end
