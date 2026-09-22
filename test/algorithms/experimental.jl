using Test
using LinearAlgebra
using Random
using FiniteMPSAlgorithms

@testset "CA-DMRG CAMPS" begin
	Random.seed!(2024)
	L = 4
	J, h = 1.0, 0.8
	SX = [0.0 1; 1 0.0]
	SZ = [1.0 0; 0 -1.0]
	I2 = [1.0 0; 0 1.0]

	# TFIM: H = -J Σ Z_i Z_{i+1} - h Σ X_i
	ds = fill(2, L)
	terms = OpSum(ds)
	for i in 1:L-1
		push!(terms, OpTerm(-J, i => SZ, i + 1 => SZ))
	end
	for i in 1:L
		push!(terms, OpTerm(-h, i => SX))
	end
	H = MPOHamiltonian(terms)

	# exact diagonalization reference (site 1 = most significant kron factor)
	op(ops...) = reshape(kron(ops...), 2^L, 2^L)
	Hdense = zeros(ComplexF64, 2^L, 2^L)
	for i in 1:L-1
		Hdense .-= J .* op(ntuple(k -> (k == i || k == i + 1) ? SZ : I2, L)...)
	end
	for i in 1:L
		Hdense .-= h .* op(ntuple(k -> k == i ? SX : I2, L)...)
	end
	Ed = eigen(Hermitian(Hdense))
	E_gs, ψ_gs = Ed.values[1], Ed.vectors[:, 1]

	camp = ground_state(H, CADMRG(D=16, maxiter=40, tol=1e-11))

	# the circuits are recorded, and the gates act on valid site pairs
	@test !isempty(camp.gates)
	@test length(camp) == L
	@test all(g -> 1 <= g.site <= L - 1, camp.gates)

	# a Pauli string on the CAMPS matches the exact (real-picture) ground state;
	# this fails if any Clifford circuit was dropped or conjugated in the wrong order
	@testset "pauli strings vs exact diagonalization" begin
		# dense two-site / single-site operator from a string
		function dense_string(s::String)
			str2mat = c -> c == 'I' ? I2 : (c == 'X' ? SX : c == 'Z' ? SZ : error("only I/X/Z here"))
			return op(ntuple(k -> str2mat(s[k]), L)...)
		end
		for s in ["ZZII", "IZZI", "IIZZ", "XIII", "IXII", "IIXI", "IIIX", "XZXI", "ZZXX"]
			ref = real(dot(ψ_gs, dense_string(s), ψ_gs))
			val = expectation(PauliTerm(s), camp)
			@test val ≈ ref atol = 1e-8
		end
		# coefficient handling (incl. complex phase)
		@test expectation(PauliTerm(-2.5, "ZZII"), camp) ≈ -2.5 * expectation(PauliTerm("ZZII"), camp)
		@test expectation(PauliTerm(1.0im, "XIII"), camp) ≈ 1.0im * expectation(PauliTerm("XIII"), camp)
		# identity string
		@test expectation(PauliTerm("IIII"), camp) ≈ 1.0 atol = 1e-10
	end

	# the Hamiltonian written as Pauli terms reproduces the exact ground energy:
	# observing the CAMPS through its circuits must not lose any information
	@testset "energy from pauli decomposition" begin
		E = zero(ComplexF64)
		for i in 1:L-1
			s = "I"^(i - 1) * "ZZ" * "I"^(L - i - 1)
			E += (-J) * expectation(PauliTerm(s), camp)
		end
		for i in 1:L
			s = "I"^(i - 1) * "X" * "I"^(L - i)
			E += (-h) * expectation(PauliTerm(s), camp)
		end
		@test real(E) ≈ E_gs atol = 1e-8
	end
end
