# dense reference of the exponentially decaying operator
#   O = Σ_k α_k Σ_{i<j} λ_k^(j-i) · a_i m_(i+1) ⋯ m_(j-1) b_j
function expdecay_dense(a, m, b, αs, λs, L)
	d = size(a, 1)
	O = zeros(ComplexF64, d^L, d^L)
	for k in eachindex(αs)
		for i in 1:L-1, j in i+1:L
			factors = [ifelse(t == i, a, ifelse(t == j, b, ifelse(i < t < j, m, I(d))))
					   for t in 1:L]
			O .+= αs[k] * λs[k]^(j - i) .* reshape(kron(factors...), d^L, d^L)
		end
	end
	return O
end

@testset "ExpDecayOpTerm/ExpDecayOpSum" begin
	Random.seed!(61)
	L = 5
	d = 2
	a = randn(ComplexF64, d, d)
	m = randn(ComplexF64, d, d)
	b = randn(ComplexF64, d, d)

	# single term: the Schur representation matches the dense reference
	t = ExpDecayOpTerm(a, m, b, 0.4, 1.7)
	Ws = SchurMPOTensor(t, zeros(ComplexF64, d, d))
	@test Ws isa SchurMPOTensor
	@test phydim(Ws) == d
	H = MPOHamiltonian(t, L)
	@test maximum(abs.(todense(MPO(tompotensors(H))) - expdecay_dense(a, m, b, [0.4], [1.7], L))) < 1e-12

	# on-site hloc: every D corner carries it, i.e. Σ_i hloc ⊗ I ⋯ on top of the pairs
	hloc = randn(ComplexF64, d, d)
	Hloc = MPOHamiltonian(t, L, hloc)
	Href = expdecay_dense(a, m, b, [0.4], [1.7], L)
	for i in 1:L
		factors = [t == i ? hloc : I(d) for t in 1:L]
		Href .+= reshape(kron(factors...), d^L, d^L)
	end
	@test maximum(abs.(todense(MPO(tompotensors(Hloc))) - Href)) < 1e-12

	# a sum of two decay channels
	αs = [0.4, 0.8]
	λs = [1.7, -0.6]
	s = ExpDecayOpSum(a, m, b, αs, λs)
	@test scalartype(s) == ComplexF64
	Hs = MPOHamiltonian(s, L)
	Ht = MPOHamiltonian(t, L)
	Ht2 = MPOHamiltonian(ExpDecayOpTerm(a, m, b, 0.8, -0.6), L)
	Wsum = todense(MPO(tompotensors(Hs)))
	Wadd = todense(MPO(tompotensors(Ht))) + todense(MPO(tompotensors(Ht2)))
	@test maximum(abs.(Wsum - Wadd)) < 1e-12
	@test maximum(abs.(Wsum - expdecay_dense(a, m, b, αs, λs, L))) < 1e-12

	# identity propagator: recovers Σ_k α_k Σ_{i<j} λ^(j-i) a_i b_j exactly
	sid = ExpDecayOpSum(a, I(2), b, αs, λs)
	Hid = MPOHamiltonian(sid, L)
	Oref = zeros(ComplexF64, 2^L, 2^L)
	for k in 1:2, i in 1:L-1, j in i+1:L
		factors = [ifelse(tt == i, a, ifelse(tt == j, b, I(2))) for tt in 1:L]
		Oref .+= αs[k] * λs[k]^(j - i) .* reshape(kron(factors...), 2^L, 2^L)
	end
	@test maximum(abs.(todense(MPO(tompotensors(Hid))) - Oref)) < 1e-12

	# debug constructor to OpSum: the explicit term expansion represents the same operator
	opsum = OpSum(s, L)
	@test opsum isa OpSum
	Hos = MPOHamiltonian(opsum)
	@test maximum(abs.(todense(MPO(tompotensors(Hos))) - Wsum)) < 1e-12

	# ground state of an exponentially decaying perturbed TFIM: the exact sum of the
	# OpTerm-built TFIM chain and the ExpDecay chain
	tfim = OpSum(fill(2, L))
	for i in 1:L
		push!(tfim, OpTerm(-1.0, i => Float64[0 1; 1 0]))
	end
	for i in 1:L-1
		push!(tfim, OpTerm(-1.0, i => Float64[1 0; 0 -1], i + 1 => Float64[1 0; 0 -1]))
	end
	Hdecay = MPOHamiltonian(ExpDecayOpSum(a, m, b, [0.3], [0.05]), L)
	Hmixed = MPOHamiltonian(MPO(tompotensors(MPOHamiltonian(tfim))) + Hdecay)
	E, ψ = ground_state(Hmixed, DMRG1(maxiter=30, tol=1e-10, D=16))
	@test isfinite(real(E))
	# the variational energy of the converged state matches its expectation value
	@test real(E) ≈ real(expectationvalue(Hmixed, ψ)) rtol = 1e-8
end

@testset "OpSum validation" begin
	SX = Float64[0 1; 1 0]
	SZ = Float64[1 0; 0 -1]
	s = OpSum([2, 2, 2])
	push!(s, term(1 => SX))
	push!(s, term(-0.5, 1 => SZ, 2 => SX))
	@test length(s) == 2 && s[1] isa OpTerm && first(s.data) === s[1]
	# iterating an OpSum yields its terms
	@test collect(s) == s.data
	# out-of-range position
	@test_throws ArgumentError push!(s, term(4 => SX))
	# operator dimension does not match the local dimension
	@test_throws DimensionMismatch push!(s, term(1 => randn(ComplexF64, 3, 3)))
	# batch construction is validated identically
	s2 = OpSum([2, 2, 2], [term(1 => SX), term(-0.5, 1 => SZ, 2 => SX)])
	@test s2.data == s.data
	# a validated OpSum assembles the same Hamiltonian as the raw (L, terms) route
	@test todense(MPO(tompotensors(MPOHamiltonian(s2)))) ≈
		  todense(MPO(tompotensors(MPOHamiltonian(3, s2.data...)))) atol = 1e-14
end
