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

	# 4-argument convenience: the propagator defaults to the identity
	t_id = ExpDecayOpTerm(a, b, 0.4, 1.7)
	@test t_id.m == I(d) && t_id.a == a && t_id.b == b && t_id.α == 0.4 && t_id.λ == 1.7
	H_id = MPOHamiltonian(t_id, L)
	@test maximum(abs.(todense(MPO(tompotensors(H_id))) - expdecay_dense(a, I(d), b, [0.4], [1.7], L))) < 1e-12

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

	# 4-argument convenience: identity propagator
	s_id = ExpDecayOpSum(a, b, αs, λs)
	@test s_id.m == I(d)
	@test maximum(abs.(todense(MPO(tompotensors(MPOHamiltonian(s_id, L)))) -
					   expdecay_dense(a, I(d), b, αs, λs, L))) < 1e-12

	# adjoint: adjointed operators and conjugated couplings; the represented chain
	# matches the dense matrix adjoint
	t_a = adjoint(t)
	@test t_a.a == adjoint(a) && t_a.m == adjoint(m) && t_a.b == adjoint(b)
	@test t_a.α == conj(0.4) && t_a.λ == conj(1.7)
	sa = adjoint(s)
	@test sa.a == adjoint(a) && sa.m == adjoint(m) && sa.b == adjoint(b)
	@test sa.αs == conj.(αs) && sa.λs == conj.(λs)
	@test maximum(abs.(todense(MPO(tompotensors(MPOHamiltonian(sa, L)))) -
					   adjoint(expdecay_dense(a, m, b, αs, λs, L)))) < 1e-12

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

	# per-site on-site operators: the chain length is length(hlocs)
	hlocs = [randn(2, 2) for _ in 1:L]
	Hlocs = MPOHamiltonian(s, hlocs)
	@test length(Hlocs) == L
	# equals the zero-hloc decay chain plus the on-site chain (a block-native sum check,
	# cf. the DenseMPO test in sparsempo.jl)
	Hons = MPOHamiltonian(L, [OpTerm(1.0, i => hlocs[i]) for i in 1:L]...)
	Href = MPO(MPOHamiltonian(s, L)) + MPO(Hons)
	@test maximum(abs.(todense(MPO(Hlocs)) - todense(Href))) < 1e-12

	# ground state of an exponentially decaying perturbed TFIM: the exact sum of the
	# OpTerm-built TFIM chain and the ExpDecay chain. DMRG assumes a Hermitian operator, so
	# the decay chain uses σy at both ends (Hermitian, complex) with an identity propagator:
	# Σ_{i<j} λ^(j-i) σy_i σy_j stays Hermitian while keeping ComplexF64 arithmetic.
	tfim = OpTerm(-1.0, 1 => Float64[0 1; 1 0])
	for i in 2:L
		tfim += OpTerm(-1.0, i => Float64[0 1; 1 0])
	end
	for i in 1:L-1
		tfim += OpTerm(-1.0, i => Float64[1 0; 0 -1], i + 1 => Float64[1 0; 0 -1])
	end
	Hdecay = MPOHamiltonian(ExpDecayOpSum(_SY, I(2), _SY, [0.3], [0.05]), L)
	Hmixed = hamiltonian(MPO(tompotensors(MPOHamiltonian(tfim))) + MPO(Hdecay))
	E, ψ, _ = ground_state(Hmixed, DMRG1(maxiter=30, tol=1e-10, D=16))
	@test isfinite(real(E))
	# the variational energy of the converged state matches its expectation value
	@test real(E) ≈ real(expectationvalue(Hmixed, ψ)) rtol = 1e-8
end

@testset "OpSum validation" begin
	SX = Float64[0 1; 1 0]
	SZ = Float64[1 0; 0 -1]
	s = term(1 => SX) + term(-0.5, 1 => SZ, 2 => SX)
	@test s isa OpSum && length(s.data) == 2 && s.data[1] isa OpTerm
	@test s.ds == [2, 2]
	# the lattice of a plain-term sum is inferred from the operators
	@test scalartype(s) == Float64
	# conflicting operator dimensions on one site
	@test_throws DimensionMismatch term(1 => SX) + term(1 => randn(ComplexF64, 3, 3))
	# mixed scalar types are promoted
	sc = term(1 => SX) + term(-0.5im, 2 => SX)
	@test scalartype(sc) == ComplexF64
	# batch construction is validated identically
	s2 = OpSum([2, 2], [term(1 => SX), term(-0.5, 1 => SZ, 2 => SX)])
	@test s2.data == s.data && s2.ds == s.ds
	# a validated OpSum assembles the same Hamiltonian as the raw (L, terms) route
	@test todense(MPO(tompotensors(MPOHamiltonian(s2)))) ≈
		  todense(MPO(tompotensors(MPOHamiltonian(2, s2.data...)))) atol = 1e-14
	# OpSum + OpTerm / OpTerm + OpSum / OpSum + OpSum accumulate on the same lattice
	s3 = s2 + term(0.5, 2 => SX)
	@test length(s3.data) == 3
	s4 = term(0.25, 2 => SZ) + s3
	@test length(s4.data) == 4
	@test s4 + s2 isa OpSum && length((s4 + s2).data) == 6
end
