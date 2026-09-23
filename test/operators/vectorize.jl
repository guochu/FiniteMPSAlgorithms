@testset "vectorize / devectorize" begin
	Random.seed!(5)
	L = 4
	ds = fill(2, L)

	# canonical MPO with a nontrivial external scale: the round trip is exact and keeps
	# bond dimensions and `scaling`
	ρ = randommpo(ComplexF64, ds; D=8, normalize=false)
	setscaling!(ρ, 1.3)
	v = vectorize(ρ)
	@test v isa CanonicalMPS
	@test phydims(v) == fill(4, L)
	@test bonddims(v) == bonddims(ρ)
	@test scaling(v) ≈ scaling(ρ)
	ρ2 = devectorize(v)
	@test ρ2 isa CanonicalMPO
	@test bonddims(ρ2) == bonddims(ρ)
	@test scaling(ρ2) ≈ scaling(ρ)
	@test todense(ρ2) ≈ todense(ρ) atol = 1e-10

	# plain MPO: same data-level round trip (no scaling field to carry)
	W = MPO(randommpo(ComplexF64, ds; D=6).data)
	vW = vectorize(W)
	@test vW isa CanonicalMPS && scaling(vW) == 1.0
	@test todense(devectorize(vW)) ≈ todense(W) atol = 1e-10

	# the merged physical dimension must be a perfect square to split back
	@test_throws DimensionMismatch devectorize(randommps(ComplexF64, [3, 3, 3, 3]; D=4))
end

@testset "superoperator" begin
	Random.seed!(15)
	L = 4
	ds = fill(2, L)
	h1 = MPO(randommpo(ComplexF64, ds; D=3).data)
	h2 = MPO(randommpo(ComplexF64, ds; D=4).data)
	exact = h1 * h2
	nrm = norm(todense(exact))

	# the superoperator is an MPO on the doubled physical space with the bond
	# dimensions of its operand
	𝒮_L = superoperator(h1; side=:left)
	𝒮_R = superoperator(h2; side=:right)
	@test 𝒮_L isa MPO && 𝒮_R isa MPO
	@test bonddims(𝒮_L) == bonddims(h1)
	@test ophydims(𝒮_L) == fill(4, L) && iphydims(𝒮_L) == fill(4, L)
	@test_throws ArgumentError superoperator(h1; side=:up)

	# route a): 𝒮_left(h1)·|h2⟩ reproduces h1·h2 (exact MPO·MPS product)
	@test todense(devectorize(𝒮_L * vectorize(h2))) ≈ todense(exact) atol = 1e-8 * nrm
	# route b): 𝒮_right(h2)·|h1⟩
	@test todense(devectorize(𝒮_R * vectorize(h1))) ≈ todense(exact) atol = 1e-8 * nrm

	# the same two routes with compressed (SVD) evaluation
	outa = devectorize(mult(𝒮_L, vectorize(h2), SVDCompression(trunc=truncdimcutoff(16, 1e-12))))
	outb = devectorize(mult(𝒮_R, vectorize(h1), SVDCompression(trunc=truncdimcutoff(16, 1e-12))))
	@test bonddim(outa) <= 16 && bonddim(outb) <= 16
	@test norm(todense(outa) - todense(exact)) / nrm < 1e-8
	@test norm(todense(outb) - todense(exact)) / nrm < 1e-8

	# canonical operands: the superoperator is linear in the represented `h`, so the
	# input's `scaling` carries into the result and the scaled routes reproduce the
	# represented product directly (no data materialization needed)
	ρ1 = randommpo(ComplexF64, ds; D=3, normalize=false)
	setscaling!(ρ1, 0.8)
	ρ2 = randommpo(ComplexF64, ds; D=4, normalize=false)
	setscaling!(ρ2, 1.5)
	ref = todense(ρ1) * todense(ρ2)
	𝒮c = superoperator(ρ1; side=:left)
	@test 𝒮c isa CanonicalMPO && scaling(𝒮c) ≈ 0.8
	@test bonddims(𝒮c) == bonddims(ρ1)
	@test todense(devectorize(𝒮c * vectorize(ρ2))) ≈ ref atol = 1e-8 * norm(ref)
	@test todense(devectorize(superoperator(ρ2; side=:right) * vectorize(ρ1))) ≈
		ref atol = 1e-8 * norm(ref)
	# plain MPO and MPOHamiltonian inputs stay scale-free
	@test superoperator(h1; side=:left) isa MPO
	hH = MPOHamiltonian(OpSum(fill(2, L), [term(1.0, 1 => _SX, 2 => _SX)]))
	@test superoperator(hH; side=:left) isa MPO
end

@testset "thermal state via TDVP purification vs ED" begin
	Random.seed!(42)
	L = 4
	ds = fill(2, L)
	p = model_params(L)
	H = mpo_model(p)
	Hd = dense_model(p)
	β = 1.0

	# thermal generator 𝒦 = H⊗I + I⊗Hᵀ on the vectorized chain (built from the
	# left/right superoperators): exp(-τ𝒦)|ρ₀⟩ = e^{-τH}·ρ₀·e^{-τH}, so evolving the
	# infinite-temperature state I/2^L to τ = β/2 gives the Gibbs state e^{-βH}/Z
	𝒦 = superoperator(H; side=:left) + superoperator(H; side=:right)

	ψ = changebond!(vectorize(infinite_temperature_state(ComplexF64, ds)); D=16)
	env = DMRGCache(𝒦, ψ)
	alg = TDVP1(stepsize=-im * β / 2 / 80, verbosity=0)
	for _ in 1:80
		sweep!(env, alg)
	end
	ρ = devectorize(env.ket)

	# ED reference: the Gibbs state of the dense Hamiltonian
	rho_ed = exp(-β * Matrix(Hermitian(Hd)))
	Z = tr(rho_ed)
	rho_ed ./= Z

	# trace of the represented state: Z/2^L (the I/2^L initial trace is conserved as
	# tr(e^{-βH})/2^L by the exact flow)
	@test real(expectation(identitympo(ComplexF64, ds), ρ)) ≈ Z / 2^L rtol = 1e-3

	# energy of the Gibbs state
	Hd_mpo = MPO(tompotensors(H))
	@test real(expectationvalue(Hd_mpo, ρ)) ≈ real(tr(Hd * rho_ed)) rtol = 1e-3

	# local observables
	I2 = Matrix{ComplexF64}(I, 2, 2)
	for (s, O) in [(2, _SZ), (3, _SX)]
		Os = reshape(kron(ntuple(k -> k == s ? O : I2, L)...), 2^L, 2^L)
		@test real(expectationvalue(term(s => O), ρ)) ≈ real(tr(Os * rho_ed)) atol = 1e-4
	end

	# the full density matrix
	ρt = todense(ρ)
	ρt ./= tr(ρt)
	@test norm(ρt - rho_ed) / norm(rho_ed) < 1e-3
end
