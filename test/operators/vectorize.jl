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
	hH = MPOHamiltonian(OpSum(fill(2, L), [term(1.0, 1 => _SX, L => _SX)]))
	@test superoperator(hH; side=:left) isa MPO
end

@testset "MPO kron and transpose" begin
	Random.seed!(25)
	L = 3
	ds = fill(2, L)

	# transpose transposes the represented operator matrix
	h = MPO(randommpo(ComplexF64, ds; D=3).data)
	@test todense(transpose(h)) ≈ collect(transpose(todense(h))) atol = 1e-10
	ρ = randommpo(ComplexF64, ds; D=3, normalize=false)
	setscaling!(ρ, 1.2)
	@test todense(transpose(ρ)) ≈ collect(transpose(todense(ρ))) atol = 1e-10
	@test scaling(transpose(ρ)) ≈ 1.2

	# kron of one-site chains is the fused-space Kronecker product: `a` carries the
	# FAST fused legs (vectorize convention), while Base.kron orders its first factor
	# slowest — so the dense forms swap the factor roles
	a1 = MPO([randn(ComplexF64, 1, 3, 1, 3)])
	b1 = MPO([randn(ComplexF64, 1, 2, 1, 2)])
	@test todense(kron(a1, b1)) ≈ kron(todense(b1), todense(a1)) atol = 1e-10
	# bond dimensions multiply; equal chain lengths are required
	a2 = MPO(randommpo(ComplexF64, ds; D=2).data)
	b2 = MPO(randommpo(ComplexF64, ds; D=4).data)
	@test bonddims(kron(a2, b2)) == bonddims(a2) .* bonddims(b2)
	@test_throws DimensionMismatch kron(h, MPO(randommpo(ComplexF64, fill(2, 2); D=3).data))

	# the superoperators are the kron products with the identity chain
	𝕀 = MPO(Float64, ds)
	@test superoperator(h, :left).data == kron(h, 𝕀).data
	@test superoperator(h, :right).data == kron(𝕀, transpose(h)).data
end

@testset "expectation scaling (regression)" begin
	# `expectation`/`expectationvalue` include `scaling(h)^L` of a scaled CanonicalMPO
	# operator (its `scaling^L` is part of the represented value)
	Random.seed!(35)
	L = 3
	ds = fill(2, L)
	ρ = randommpo(ComplexF64, ds; D=2, normalize=false)
	ρs = deepcopy(ρ)
	setscaling!(ρs, 1.7)
	Hd = MPO(randommpo(ComplexF64, ds; D=2).data)
	Hs = CanonicalMPO(Hd.data; scaling=2.3)
	@test real(expectationvalue(Hs, ρs)) ≈ 2.3^L * real(expectationvalue(Hd, ρs)) rtol = 1e-8
	@test real(expectation(Hs, ρs)) ≈ 2.3^L * real(expectation(Hd, ρs)) rtol = 1e-8
end
