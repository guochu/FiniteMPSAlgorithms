using FiniteMPSAlgorithms
using Test, LinearAlgebra, Random
using FiniteMPSAlgorithms: SVD, QR, QRpos, LQ, LQpos, SDD, Polar, l_LL, r_RR



@testset "structures" begin
	Random.seed!(1234)
	L = 6
	ds = fill(2, L)
	ψ = randommps(Float64, fill(2, L); D=8)

	# randommps / randommpo are always canonical (any `normalize` keyword value);
	# with normalize=true the total norm is 1
	for normalize in (true, false)
		ψr = randommps(Float64, ds; D=4, normalize)
		@test ψr isa AbstractMPS
		@test iscanonical(ψr)
		ρr = randommpo(Float64, ds; D=4, normalize)
		@test ρr isa AbstractMPO
		@test iscanonical(ρr)
		if normalize
			@test norm(ψr) ≈ 1 atol = 1e-8
			@test norm(ρr) ≈ 1 atol = 1e-8
		else
			@test isfinite(norm(ψr)) && isfinite(norm(ρr))
		end
	end
	ψs2 = randommps(ComplexF64, ds; D=4, normalize=true)
	@test iscanonical(ψs2)
	@test norm(ψs2) ≈ 1 atol = 1e-10
	ρs2 = randommpo(ComplexF64, ds; D=4, normalize=true)
	@test iscanonical(ρs2)
	@test norm(ρs2) ≈ 1 atol = 1e-10
	# tensor aliases
	@test ψ[1] isa MPSTensor
	@test ρs2[1] isa MPOTensor
	# canonical form
	@test all(isrightcanonical(ψ[i]) for i in 1:L)
	@test svectors_uninitialized(ψ) == false
	@test iscanonical(ψ; atol=1e-8)
	# normalization: dot includes scaling^L
	@test dot(ψ, ψ) ≈ 1 atol = 1e-10
	@test norm(ψ) ≈ 1 atol = 1e-10
	# per-site scaling convention
	ψ2 = ψ * 3.0
	@test norm(ψ2) ≈ 3.0 atol = 1e-8
	@test distance(ψ, ψ2) ≈ 2 * norm(ψ) atol = 1e-6
	# fidelity: normalized overlap, invariant under global phase and external scale
	# (distance sees both, fidelity sees neither)
	ψph = complex(ψ) * cis(0.7)
	@test fidelity(ψ, ψ) ≈ 1 atol = 1e-12
	@test fidelity(ψ, ψ2) ≈ 1 atol = 1e-12
	@test fidelity(ψph, ψ) ≈ 1 atol = 1e-12
	@test distance(ψph, ψ) > 1e-3
	@test infidelity(ψph, ψ) ≈ 0 atol = 1e-12
	# generic pair vs the direct dense formula
	φo = randommps(ComplexF64, ds; D=4)
	vψ, vφ = todense(ψ), todense(φo)
	@test fidelity(ψ, φo) ≈ abs(dot(vψ, vφ)) / (norm(vψ) * norm(vφ)) atol = 1e-10
	# exact sum (block-diagonal)
	ψs = ψ + ψ
	@test norm(ψs) ≈ 2.0 atol = 1e-6
	# entanglement of a random state is positive
	@test entanglement_entropy(ψ; bond=3) > 0
	# todense helper sanity
	@test todense(prodmps(Float64, fill(2, L), fill(1, L))) ≈ [1.0; zeros(2^L - 1)]
end

@testset "hadamard product (⊙)" begin
	Random.seed!(71)
	L = 6
	ds = fill(2, L)
	ψ = randommps(ComplexF64, ds; D=4)
	φ = randommps(ComplexF64, ds; D=4)
	vψ, vφ = todense(ψ), todense(φ)

	# exact pointwise product of amplitudes
	χ = ψ ⊙ φ
	@test todense(χ) ≈ vψ .* vφ atol = 1e-12

	# bond dimensions are the pairwise product
	@test bonddims(χ) == bonddims(ψ) .* bonddims(φ)
	# physical lattice unchanged
	@test phydims(χ) == ds

	# per-site scaling multiplies
	ψs, φs = copy(ψ), copy(φ)
	setscaling!(ψs, 1.5)
	setscaling!(φs, 2.0)
	χs = ψs ⊙ φs
	@test scaling(χs) ≈ 3.0

	# Hadamard with a product state keeps the bond profile of the other factor
	prod = prodmps(Float64, ds, fill(1, L))
	χp = ψ ⊙ prod
	@test bonddims(χp) == bonddims(ψ)
	@test todense(χp) ≈ vψ .* todense(prod) atol = 1e-12

	# mismatched length / physical dimensions
	ψ5 = randommps(ComplexF64, fill(2, L-1); D=4)
	@test_throws DimensionMismatch ψ ⊙ ψ5
	ψ3 = randommps(ComplexF64, fill(3, L); D=4)
	@test_throws DimensionMismatch ψ ⊙ ψ3
end

@testset "structures coverage" begin
    Random.seed!(42)
    L = 6
    ds = fill(2, L)

    # SparseMPOHamiltonian is an alias for MPOHamiltonian
    p = model_params(L)
    H = mpo_model(p)
    @test SparseMPOHamiltonian === MPOHamiltonian
    @test H isa SparseMPOHamiltonian

    # CanonicalMPO: construction and basic properties
    ρ = randommpo(ComplexF64, ds; D=4)
    @test ρ isa CanonicalMPO
    @test length(ρ) == L
    @test norm(ρ) ≈ 1 atol = 1e-8
    # identity CanonicalMPO
    ρid = CanonicalMPO(Float64, ds)
    @test tr(ρid) ≈ 2^L

    # Orthogonalize: construction variants
    orth1 = Orthogonalize(SVD(), truncdim(4))
    @test orth1.orth == SVD()
    @test orth1.trunc == truncdim(4)
    orth2 = Orthogonalize(QR(), NoTruncation(); normalize=true)
    @test orth2.normalize
    orth3 = Orthogonalize(; alg=QR(), trunc=NoTruncation())
    @test orth3.orth == QR()

    # term: convenience constructor for OpTerm
    σx = Float64[0 1; 1 0]
    t1 = term(-1.0, 1 => σx)
    @test t1 isa OpTerm
    @test t1.coeff == -1.0
    t2 = term(2 => σx)  # default coeff = 1
    @test t2.coeff == 1.0

    # isleftcanonical on MPS tensor
    ψ = randommps(Float64, ds; D=4)
    leftorth!(ψ; alg=Orthogonalize(QR()))
    @test isleftcanonical(ψ[1])
    # MPO: a left-canonical (bond-1) MPO chain
    W = identitympo(Float64, ds)

    # canonicalize (non-mutating): input unchanged
    ψ2 = randommps(Float64, ds; D=4)
    ψ2copy = copy(ψ2)
    ψ3 = canonicalize(ψ2)
    @test distance(ψ2, ψ2copy) < 1e-12  # original not modified

    # space_l / space_r on MPS and MPO
    @test space_l(ψ) == 1               # leftmost boundary
    @test space_r(ψ) == 1              # rightmost boundary
    @test space_l(ψ[2]) == size(ψ[2], 1)
    @test space_r(ψ[2]) == size(ψ[2], 3)
    @test space_l(W) == 1
    @test space_r(W) == 1

    # phydim / iphydim on MPO tensor
    @test phydim(W[1]) == 2
    @test iphydim(W[1]) == 2

    # phydims on MPS
    @test phydims(ψ) == ds

    # ophydims / iphydims on MPO
    @test ophydims(W) == ds
    @test iphydims(W) == ds

    # bonddims on MPS
    @test bonddims(ψ) == [bonddim(ψ, b) for b in 1:L-1]

    # setscaling!
    ψs = copy(ψ)
    setscaling!(ψs, 2.0)
    @test scaling(ψs) == 2.0

    # unset_svectors! on MPS
    ψu = copy(ψ)
    unset_svectors!(ψu)
    @test svectors_uninitialized(ψu)

    # internal transfer primitives: stateless overlap transfer (MPS)
    hl = FiniteMPSAlgorithms._updateleft(ones(Float64, 1, 1), ψ[1], ψ[1])
    @test size(hl, 1) == size(ψ[1], 3)
    hr = FiniteMPSAlgorithms._updateright(ones(Float64, 1, 1), ψ[end], ψ[end])
    @test size(hr, 2) == size(ψ[end], 1)

    # stateless MPO-MPS environment transfer
    h3 = l_LL(ψ, W, ψ)   # (1, 1, 1)
    hl3 = FiniteMPSAlgorithms._updateleft(h3, ψ[1], W[1], ψ[1])
    @test size(hl3) == (space_r(ψ[1]), size(W[1], 3), space_r(ψ[1]))

    # trace transfer
    v = ones(Float64, 1)
    vl = FiniteMPSAlgorithms._updatetraceleft(v, W[1])
    @test length(vl) == size(W[1], 3)
    vr = FiniteMPSAlgorithms._updatetraceright(v, W[end])
    @test length(vr) == size(W[end], 1)
    # full trace of identity MPO == d^L
    @test tr(identitympo(Float64, ds)) ≈ 2.0^L

    # entanglement_spectrum / schmidt_values
    ψe = randommps(Float64, ds; D=8)
    spec = entanglement_spectrum(ψe; bond=3)
    sv = schmidt_values(ψe; bond=3)
    @test spec ≈ abs2.(sv)
    @test sum(spec) ≈ 1 atol = 1e-10

    # randommpo
    Wrand = randommpo(ComplexF64, ds; D=4)
    @test Wrand isa CanonicalMPO
    @test length(Wrand) == L
    @test bonddim(Wrand) <= 4

    # changebond! on MPS: grow and shrink to the same cap
    ψsmall = randommps(Float64, ds; D=2)
    changebond!(ψsmall; D=8)
    @test 2 <= bonddim(ψsmall) <= 8
    @test iscanonical(ψsmall; atol=1e-8)
    changebond!(ψsmall; D=3)
    @test bonddim(ψsmall) <= 3
    @test iscanonical(ψsmall; atol=1e-8)

    # changebond! on MPO
    Wsmall = randommpo(Float64, ds; D=2)
    changebond!(Wsmall; D=8)
    @test 2 <= bonddim(Wsmall) <= 8
    changebond!(Wsmall; D=3)
    @test bonddim(Wsmall) <= 3

    # truncate! on MPS (truncate! is exported in both tensorops and structures)
    ψtr = randommps(Float64, ds; D=8)
    truncate!(ψtr; trunc=truncdim(4))
    @test bonddim(ψtr) <= 4
end
