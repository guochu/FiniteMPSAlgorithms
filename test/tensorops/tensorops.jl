using FiniteMPSAlgorithms
using Test, LinearAlgebra, Random
using FiniteMPSAlgorithms: SVD, QR, QRpos, LQ, LQpos, SDD, Polar

@testset "tensorops" begin
	# tsvd vs LinearAlgebra.svd
	A = randn(ComplexF64, 12, 7)
	u, s, v, err = tsvd(copy(A))
	U, S, V = svd(A)
	@test u * Diagonal(s) * v ≈ A
	@test s ≈ S[1:length(s)]
	@test err < 1e-13
	# truncation
	tr = truncdimcutoff(4, 1e-14)
	u2, s2, v2, err2 = tsvd(copy(A); trunc=tr)
	@test length(s2) == 4
	@test norm(A - u2 * Diagonal(s2) * v2) ≈ norm(view(S, 5:length(S)))
	# tensor factorization groupings
	T = randn(3, 4, 5, 6)
	q, r = leftorth!(copy(T), (1, 2), (3, 4))
	Tq, Tr = tie(q, (2, 1)), tie(r, (1, 2))
	@test Tq' * Tq ≈ I
	@test Tq * Tr ≈ tie(T, (2, 2))
	l, q2 = rightorth!(copy(T), (1,), (2, 3, 4))
	Tq2 = reshape(q2, size(q2, 1), :)
	@test Tq2 * Tq2' ≈ I
	@test reshape(l * Tq2, size(T)) ≈ T
	# isometry & helpers
	@test isometry(3, 5)' * isometry(3, 5) ≈ Diagonal([1, 1, 1, 0, 0])
	@test scalar(fill(2.0)) == 2.0
	@test renyi_entropy([0.25, 0.25, 0.25, 0.25]) ≈ log(4)
	# truncate! public wrapper
	v = collect(1.0:-1e-2:0.0)
        v2, e = truncate!(copy(v), TruncateDim(50))
        @test length(v2) == 50
end

@testset "tensorops coverage" begin
    Random.seed!(31)
    # QR / QRpos / LQ / LQpos / SVD / SDD / Polar as matrix factorization algorithms
    A = randn(ComplexF64, 8, 5)
    for alg in (QR(), QRpos(), SVD(), SDD(), Polar())
        q, r = leftorth!(copy(A), alg)
        @test q' * q ≈ I
        @test q * r ≈ A
    end
    # rightorth needs cols >= rows for Polar; use a wide matrix
    Aw = randn(ComplexF64, 5, 8)
    for alg in (LQ(), LQpos(), SVD(), SDD(), Polar())
        l, q = rightorth!(copy(Aw), alg)
        @test q * q' ≈ I
        @test l * q ≈ Aw
    end
    # adjoint of algorithm types
    @test adjoint(QRpos()) == LQpos()
    @test adjoint(QR()) == LQ()
    @test adjoint(LQpos()) == QRpos()
    @test adjoint(LQ()) == QR()
    @test adjoint(SVD()) == SVD()

    # tsvd! (tensor version with left/right grouping)
    T = randn(3, 4, 5, 6)
    u, s, v, err = tsvd!(copy(T), (1, 2), (3, 4))
    @test size(u) == (3, 4, length(s))
    @test size(v) == (length(s), 5, 6)
    @test err < 1e-13

    # leftorth / rightorth (non-mutating versions preserve input)
    A0 = randn(6, 4)
    q, r = leftorth(A0; alg=QRpos())
    @test A0 ≈ copy(A0)  # input not modified
    l, q2 = rightorth(A0; alg=LQpos())
    @test A0 ≈ copy(A0)

    # permute: single permutation and left/right grouping
    T2 = randn(2, 3, 4)
    Pv = permute(T2, (3, 1, 2))
    @test size(Pv) == (4, 2, 3)
    @test collect(Pv) == permutedims(T2, (3, 1, 2))
    # left/right grouping == full perm
    Pv2 = permute(T2, (1,), (2, 3))
    @test size(Pv2) == (2, 3, 4)

    # distance2
    x = randn(10)
    y = randn(10)
    @test distance2(x, y) ≈ norm(x - y)^2
    @test distance2(x, x) ≈ 0 atol = 1e-26

    # TruncateRelError / truncrelerr
    s = collect(1.0:-0.01:0.0)
    s2, err = truncate!(copy(s), truncrelerr(0.3))
    @test all(s2 .> 0.3 * norm(s))
    @test length(s2) < length(s)
    # keyword constructor
    tr = TruncateRelError(ϵ=0.1)
    @test tr.ϵ == 0.1
    s3, err3 = truncate!(collect(s), tr)
    @test all(s3 .> 0.1 * norm(s))
end

@testset "default truncation definition" begin
    # pin the exported default: a bond-capped relative cutoff with `Defaults.D`,
    # gauging precision `Defaults.tolgauge`, and at least one singular value kept
    dt = FiniteMPSAlgorithms.DefaultTruncation
    @test dt isa TruncateDimCutoff
    @test dt.D == Defaults.D
    @test dt.ϵ == Defaults.tolgauge
    @test dt.add_back == 1
    @test dt === truncdimcutoff(D=Defaults.D, ϵ=Defaults.tolgauge, add_back=1)
end
