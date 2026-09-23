using FiniteMPSAlgorithms
using Test, LinearAlgebra, Random
using FiniteMPSAlgorithms: SVD, QR, QRpos, LQ, LQpos, SDD, Polar

# Dense/tensor linear algebra: orthogonal factorizations, tsvd, permute, isometry, distances

@testset "linalg" begin
	Random.seed!(117)

	# distance / distance2
	a = randn(3, 4)
	b = randn(3, 4)
	@test distance2(a, b) ≈ norm(a - b)^2
	@test distance(a, b) ≈ norm(a - b)
	@test distance2(a, a) == 0.0
	@test distance(a, a) == 0.0
	x = randn(10)
	y = randn(10)
	@test distance2(x, y) ≈ norm(x - y)^2
	@test distance2(x, x) ≈ 0 atol = 1e-26

	# isometry
	@test isometry(3) == Matrix(I, 3, 3)
	@test isometry(3) isa Matrix{Float64}
	@test isometry(ComplexF64, 3) == Matrix(I, 3, 3)
	@test isometry(ComplexF64, 3) isa Matrix{ComplexF64}
	i34 = isometry(3, 4)
	@test i34 == [1 0 0 0; 0 1 0 0; 0 0 1 0]
	@test i34 * i34' == Matrix(I, 3, 3)
	i32 = isometry(ComplexF64, 3, 2)
	@test i32 == [1 0; 0 1; 0 0]
	@test i32' * i32 == Matrix(I, 2, 2)
	@test isometry(3, 5)' * isometry(3, 5) ≈ Diagonal([1, 1, 1, 0, 0])

	# permute: single permutation and left/right grouping
	t = randn(2, 3, 4)
	@test permute(a, (2, 1)) ≈ permutedims(a, (2, 1))
	@test permute(a, (1,), (2,)) ≈ a
	@test permute(t, (1, 2), (3,)) ≈ permute(t, (1, 2, 3))
	Pv = permute(t, (3, 1, 2))
	@test size(Pv) == (4, 2, 3)
	@test collect(Pv) == permutedims(t, (3, 1, 2))
	Pv2 = permute(t, (1,), (2, 3))
	@test size(Pv2) == (2, 3, 4)

	# tsvd vs LinearAlgebra.svd
	A = randn(ComplexF64, 12, 7)
	u, s, v, err = tsvd(copy(A))
	U, S, svdV = svd(A)
	@test u * Diagonal(s) * v ≈ A
	@test s ≈ S[1:length(s)]
	@test err < 1e-13

	# tsvd and tsvd! (in-place tensor version covers permute + reconstruction)
	t3 = randn(2, 3, 4)
	t3c = copy(t3)
	u3, s3, v3, err3 = tsvd!(t3, (1, 2), (3,))
	md = length(s3)
	@test size(u3) == (2, 3, md)
	@test size(v3) == (md, 4)
	r = zeros(size(t3c))
	for k in 1:md
		r += u3[:, :, k] .* s3[k] .* reshape(v3[k, :], 1, 1, 4)
	end
	@test r ≈ t3c
	@test err3 == 0.0

	ac = copy(A)
	for alg in (SVD(), SDD())
		ua, sa, va, errs = tsvd(A; alg=alg)
		@test A == ac                       # input not modified
		@test errs == 0.0
		@test ua * Diagonal(sa) * va ≈ A
		@test all(sa .>= 0)
		# alg keyword also available for tsvd!
		u2, s2, v2, err2 = tsvd!(copy(A); alg=alg)
		@test s2 ≈ s
	end
	# SVD/SDD drivers give the same singular values; truncation via tsvd
	_, s_svd, _, _ = tsvd(A; alg=SVD())
	_, s_sdd, _, _ = tsvd(A; alg=SDD())
	@test s_svd ≈ s_sdd
	tr = truncdimcutoff(4, 1e-14)
	u2, s2, v2, err2 = tsvd(copy(A); trunc=tr)
	@test length(s2) == 4
	@test norm(A - u2 * Diagonal(s2) * v2) ≈ norm(view(S, 5:length(S)))
end

@testset "linalg coverage" begin
	Random.seed!(31)
	# QR / QRpos / LQ / LQpos / SVD / SDD / Polar: matrix kw form and tensor groupings
	A = randn(ComplexF64, 8, 5)
	Ac = copy(A)
	for alg in (QR(), QRpos(), SVD(), SDD(), Polar())
		q, r = leftorth(A; alg=alg)
		@test A == Ac                       # input not modified
		@test q * r ≈ A
		@test q' * q ≈ Matrix(I, 5, 5)
		q, r = leftorth!(copy(A), alg)
		@test q' * q ≈ I
		@test q * r ≈ A
	end
	# rightorth needs cols >= rows for Polar; use a wide matrix
	Aw = randn(ComplexF64, 5, 8)
	for alg in (LQ(), LQpos(), SVD(), SDD(), Polar())
		l, q = rightorth(Aw; alg=alg)
		@test q * q' ≈ I
		@test l * q ≈ Aw
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

	# SVD-based orthogonalization with atol truncates small singular values
	B = Diagonal([1.0, 1.0e-12, 1.0])
	Qb, Rb = leftorth(Matrix(B); alg=SVD(), atol=1.0e-9)
	@test size(Qb, 2) == 2
	@test Qb * Rb ≈ B

	# tensor leftorth / rightorth groupings
	T = randn(4, 3, 2)
	Tc = copy(T)
	uu, vv = leftorth(T, (1, 2), (3,))
	@test T == Tc
	sdim = size(vv, 1)
	@test size(uu) == (4, 3, sdim)
	@test size(vv) == (sdim, 2)
	@test reshape(uu, :, sdim)' * reshape(uu, :, sdim) ≈ Matrix(I, sdim, sdim)
	@test reshape(reshape(uu, :, sdim) * reshape(vv, sdim, :), 4, 3, 2) ≈ Tc
	T = randn(4, 3, 2)
	Tc = copy(T)
	uu, vv = rightorth(T, (1,), (2, 3))
	@test T == Tc
	sdim = size(vv, 1)
	@test size(uu) == (4, sdim)
	@test size(vv) == (sdim, 3, 2)
	@test reshape(vv, sdim, :) * reshape(vv, sdim, :)' ≈ Matrix(I, sdim, sdim)
	@test reshape(reshape(uu, :, sdim) * reshape(vv, sdim, :), 4, 3, 2) ≈ Tc

	# 4-leg tensor factorization groupings
	T4 = randn(3, 4, 5, 6)
	q, r = leftorth!(copy(T4), (1, 2), (3, 4))
	Tq, Tr = tie(q, (2, 1)), tie(r, (1, 2))
	@test Tq' * Tq ≈ I
	@test Tq * Tr ≈ tie(T4, (2, 2))
	l, q2 = rightorth!(copy(T4), (1,), (2, 3, 4))
	Tq2 = reshape(q2, size(q2, 1), :)
	@test Tq2 * Tq2' ≈ I
	@test reshape(l * Tq2, size(T4)) ≈ T4

	# tsvd! (tensor version with left/right grouping)
	T5 = randn(3, 4, 5, 6)
	u5, s5, v5, err5 = tsvd!(copy(T5), (1, 2), (3, 4))
	@test size(u5) == (3, 4, length(s5))
	@test size(v5) == (length(s5), 5, 6)
	@test err5 < 1e-13

	# helpers
	@test scalar(fill(2.0)) == 2.0
end
