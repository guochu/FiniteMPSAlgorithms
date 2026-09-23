using FiniteMPSAlgorithms
using Test, LinearAlgebra, Random

# Truncation schemes, SVD compression and related utilities

@testset "truncation schemes          " begin
	@test truncdim(3) isa TruncateDim
	@test truncdim(D=4) isa TruncateDim
	@test truncdim(D=4).D == 4
	@test truncrelerr(ϵ=1.0e-3) isa TruncateRelError
	@test truncrelerr(ϵ=1.0e-3).ϵ == 1.0e-3
	@test truncdimcutoff(D=5, ϵ=1.0e-3) isa TruncateDimCutoff
	@test truncdimcutoff(5, 1.0e-3) isa TruncateDimCutoff
	@test NoTruncation() isa TruncationScheme
	@test truncrelerr(1.0e-8) isa TruncateRelError
	@test truncrelerr(1.0e-8).ϵ == 1.0e-8

	a = randn(6, 5)
	# tsvd! destroys its input, use the non-mutating tsvd since a is reused below
	u, s, v, err = tsvd(a, trunc=truncdim(3))
	@test length(s) == 3
	@test err ≈ norm(svdvals(a)[4:end])
	u2, s2, v2, err2 = tsvd(a, trunc=truncrelerr(ϵ=1.0e-10))
	@test length(s2) == 5
	@test err2 < 1.0e-8
	u3, s3, v3, err3 = tsvd(a, trunc=truncdimcutoff(D=2, ϵ=1.0e-10))
	@test length(s3) == 2
	@test err3 > 0
	u4, s4, v4, err4 = tsvd(a)
	@test length(s4) == 5
	@test err4 == 0.0
	@test u4 * Diagonal(s4) * v4 ≈ a
	# tsvd! on a copy gives the same result
	u5, s5, v5, err5 = tsvd!(copy(a), trunc=truncdim(3))
	@test s5 ≈ s

	# truncate! public wrapper
	v = collect(1.0:-1e-2:0.0)
	v2, e = truncate!(copy(v), TruncateDim(50))
	@test length(v2) == 50
end

# Fix the truncation semantics (cf. Z2Tensors/test/tensors.jl): ϵ of `truncrelerr`
# and `truncdimcutoff` is measured on the *normalized* vector of singular values,
# i.e. a singular value σᵢ (sorted decreasingly) is kept iff σᵢ > ϵ·‖σ‖₂.
@testset "truncation semantics        " begin
	a = randn(8, 6)
	s = svdvals(a)                # decreasing
	n = norm(s)                   # ‖σ‖₂

	# relerr keeps exactly the singular values with σᵢ > ϵ·‖σ‖₂ ...
	ϵ = 0.2
	d = count(>(ϵ * n), s)
	u1, s1, v1, err1 = tsvd(a, trunc=truncrelerr(ϵ))
	@test length(s1) == d
	@test all(>(ϵ * n), s1)
	# ... which is the same as keeping the largest `d` singular values
	u2, s2, v2, err2 = tsvd(a, trunc=truncdim(d))
	@test s1 == s2 && err1 ≈ err2
	# on a pre-normalized matrix, relerr(ϵ) is an absolute cutoff at ϵ
	an = a ./ n
	sn = svdvals(an)
	_, s3, _, _ = tsvd(an, trunc=truncrelerr(ϵ))
	@test s3 ≈ sn[sn .> ϵ]
	# strict, stable boundary: ϵ → nextfloat(ϵ) drops exactly the marginal values
	_, s4, _, _ = tsvd(a, trunc=truncrelerr(nextfloat(ϵ)))
	@test length(s4) == count(>(nextfloat(ϵ) * n), s)

	# dimcutoff with a non-binding D is exactly relerr; its error is *relative*
	u5, s5, v5, err5 = tsvd(a, trunc=truncdimcutoff(D=20, ϵ=ϵ))
	_, s6, _, err6 = tsvd(a, trunc=truncrelerr(ϵ))
	@test s5 == s6
	@test err5 ≈ norm(s[d+1:end]) / n      # relative truncation error
	@test err6 ≈ norm(s[d+1:end])          # relerr reports the absolute tail norm

	# add_back keeps at least that many singular values, but never more than D
	_, s7, _, _ = tsvd(a, trunc=truncdimcutoff(D=10, ϵ=0.9, add_back=3))
	@test length(s7) == max(3, count(>(0.9 * n), s))
	_, s8, _, _ = tsvd(a, trunc=truncdimcutoff(D=2, ϵ=1.0e-16, add_back=5))
	@test length(s8) == 2

	# truncdim error: the 2-norm of the discarded tail; NoTruncation: nothing dropped
	_, s9, _, err9 = tsvd(a, trunc=truncdim(4))
	@test err9 ≈ norm(s[5:end])
	_, st, _, errt = tsvd(a)
	@test length(st) == 6 && errt == 0.0

	# keyword constructors agree with the convenience functions
	@test TruncateRelError(ϵ=0.1) == truncrelerr(0.1)
	@test TruncateDimCutoff(D=5, ϵ=0.1) == truncdimcutoff(D=5, ϵ=0.1)

	# tsvd! (in place) agrees with tsvd
	ub, sb, vb, errb = tsvd!(copy(a), trunc=truncdimcutoff(D=4, ϵ=1.0e-3))
	_, snb, _, errnb = tsvd(a, trunc=truncdimcutoff(D=4, ϵ=1.0e-3))
	@test sb == snb && errb ≈ errnb
end

@testset "renyi_entropy             " begin
	v = [0.25, 0.75]
	@test renyi_entropy(v) ≈ -(0.25 * log(0.25) + 0.75 * log(0.75))
	@test renyi_entropy(v; α=2) ≈ -log(0.25^2 + 0.75^2)
	@test renyi_entropy([1.0]) == 0.0
	@test_throws ArgumentError renyi_entropy([0.5, 0.6])     # not normalized
	@test_throws ArgumentError renyi_entropy([-0.5, 1.5])    # negative entries
	# on normalized squared singular values
	u, s, v2, _ = tsvd!(randn(5, 5))
	p = s.^2 ./ sum(s.^2)
	@test renyi_entropy(p) > 0
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

@testset "SVDCompression constructor   " begin
	# `trunc` of `SVDCompression` accepts any TruncationScheme
	schemes = (truncdimcutoff(D=10, ϵ=1.0e-8), truncdim(10), truncrelerr(ϵ=1.0e-8),
			   truncrelerr(1.0e-8), NoTruncation())
	for trunc in schemes
		@test SVDCompression(trunc=trunc) isa SVDCompression
		@test SVDCompression(trunc=trunc).trunc == trunc
	end
	@test SVDCompression() isa SVDCompression
	@test SVDCompression(trunc=truncdim(4), verbosity=2).trunc == truncdim(4)
end
