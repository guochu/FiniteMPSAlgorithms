# direct sampling of configuration amplitudes from a canonical MPS (Born distribution)

"""
	sample(ψ::CanonicalMPS, N::Integer) -> Vector{NTuple{L,Int}}

Draw `N` configuration samples `𝐱 = (x₁, …, x_L)` from `ψ` by direct sequential
sampling: the chain is swept right to left, at each site the conditional distribution
over the local index is obtained from the stored bond spectrum
(`P(x_s, aL | suffix) ∝ s[s][aL]²·‖A_s[aL, x_s, :]·w‖²` with `w` the sampled-suffix
amplitude vector), and one index is drawn per site.

!!! warning "assumptions"
	The function ASSUMES that `ψ` is right-canonical and represents a PURE quantum
	state: samples follow the Born distribution `P(𝐱) = |ψ(𝐱)|²`. It must not be used
	on chains meant as classical (high-dimensional) probability distributions. If the
	Schmidt values are uninitialized, `ψ` is canonicalized in place first (without
	truncation).
"""
function sample(ψ::CanonicalMPS, N::Integer)
	svectors_uninitialized(ψ) && canonicalize!(ψ; alg=Orthogonalize(SVD(), NoTruncation(), false))
	L = length(ψ)
	T = scalartype(ψ)
	samples = Vector{NTuple{L,Int}}(undef, N)
	for n in 1:N
		x = Vector{Int}(undef, L)
		w = ones(T, 1)                     # sampled-suffix amplitude (empty at bond L)
		for s in L:-1:1
			A = ψ[s]
			D, d = size(A, 1), size(A, 2)
			λ = ψ.s[s]                     # spectrum at bond s-1 (boundary ones(1) at s = 1)
			# joint weights over (aL, x_s): λ[aL]²·‖A_s[aL, x_s, :]·w‖²
			m = reshape(A, D * d, size(A, 3)) * w
			p = vec(reshape(abs2.(m), D, d) .* abs2.(λ))
			# one categorical draw by cumulative search (column-major: aL fastest)
			r = rand() * sum(p)
			c = zero(r)
			idx = length(p)
			for i in eachindex(p)
				c += p[i]
				if r <= c
					idx = i
					break
				end
			end
			# the drawn aL is a sampling aid only: the suffix vector contracts the FULL
			# local index (the left weight re-enters through the next bond spectrum)
			x[s] = (idx - 1) ÷ D + 1
			w = reshape(A[:, x[s], :], D, size(A, 3)) * w
		end
		samples[n] = ntuple(i -> x[i], L)
	end
	return samples
end
