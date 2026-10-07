# ---------- test model: non-uniform complex next-nearest-neighbour chain ----------
#
#   H = -Σ_i h_i σx_i - Σ_i J1_i σz_i σz_{i+1} - Σ_i J2_i σy_i σy_{i+2}
#
# The fields `h_i` and nearest-neighbour couplings `J1_i` differ from site to site (no
# translation symmetry); the next-nearest `σy⊗σy` terms have complex matrix entries, so
# the model exercises ComplexF64 arithmetic end-to-end while keeping H Hermitian
# (σy⊗σy is real symmetric and all coefficients are real).

const _SX = Float64[0 1; 1 0]
const _SZ = Float64[1 0; 0 -1]
const _SY = ComplexF64[0 -im; im 0]

"""
        model_params(L) -> (hs, J1, J2)

Deterministic non-uniform parameters of the test model on `L` sites.
"""
model_params(L::Int) = (
        hs = [0.65 + 0.8 * sin(1.3 * i + 0.2) for i in 1:L],
        J1 = [0.45 + 0.35 * cos(0.9 * i) for i in 1:L-1],
        J2 = [0.25 * sin(1.7 * i + 0.6) for i in 1:L-2],
)

# dense reference matrix of the model
function dense_model(p)
        L = length(p.hs)
        op(ops...) = reshape(kron(ops...), 2^L, 2^L)
        I2 = Matrix{ComplexF64}(I, 2, 2)
        H = zeros(ComplexF64, 2^L, 2^L)
        for i in 1:L
                H .-= p.hs[i] .* op(ntuple(k -> k == i ? _SX : I2, L)...)
        end
        for i in 1:L-1
                H .-= p.J1[i] .* op(ntuple(k -> (k == i || k == i+1) ? _SZ : I2, L)...)
        end
        for i in 1:L-2
                H .-= p.J2[i] .* op(ntuple(k -> (k == i || k == i+2) ? _SY : I2, L)...)
        end
        return H
end

# MPOHamiltonian assembled from product terms (OpTerm route, Schur form)
function mpo_model(p)
        L = length(p.hs)
        ds = fill(2, L)
        terms = OpTerm(-p.hs[1], 1 => _SX)
        for i in 2:L
                terms += OpTerm(-p.hs[i], i => _SX)
        end
        for i in 1:L-1
                terms += OpTerm(-p.J1[i], i => _SZ, i + 1 => _SZ)
        end
        for i in 1:L-2
                terms += OpTerm(-p.J2[i], i => _SY, i + 2 => _SY)
        end
        return MPOHamiltonian(terms)
end

# test-side wrap of a dense 4-index chain into the Schur-form MPOHamiltonian — the
# former core constructor, kept here because it is only needed when comparing the
# sparse and dense representations. The dense 4-index layout (wl, p_out, wr, p_in) is
# the block layout of the Schur form. Boundary-shaped sites (the 1×n vacuum row of the
# first site / the n×1 closing column of the last site, in the tompotensors convention)
# are padded back to the full logical shape: the padded channels are never reached (the
# boundary environments select row 1 / column space_r) and are dropped again by
# tompotensors. The corner blocks (1,1) and (wl,wr) must be proportional to the
# identity and are normalized to the implied identity — for sums of Schur-convention
# chains the corner scalars are exactly the spurious vacuum/closing path counts, so
# the normalization is exact.
function hamiltonian(data::AbstractVector{<:MPOTensor})
	return MPOHamiltonian([_sparse_from_dense(W) for W in data])
end
hamiltonian(h::MPO) = hamiltonian(h.data)

function _sparse_from_dense(W::Array{T,4}) where {T<:Number}
	wl, dout, wr, din = size(W)
	(dout == din) ||
		throw(DimensionMismatch("SchurMPOTensor requires square local operators"))
	m, n = max(wl, 2), max(wr, 2)
	Os = Array{Union{Matrix{T},T},2}(fill(zero(T), m, n))
	Os[1, 1] = Os[m, n] = one(T)
	Id = Matrix{T}(I, dout, dout)
	for j in 1:wr, i in 1:wl
		v = W[i, :, j, :]
		iszero(v) && continue
		if wl == 1 && wr == 1
			# a single 1×1 site is pure content: the D corner
			Os[1, 2] = v
			continue
		end
		if (wr > 1 && i == 1 && j == 1) || (wl > 1 && i == wl && j == wr)
			# the vacuum diagonal (1,1) exists only while the vacuum column is present
			# (wr > 1: on the last site column 1 is the closing column, so its (1,1) is
			# the C content); the closing diagonal (wl,wr) exists only while the vacuum
			# row is present (wl > 1: on the first site row 1 carries the D content):
			# normalize a scalar·identity to the implied identity
			c = tr(v) / dout
			v ≈ c * Id ||
				throw(ArgumentError("the corner blocks (1,1) and (wl,wr) of a Schur-form chain must be proportional to the identity"))
			continue
		end
		# a boundary-shaped last site stores its closing column as column 1: remap it to
		# the padded closing column
		Os[i, wr == 1 ? n : j] = v
	end
	return SchurMPOTensor{T}(Os, dout)
end

# matrix in the Kronecker convention (i1 i2)',(i1 i2), i1 slowest -> documented gate
# tensor (i1', i2', i1, i2)
function gate_tensor(M::AbstractMatrix)
        d2r, d2c = size(M)
        dr, dc = isqrt(d2r), isqrt(d2c)
        (dr^2 == d2r && dc^2 == d2c) || throw(ArgumentError("dimensions must be perfect squares"))
        return permutedims(reshape(M, dr, dc, dr, dc), (2, 1, 4, 3))
end

# the same model assembled per term: canonical Schur chains → dense chains in the
# tompotensors convention → exact dense sum (the corner path counts of the sum are
# normalized by the `MPOHamiltonian(::MPO)` wrap)
function mpo_model_prod(p)
        L = length(p.hs)
        ds = fill(2, L)
        H = MPO(tompotensors(MPOHamiltonian(ds, OpTerm(-p.hs[1], 1 => _SX))))
        for i in 2:L
                H = H + MPO(tompotensors(MPOHamiltonian(ds, OpTerm(-p.hs[i], i => _SX))))
        end
        for i in 1:L-1
                H = H + MPO(tompotensors(MPOHamiltonian(ds, OpTerm(-p.J1[i], i => _SZ, i + 1 => _SZ))))
        end
        for i in 1:L-2
                H = H + MPO(tompotensors(MPOHamiltonian(ds, OpTerm(-p.J2[i], i => _SY, i + 2 => _SY))))
        end
        return hamiltonian(H)
end
