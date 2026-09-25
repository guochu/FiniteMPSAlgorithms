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
        terms = OpSum(ds)   # validated term collection
        for i in 1:L
                push!(terms, OpTerm(-p.hs[i], i => _SX))
        end
        for i in 1:L-1
                push!(terms, OpTerm(-p.J1[i], i => _SZ, i + 1 => _SZ))
        end
        for i in 1:L-2
                push!(terms, OpTerm(-p.J2[i], i => _SY, i + 2 => _SY))
        end
        return MPOHamiltonian(terms)
end

# matrix in the Kronecker convention (i1 i2)',(i1 i2), i1 slowest -> documented gate
# tensor (i1', i2', i1, i2)
function gate_tensor(M::AbstractMatrix)
        d2r, d2c = size(M)
        dr, dc = isqrt(d2r), isqrt(d2c)
        (dr^2 == d2r && dc^2 == d2c) || throw(ArgumentError("dimensions must be perfect squares"))
        return permutedims(reshape(M, dr, dc, dr, dc), (2, 1, 4, 3))
end

# the same model assembled from product MPOs and the exact `+` (prodmpo route)
function mpo_model_prod(p)
        L = length(p.hs)
        ds = fill(2, L)
        H = prodmpo(ComplexF64, ds, 1, _SX) * (-p.hs[1])
        for i in 2:L
                H = H + prodmpo(ComplexF64, ds, i, _SX) * (-p.hs[i])
        end
        for i in 1:L-1
                H = H + prodmpo(ComplexF64, ds, [i, i+1], [_SZ, _SZ]) * (-p.J1[i])
        end
        for i in 1:L-2
                H = H + prodmpo(ComplexF64, ds, [i, i+2], [_SY, _SY]) * (-p.J2[i])
        end
        return MPOHamiltonian(H)
end
