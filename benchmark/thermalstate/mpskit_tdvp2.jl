# MPSKit reference runs for the two-site TDVP (`TDVP2`) alignment checks of this package.
#
# MPSKit is not a dependency of the benchmark project, so run this file in an environment
# that has MPSKit (e.g. the default project, without `--project=benchmark/thermalstate`):
#
#   julia benchmark/thermalstate/mpskit_tdvp2.jl
#
# It reproduces
#
#  (1) the MPSKit column of `tdvp2_l10.jl`: the L = 10, β = 1 benchmark model (spin-1/2
#      Heisenberg, S = σ/2, J = 1, plus a non-uniform Sz field) cooled with
#      `TDVP2(trscheme = truncrank(D))` from the bond-dimension-1 infinite-temperature
#      state. Our `bench_tdvp2_l10` gives 9.048e-3 / 3.632e-4 / 2.951e-6 for the density
#      matrix error at D = 8/16/32 against the numbers printed below;
#
#  (2) the growth-phase numbers quoted in the TDVP2 testset (`test/algorithms/timeevo.jl`):
#      the same run on the *test* model (σx field + σzσz nearest-neighbour + σyσy
#      next-nearest-neighbour, L = 4). There the first sweep still runs while the bonds
#      grow, its projectors are not yet the full tangent space, and a fixed O(dτ) term
#      remains — the same term our sweeps produce, digit for digit.
#
# Both runs use the vectorized route: a `FiniteMPS` with ONE physical leg of dimension 4
# per site and the doubled generator G = H ⊗ I + I ⊗ Hᵀ. The step increment is purely
# imaginary, `-im·(β/2)/nst`, so that one `timestep` cools each side by β/2/nst.

using MPSKit, TensorKit, LinearAlgebra
using Printf

const V2 = ℂ^2
const V4 = ℂ^4
const I2 = Matrix{ComplexF64}(I, 2, 2)
const I4 = Matrix{ComplexF64}(I, 4, 4)

const _SX = [0.0 1.0; 1.0 0.0]
const _SY = [0.0 -1.0im; 1.0im 0.0]
const _SZ = [1.0 0.0; 0.0 -1.0]

# the (a_i, b_i, a_j, b_j) <-> (a_i, a_j, b_i, b_j) permutation of the doubled pair space
const Q = let
    q = zeros(ComplexF64, 16, 16)
    for ai in 0:1, aj in 0:1, bi in 0:1, bj in 0:1
        q[8ai+4bi+2aj+bj+1, 8ai+4aj+2bi+bj+1] = 1
    end
    q
end
doubled2(o2::Matrix) = Q * (kron(o2, I4) + kron(I4, transpose(o2))) * Q'
doubled1(h::Matrix) = kron(h, I2) + kron(I2, transpose(h))

# ---------------------------------------------------------------- model (1): L = 10
bench_hf(i) = 0.4 * sin(1.3 * i + 0.2)

function bench_model(L)
    # spin-1/2 convention: S = σ/2 (so the field term is hf[i]·Sz, not hf[i]·σz)
    o2 = kron(0.5 .* _SX, 0.5 .* _SX) + kron(0.5 .* _SY, 0.5 .* _SY) +
         kron(0.5 .* _SZ, 0.5 .* _SZ)
    terms = Any[(i, i + 1) => TensorMap(doubled2(o2), V4 ⊗ V4, V4 ⊗ V4) for i in 1:(L - 1)]
    for i in 1:L
        push!(terms, (i,) => TensorMap(doubled1(bench_hf(i) .* (0.5 .* _SZ)), V4, V4))
    end
    return FiniteMPOHamiltonian(fill(V4, L), terms...)
end

function bench_dense(L)
    op_at(op, i) = foldl(kron, [j == i ? op : I2 for j in 1:L])
    Hd = zeros(ComplexF64, 2^L, 2^L)
    for i in 1:(L - 1), op in (0.5 .* _SX, 0.5 .* _SY, 0.5 .* _SZ)
        Hd .+= op_at(op, i) * op_at(op, i + 1)
    end
    for i in 1:L
        Hd .+= bench_hf(i) .* op_at(0.5 .* _SZ, i)
    end
    return Hd
end

# ---------------------------------------------------------------- model (2): L = 4 test model
const TEST_HS = [0.65 + 0.8 * sin(1.3 * i + 0.2) for i in 1:4]
const TEST_J1 = [0.45 + 0.35 * cos(0.9 * i) for i in 1:3]
const TEST_J2 = [0.25 * sin(1.7 * i + 0.6) for i in 1:2]

function test_model(L)
    terms = Any[]
    for i in 1:(L - 1)
        push!(terms, (i, i + 1) =>
            TensorMap(doubled2(-TEST_J1[i] .* kron(_SZ, _SZ)), V4 ⊗ V4, V4 ⊗ V4))
    end
    for i in 1:(L - 2)
        push!(terms, (i, i + 2) =>
            TensorMap(doubled2(-TEST_J2[i] .* kron(_SY, _SY)), V4 ⊗ V4, V4 ⊗ V4))
    end
    for i in 1:L
        push!(terms, (i,) => TensorMap(doubled1(-TEST_HS[i] .* _SX), V4, V4))
    end
    return FiniteMPOHamiltonian(fill(V4, L), terms...)
end

function test_dense(L)
    op_at(op, i) = foldl(kron, [j == i ? op : I2 for j in 1:L])
    Hd = zeros(ComplexF64, 2^L, 2^L)
    for i in 1:L
        Hd .+= -TEST_HS[i] .* op_at(_SX, i)
    end
    for i in 1:(L - 1)
        Hd .+= -TEST_J1[i] .* op_at(_SZ, i) * op_at(_SZ, i + 1)
    end
    for i in 1:(L - 2)
        Hd .+= -TEST_J2[i] .* op_at(_SY, i) * op_at(_SY, i + 2)
    end
    return Hd
end

# ---------------------------------------------------------------- state handling
identity_state(L) =
    FiniteMPS(fill(TensorMap(reshape(ComplexF64[1, 0, 0, 1], (1, 4, 1)), ℂ^1 ⊗ V4, ℂ^1), L))

# contract AC[1] * AR[2:end], i.e. the vectorized state as a dense vector (site 1 slowest)
function dense_vec(psi)
    L = length(psi)
    T = convert(Array, psi.AC[1])
    M = reshape(T, (4, size(T, ndims(T))))
    for i in 2:L
        A = convert(Array, psi.AR[i])
        dl, dr = size(A, 1), size(A, ndims(A))
        M = reshape(M * reshape(A, (dl, 4 * dr)), (size(M, 1) * 4, dr))
    end
    return M[:, 1]
end

# the fused (po, pin) indices back into a 2^L × 2^L density matrix
function rho_of(psi, L)
    R = reshape(dense_vec(psi), ntuple(_ -> 4, L))
    R = permutedims(R, reverse(1:L))
    R = reshape(R, ntuple(_ -> 2, 2L))
    perm = (ntuple(k -> 2k, L)..., ntuple(k -> 2k - 1, L)...)
    return reshape(permutedims(R, perm), (2^L, 2^L))
end

# ---------------------------------------------------------------- runs
function run_tdvp2(H, Hd, L, β, D, nst)
    psi = identity_state(L)
    for _ in 1:nst
        psi, = timestep(psi, H, 0.0, -im * (β / 2) / nst, TDVP2(; trscheme = truncrank(D)))
    end
    ex = exp(Matrix(-β * Hermitian(Hd)))
    ρed = ex / tr(ex)
    r = rho_of(psi, L)
    rn = r / tr(r)
    return (; E = real(tr(Hd * rn)), err = norm(rn - ρed) / norm(ρed),
        bonds = [dim(right_virtualspace(psi, n)) for n in 1:L])
end

println("(1) benchmark model: L = 10, β = 1, nst = 20, D = 8 / 16 / 32")
let
    L = 10
    β = 1.0
    Hd = bench_dense(L)
    ex = exp(Matrix(-β * Hermitian(Hd)))
    ρed = ex / tr(ex)
    @printf("  ED: E = %.12f\n", real(tr(Hd * ρed)))
    H = bench_model(L)
    r0 = rho_of(identity_state(L), L)
    @printf("  sanity: ||rho0/tr(rho0) - I/2^L|| = %.3e\n", norm(r0 / tr(r0) - I(2^L) / 2^L))
    for D in (8, 16, 32)
        r = run_tdvp2(H, Hd, L, β, D, 20)
        @printf("  D=%2d  E=%.12f  err=%.4e  bonds=%s\n", D, r.E, r.err, string(r.bonds))
    end
end

println("\n(2) test model: L = 4, β = 1, D = 32, nst = 20 / 40 / 80")
let
    L = 4
    β = 1.0
    Hd = test_dense(L)
    ex = exp(Matrix(-β * Hermitian(Hd)))
    ρed = ex / tr(ex)
    @printf("  ED: E = %.12f\n", real(tr(Hd * ρed)))
    H = test_model(L)
    for nst in (20, 40, 80)
        r = run_tdvp2(H, Hd, L, β, 32, nst)
        @printf("  nst=%2d  E=%.12f  E rel=%.3e  err=%.4e  bonds=%s\n", nst, r.E,
            abs(r.E - real(tr(Hd * ρed))) / abs(real(tr(Hd * ρed))), r.err, string(r.bonds))
    end
end
