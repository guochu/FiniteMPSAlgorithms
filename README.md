# FiniteMPSAlgorithms

A Julia package of finite matrix-product-state (MPS) and matrix-product-operator (MPO)
algorithms operating on **dense tensors**. Chains are plain dense site-tensor lists; all
heavy contractions are written with `@tensor` and the linear algebra is delegated to
BLAS/LAPACK-backed matrix factorizations.

## Features

- **Chain structures**: `CanonicalMPS` (rank-3 tensors) and `CanonicalMPO` (rank-4
  tensors) with a strict separation from static `MPO` containers; canonical gauging
  (`leftorth!`, `rightorth!`, `canonicalize!`, `canonicalize`), truncating SVDs
  (`tsvd`, `truncate!`) and observable evaluation (`expectation`, `expectationvalue`,
  `entanglement_entropy`, `tr`, `norm`).
- **Arithmetic on chains** with a unified interface:
  - `mult(h, x, alg)` — operator application `h·x` (MPO·MPS and MPO·MPO),
  - `add(chains, alg)` — (truncated) sums of MPS/MPO chains,
  - `compress(x, alg)` — compression of a single chain,
  - `hadamard(ψA, ψB, alg)` / `⊙` — pointwise (site-tensor) products,
  - `linsolve(A, y, alg; D)` — variational solution of `A·x ≈ y`.
  Each function exists in an in-place `f!(out, ...)` variant and dispatches on the
  algorithm object: the one-pass `SVDCompression` route or the variational
  single-site ALS (`DMRG1`) route. On-the-fly SVD initial guesses are provided by
  `svdguess_mult`, `svdguess_add`, `svdguess_compress`, `svdguess_hadamard`.
- **Ground and excited states**: `ground_state`, `excited_state` (DMRG with
  orthogonality projectors).
- **Real and imaginary time evolution**:
  - `TDVP1` — single-site time-dependent variational principle,
  - `timeevolve!` with `WI` / `WII` / `ComplexStepper` — W-matrix (Schur-form)
    evolution of long-range MPO Hamiltonians,
  - TEBD-style gates: `UnitaryGate`, `GeneralGate`, `apply!`, `swap!`.
- **Open quantum systems**: density matrices are `CanonicalMPO`s; the `mult`/`add`
  interface composes arbitrary Lindblad generators (see the example below).
- **Seq2seq fitting**: `seq2seq(xs, ys, alg; α, D)` fits an MPO map to a dataset of
  MPS pairs, following [MPSLearning.jl](https://github.com/guochu/MPSLearning).

## Installation

```julia
julia> ] add https://github.com/guochu/FiniteMPSAlgorithms.git
```

Requires Julia ≥ 1.10. Dependencies: `TensorOperations`, `MatrixAlgebraKit`,
`KrylovKit`.

## Quick start

### Ground state (DMRG)

```julia
using FiniteMPSAlgorithms

L = 8
SX = Float64[0 1; 1 0]; SZ = Float64[1 0; 0 -1]

# transverse-field Ising chain from product terms
terms = vcat([OpTerm(-1.0, i => SX) for i in 1:L],
             [OpTerm(-0.5, i => SZ, i + 1 => SZ) for i in 1:L-1])
H = MPOHamiltonian(L, terms)

E, ψ = ground_state(H, DMRG1(maxiter = 50, tol = 1e-10); D = 32)
```

### Operator application and other chain arithmetic

```julia
# exact product followed by one truncating SVD sweep
ϕ = mult(H, ψ, SVDCompression(truncdim(16)))

# variational ALS route: bond cap D is set by the initial guess
χ = mult(H, ψ, DMRG1(maxiter = 20, tol = 1e-10); D = 16)

s = add([ψ, ψ], SVDCompression(truncdim(64)))        # 2ψ (truncated)
c = compress(χ, DMRG1(maxiter = 20, tol = 1e-10); D = 8)
```

### Time evolution

```julia
ψ = prodmps(ComplexF64, fill(2, L), 1)               # product state
dt, nsteps = -0.1im, 20                              # real time: stepsize is complex
for _ in 1:nsteps
    timeevolve!(ψ, H, dt, WII(); trunc = truncdimcutoff(64, 1e-12))
end
```

### Open systems (Lindblad evolution)

Density matrices are `CanonicalMPO`s and evolve with the same `mult`/`add`
interface, e.g. for `dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})` a
second-order Runge–Kutta step is assembled from `mult` products and `add` sums:

```julia
ρ = infinite_temperature_state(ComplexF64, fill(2, L))   # or DensityOperator(ψ)
Hm = MPO(tompotensors(H))                                # dense operand
Ld = prodmpo(ComplexF64, fill(2, L), 3, SX)              # jump operator at site 3
Dt = prodmpo(ComplexF64, fill(2, L), 3, SX' * SX)

k = add([mult((-im * dt) * Hm, ρ, svd), mult(ρ, (im * dt) * Hm, svd),   # -i dt [H, ρ]
         mult(mult(dt * Ld, ρ, svd), Ld, svd),                          # dt LρL†
         mult(ρ, (-dt / 2) * Dt, svd), mult((-dt / 2) * Dt, ρ, svd)],   # -dt/2 {D, ρ}
        svd)
```

See `test/timeevo.jl` and `test/tebd.jl` for complete Lindblad integrators validated
against exact diagonalization.

### Seq2seq (fit an MPO map to data)

```julia
W, traj = seq2seq(xs, ys, DMRG1(maxiter = 25, tol = 1e-12); α = 0.01, D = 2)
```

minimizes `Σ_n ||W·x_n − y_n||²` over a dataset of MPS pairs `(xs[n], ys[n])`.

## Conventions

- **Index order**: MPS tensors `A[aL, p, aR]`; MPO tensors `W[aL, po, aR, pi]`
  (slot 2 output / slot 4 input). Dense conversions put site 1 at the slowest
  (Kronecker) index.
- **External scale**: a canonical chain carries a per-site scale in its `scaling`
  field; the represented operator/state includes the factor `scaling^L`. The scale is
  *never* materialized as a `scaling^L` power on the data — results of `mult`, `add`,
  `hadamard` inherit it through the output's `scaling` field, which keeps long chains
  with huge norms overflow-safe.
- **Algorithms are positional arguments**: `f(inputs..., alg; kwargs...)`. The
  variational `DMRG1` route takes the bond cap as the `D` keyword; its initial guess is
  drawn by the corresponding `svdguess_*` function.
- **Truncation schemes**: `truncdim(D)`, `truncrelerr(ε)`, `truncdimcutoff(D, ε)`.

## Documentation

All exported functions carry complete docstrings (`?ground_state`, `?mult`, ...),
which are the authoritative reference. The `docs/` folder follows the standard
Documenter.jl layout (`julia --project=docs docs/make.jl` builds the manual; it
requires `Documenter` to be installed).

## Testing

```julia
julia> ] test FiniteMPSAlgorithms
```

The test suite validates every algorithm against exact diagonalization on small
systems, including unitary time evolution (TEBD, WII, TDVP1), open-system Lindblad
evolution, and overflow-safety on long chains (L = 500) with extreme scales.

## Citation

If you use the seq2seq (MPO fitting) functionality of this package in your work, please
cite the paper behind [MPSLearning.jl](https://github.com/guochu/MPSLearning):

```bibtex
@article{Guo2018,
  author  = {Chu Guo and Zhanming Jie and Wei Lu and Dario Poletti},
  title   = {Matrix product operators for sequence to sequence learning},
  journal = {Physical Review E},
  volume  = {98},
  pages   = {012104},
  year    = {2018},
  doi     = {10.1103/PhysRevE.98.012104},
  eprint  = {1803.10908},
  archivePrefix = {arXiv},
}
```
