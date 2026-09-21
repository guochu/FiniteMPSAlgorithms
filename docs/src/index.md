# FiniteMPSAlgorithms

A Julia package of finite matrix-product-state (MPS) and matrix-product-operator (MPO)
algorithms on dense tensors.

## Overview

- **Chain structures**: [`CanonicalMPS`](@ref) and [`CanonicalMPO`](@ref) with strict
  separation from the static [`MPO`](@ref) container; canonical gauging, truncating
  SVDs and observable evaluation.
- **Chain arithmetic**: [`mult`](@ref) (operator application), [`add`](@ref) (sums),
  [`compress`](@ref), [`hadamard`](@ref) (pointwise products) and
  [`linsolve`](@ref) (variational linear solve), each dispatching on an algorithm
  object (`SVDCompression` one-pass route or `DMRG1` variational ALS route) with
  in-place `!` variants.
- **Ground / excited states**: [`ground_state`](@ref), [`excited_state`](@ref).
- **Time evolution**: `TDVP1` (single-site TDVP), W-matrix evolution
  ([`timeevolve!`](@ref) with `WI` / `WII` / `ComplexStepper`), TEBD gates
  (`UnitaryGate`, `GeneralGate`, [`apply!`](@ref), `swap!`).
- **Open quantum systems**: density matrices as `CanonicalMPO`s; Lindblad generators
  assembled from `mult` / `add`.
- **Seq2seq**: [`seq2seq`](@ref) fits an MPO map to a dataset of MPS pairs.

## Quick start

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

## Conventions

- MPS tensors `A[aL, p, aR]`; MPO tensors `W[aL, po, aR, pi]` (slot 2 output, slot 4
  input). Dense conversions put site 1 at the slowest (Kronecker) index.
- A canonical chain carries a per-site external scale in its `scaling` field; the
  represented state/operator includes `scaling^L`. The scale is never materialized as
  a `scaling^L` power on the data (overflow safety on long chains).
- Algorithms are positional arguments: `f(inputs..., alg; kwargs...)`. The variational
  `DMRG1` route takes the bond cap as the `D` keyword and draws its initial guess from
  the corresponding `svdguess_*` function.
