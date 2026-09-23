# Interface changes

## Dense conversion: `order` keyword renamed to `:msb` / `:lsb`

`todense`, `tomps` and `tompo` take an `order` keyword selecting the endianness of the
merged physical index. The values `:big` / `:little` were renamed to `:msb` / `:lsb`
(most-/least-significant-first); the default behavior is unchanged (`:msb`).

- `order = :msb` — **big-endian / Kronecker order** (default): site 1 is the most
  significant (slowest) index, i.e. the dense object reads like the `kron` product
  `v₁ ⊗ v₂ ⊗ ⋯ ⊗ v_L` of the local factors site by site — the convention of most
  quantum-information texts (endian terminology) and of TT/MPS literature built on
  `kron` (Kronecker terminology).
- `order = :lsb` — **little-endian / column-major flattening**: site 1 is the least
  significant (fastest) index — the natural reading of `vec(reshape(v, ds...))` in
  Julia's column-major memory layout, which is also what a left-to-right TT-SVD sweep
  produces natively.

The docstrings of `todense` (MPS and MPO methods), `tomps` and `tompo` spell out both
readings.

## New: `Base.sum(::CanonicalMPS)`

Sum of all amplitudes `Σ_{i₁,…,i_L} ψ(i₁,…,i_L)` of the represented state (all
physical indices contracted with all-ones vectors), including the `scaling^L` factor.

## New: `copyphydims(::CanonicalMPS) -> CanonicalMPO`

Reinterprets an MPS as an operator chain on the same bond dimension: every site tensor
`A[aL, p, aR]` becomes the physical-space diagonal `W[aL, po, aR, pi] = A[aL, po, aR]·δ_{po,pi}`
(the physical dimensions are copied to both the bra and the ket side). Applying the
result to another MPS with the same bond structure reproduces the element-wise
(Hadamard) product:

```julia
copyphydims(ψA) * ψB == ⊙(ψA, ψB)
```

so a Hadamard product between chains of matching bond dimensions can equivalently be
evaluated as an MPO·MPS multiplication. The `scaling` of `ψ` is carried over; the
Schmidt values of the output are unset (the MPS spectrum does not satisfy the MPO
canonical conditions).

## New: `swap!` for `CanonicalMPO`, `permute!` / `permuted` for chains

- `swap!(ρ::CanonicalMPO, i; trunc)` exchanges the `(p_out, p_in)` content of the
  neighboring sites `i`, `i+1` — the operator-space analog of the MPS `swap!`
  (Hastings fold–SVD–back-project update, aligned with TEMPO/GTEMPO). The
  right-canonical form and the recorded spectra are preserved (up to the truncation
  error).
- `permute!(x, perm; kwargs...)` reorders the site contents of an MPS/MPO chain in
  place so that afterwards site `i` carries the former content of site `perm[i]`
  (matching `Base.permute!` on plain vectors), as the sequence of neighboring swaps
  given by `permutation2swaps`.
- `permuted(x, perm; kwargs...)` is the non-mutating counterpart (returns a new chain).

## Scaling-safe arithmetic for operator chains

The strict arithmetic primitives (`_mul_data`, `_plus_data`, `_apply_data`) now operate
on the raw data only; dispatch wrappers attach the external `scaling` so that exported
operations are correct for `CanonicalMPO` operands:

- `*(hA, hB)`: the result carries the product of the represented chains — a
  `CanonicalMPO` result with `scaling(hA)·scaling(hB)` when both factors are canonical,
  and the canonical factor's scale otherwise (previously the scale of a `CanonicalMPO`
  factor was silently dropped).
- `+(hA, hB)`, `-(hA, hB)`: the `scaling` of `CanonicalMPO` summands is folded into the
  data (canonical + canonical returns a `CanonicalMPO` with `scaling = 1`, mirroring
  the `CanonicalMPS` sum; mixed sums return a plain `MPO` carrying the folded scale).
- `*(h::AbstractMPO, ψ::CanonicalMPS)`: a new `*(h::CanonicalMPO, ψ)` method carries
  `scaling(h)·scaling(ψ)` into the result (previously `h`'s scale was dropped).
- `MPOHamiltonian` mixings route through the same wrappers, so e.g.
  `MPOHamiltonian * CanonicalMPO` now preserves the canonical scale.
- `mult` (SVD route) builds on the scaling-aware product: the represented value of the
  result is exact for scaled canonical operands. Because the canonicalization sweep
  keeps the data normalized (data-normalized gauge), the raw data norms of the exact
  product are absorbed into the output's `scaling` field — the represented value is the
  contract, not the bare operand-scale product.

Scalar multiplication (`*`, `/`, unary `-`) was already scaling-aware via the
gauge-preserving `lmul!`. Scale-aware by construction and unchanged: `dot`, `norm`,
`tr`, `distance`, `fidelity`/`infidelity`, `todense`, `expectation`, `expectationvalue`,
`vectorize`/`devectorize`.

## `superoperator` carries the `CanonicalMPO` scaling

`superoperator` dispatches on the operand type: `MPO` / `MPOHamiltonian` inputs return a
plain `MPO` (they carry no external scale), while a `CanonicalMPO` input returns a
`CanonicalMPO` whose `scaling` field carries the input's external scale — the
represented superoperator is linear in the represented `h`, so the routes
`devectorize(superoperator(h₁; :left) * vectorize(h₂))` and
`devectorize(superoperator(h₂; :right) * vectorize(h₁))` reproduce the represented
product `h₁·h₂` without any manual scale bookkeeping. The scale-free data-level builder
is the internal `_superoperator_data` (underscore-prefixed, per the convention that
data-only internals are explicitly marked).

## Follow-up interface refinements

- `permuted` renamed to `permute`: the non-mutating chain permutation now extends the
  tensor-level `permute` of the tensorops layer (dispatch on the chain argument).
- `copyphydims`: the redundant `unset_svectors!` was dropped — a freshly constructed
  `CanonicalMPO` has uninitialized Schmidt values by construction.
- `DefaultTruncation` now keeps at least one singular value (`add_back = 1`), avoiding
  degenerate zero-dimension bonds when an entire spectrum falls below the cutoff. The
  definition is pinned by a test.
- `OpSum`: the lattice `ds` is stored as a plain `Vector{Int}` (no lattice-length type
  parameter); the terms stay validated against `ds` on construction / `push!`.

## New: `DMRG2` — the two-site variational engine for all iterative arithmetics

`DMRG2(; maxiter, tol, trunc, verbosity)` joins `DMRG1` in `algdefs.jl` as an accepted
algorithm of every iterative entry point: `mult` (MPO·MPS and MPO·MPO),
`mult!`, `add`, `compress`, `hadamard`, `linsolve` (and hence every route built on
them). Instead of single-site ALS updates with a fixed bond profile, the two-site
engine optimizes neighboring site pairs jointly and re-splits them by a truncating SVD
under `alg.trunc`, so the bond dimension adapts during the sweeps — growth where the
environment demands it, truncation where the scheme caps it. The initial guesses are
drawn with the bond cap carried by `alg.trunc` (`_guess_bond`, falling back to
`Defaults.D`).

Implementation notes (src/algorithms/arithmetics/dmrg2.jl):

- the singular values of the two-site re-split are absorbed into the sweep direction
  WITHOUT normalization (as in the single-site ALS gauge moves): the data keeps the
  scale of the local targets through the sweeps, and the drivers only attach the
  operand external scales through the `scaling` field (like DMRG1). The one exception
  is `add`, which restores the physical scale of the (cancellation-prone) sum from the
  problem's KKT eigenvalue β = Σ_n ⟨bra|ket_n⟩ / ⟨bra|bra⟩ (`setscaling!` first, the
  folding `lmul!` second);
- the per-pair loss `norm(target)` is monotone in processing time for exact local
  solves; a `NoTruncation` small-system testset pins the local machinery (targets,
  re-splits, sample/environment bookkeeping) independently of truncation effects;
- `mult(hA::MPOHamiltonian, …, ::DMRG2)` entry points expand to the dense MPO layer
  (the out-of-place route), as with the other engines.

## `seq2seq` gains a dedicated algorithm type `Seq2Seq`

`seq2seq` / `seq2seq!` / `init_seq2seqcache` now take their configuration through the
dedicated algorithm type `Seq2Seq(; maxiter, tol, D, α, verbosity)` (positional
argument, mirroring `ALSRecon`/`DMRG1` usage). The Hilbert-Schmidt ridge strength `α`
moved from the `α` keyword argument into the algorithm definition; the ridge itself
(the seq2seq regularization of the local normal equations, excluded from the reported
loss) is unchanged. The default-arg forms `seq2seq(xs, ys)` and `seq2seq(xs, ys, Seq2Seq(D=4))`
replace the previous `seq2seq(xs, ys; α)` / DMRG1-based signatures.
