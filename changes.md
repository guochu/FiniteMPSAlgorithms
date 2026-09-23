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
`vectorize`/`devectorize`. `superoperator` remains a data-level utility (documented).

## Follow-up interface refinements

- `permuted` renamed to `permute`: the non-mutating chain permutation now extends the
  tensor-level `permute` of the tensorops layer (dispatch on the chain argument).
- `copyphydims`: the redundant `unset_svectors!` was dropped — a freshly constructed
  `CanonicalMPO` has uninitialized Schmidt values by construction.
- `DefaultTruncation` now keeps at least one singular value (`add_back = 1`), avoiding
  degenerate zero-dimension bonds when an entire spectrum falls below the cutoff. The
  definition is pinned by a test.
- `OpSum`: the lattice `ds` is now part of the type (`OpSum{L}` with
  `ds::NTuple{L,Int}`), so `OpSum`s on different lattices are distinct types. All
  existing vector-based constructors keep working.
