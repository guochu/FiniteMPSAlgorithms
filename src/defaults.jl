module Defaults
using ..KrylovKit
using ..FiniteMPSAlgorithms: truncdimcutoff, truncrelerr

const D         = 64        # default maximum bond dimension
const tolgauge  = 1.0e-14   # truncation precision for gauging/construction
const tol       = 1.0e-12   # algorithm convergence precision
const maxiter   = 100
const verbosity = 1
const krylovdim = 30        # default Krylov dimension of the local eigenvalue problems

# stub for the default orthogonalization factorization of the gauge moves (attached
# below, after the tensorops layer defines the QR/LQ wrappers)
function alg_orth end

"""
	Defaults.alg_eigsolve(; ishermitian=true, tol=tol, maxiter=200, krylovdim=krylovdim, eager=true)

Eigenvalue solver (a KrylovKit algorithm object): Lanczos for `ishermitian = true`,
Arnoldi otherwise. The local effective-Hamiltonian problems of the variational engines
(`DMRG1`, `DMRG2`, `PDMRG`, `CADMRG`) pass it to `KrylovKit.eigsolve` positionally.
"""
function alg_eigsolve(; ishermitian::Bool = true, tol = tol, maxiter::Int = 200,
					  krylovdim::Int = krylovdim, eager::Bool = true)
	return ishermitian ?
		   KrylovKit.Lanczos(; tol, maxiter, krylovdim, eager) :
		   KrylovKit.Arnoldi(; tol, maxiter, krylovdim, eager)
end

"""
	Defaults.alg_expsolve(; ishermitian=true, tol=tol, maxiter=100, krylovdim=25)

Exponential solver for the local time steps of `TDVP1`/`TDVP2` (passed positionally to
`KrylovKit.exponentiate`): Lanczos for `ishermitian = true`, Arnoldi otherwise.
"""
function alg_expsolve(; ishermitian::Bool = true, tol = tol, maxiter::Int = 100,
					  krylovdim::Int = 25)
	return ishermitian ?
		   KrylovKit.Lanczos(; tol, maxiter, krylovdim) :
		   KrylovKit.Arnoldi(; tol, maxiter, krylovdim)
end

"""
	Defaults.alg_linsolve(; tol=tol, maxiter=maxiter)

Linear solver for the ALS local normal equations (`linsolve`, `ALSRecon`, `ALSRecon2`):
the local equation `(M + α·R)·z = rhs` is Hermitian PSD (positive definite for α > 0),
so the `CG` wrapper applies. Passed positionally to
`KrylovKit.linsolve(operator, b, x₀, alg)`. (`Seq2Seq` solves the same equation densely
with a direct `\\`.)
"""
alg_linsolve(; tol = tol, maxiter::Int = maxiter) = KrylovKit.CG(; tol, maxiter)

"""
	Defaults.alg_trunc()

The package-level default truncation scheme of the iterative engines: capped at `D`,
relative cutoff `tolgauge`, keeping at least one singular value (`add_back = 1`, so a
degenerate zero-dimension bond cannot appear when an entire spectrum falls below the
cutoff).
"""
alg_trunc() = truncdimcutoff(D = D, ϵ = tolgauge; add_back = 1)

"""
	Defaults.alg_orth_trunc()

Default truncation of the canonical-form routines (`truncate!`/`canonicalize!`): a pure
relative-spectrum cutoff, no dimension cap — gauging must not silently cap bond
dimensions.
"""
alg_orth_trunc() = truncrelerr(ϵ = tolgauge)
end

# the default orthogonalization factorization of the gauge moves: QR with a positive
# diagonal; the `rightorth!` moves use the adjoint (`alg'` = LQpos), so one setting
# covers both sweep directions
Defaults.alg_orth() = QRpos()
