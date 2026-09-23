module Defaults
const D         = 64        # default maximum bond dimension
const tolgauge  = 1.0e-14   # truncation precision for gauging/construction
const tol       = 1.0e-12   # algorithm convergence precision
const maxiter   = 100
const verbosity = 1
end

const DefaultTruncation = truncdimcutoff(D = Defaults.D, ϵ = Defaults.tolgauge; add_back = 1)

# default truncation of the canonical-form routines (`truncate!`/`canonicalize!`): a pure
# relative-spectrum cutoff, no dimension cap — gauging must not silently cap bond dimensions
const DefaultOrthTruncation = truncrelerr(ϵ = Defaults.tolgauge)

# the KrylovKit linear-solver wrapper used by the ALS-style local normal-equation
# solves (`linsolve`, `ALSRecon`): the local equation (M + α·R)·z = rhs is
# Hermitian PSD (positive definite for α > 0), so the `CG` wrapper applies. Passed
# positionally to `KrylovKit.linsolve(operator, b, x₀, solver)`. (`Seq2Seq` solves
# the same equation densely with a direct `\`.)
const DefaultLinearSolver = KrylovKit.CG(; maxiter = Defaults.maxiter, tol = Defaults.tol)
