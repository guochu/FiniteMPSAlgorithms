module Defaults
const D         = 64        # default maximum bond dimension
const tolgauge  = 1.0e-14   # truncation precision for gauging/construction
const tol       = 1.0e-12   # algorithm convergence precision
const maxiter   = 100
const verbosity = 1
end

const DefaultTruncation = truncdimcutoff(D = Defaults.D, ϵ = Defaults.tolgauge; add_back = 0)

# default truncation of the canonical-form routines (`truncate!`/`canonicalize!`): a pure
# relative-spectrum cutoff, no dimension cap — gauging must not silently cap bond dimensions
const DefaultOrthTruncation = truncrelerr(ϵ = Defaults.tolgauge)
