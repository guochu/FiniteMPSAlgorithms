# shared environment for the thermal-state benchmarks
using LinearAlgebra
using Logging
using Random
using TensorOperations
using FiniteMPSAlgorithms
# disambiguate from LinearAlgebra's factorization objects
using FiniteMPSAlgorithms: SVD, Orthogonalize, NoTruncation, QR, QRpos, LQ, LQpos, SDD, Polar, DefaultTruncation, l_LL, r_RR

include(joinpath(@__DIR__, "..", "..", "test", "helpers.jl"))

function say(args...)
    println(args...)
    flush(stdout)
end

# `changebond!` leaves the chain in whatever gauge the resize produced; the TDVP/DMRG
# sweeps need a canonical one (no truncation, normalize = false)
restore_gauge!(x) = canonicalize!(x; alg=Orthogonalize(SVD(), NoTruncation(), false))
