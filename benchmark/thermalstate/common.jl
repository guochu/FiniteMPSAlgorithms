# shared environment for the thermal-state benchmarks
using LinearAlgebra
using Logging
using Printf
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
