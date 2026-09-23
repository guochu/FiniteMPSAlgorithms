module FiniteMPSAlgorithms

using LinearAlgebra
using LinearAlgebra: BlasFloat
using TensorOperations
import TensorOperations: scalartype
using MatrixAlgebraKit: MatrixAlgebraKit, left_orth!, right_orth!, svd_compact!,
	QRIteration, DivideAndConquer, SafeDivideAndConquer, TruncatedAlgorithm, trunctol,
	LeftOrthAlgorithm, RightOrthAlgorithm, diagview, isunitary
using KrylovKit: KrylovKit, eigsolve, exponentiate

# tensorops
export TruncationScheme, NoTruncation, TruncateDim, TruncateRelError, TruncateDimCutoff,
	truncdim, truncrelerr, truncdimcutoff, truncate!,
	QR, QRpos, LQ, LQpos, SVD, SDD, Polar,
	tsvd!, tsvd, leftorth!, leftorth, rightorth!, rightorth,
	tie, permute, scalar, isometry, renyi_entropy, distance, distance2, fidelity, infidelity
# structures
export AbstractMPS, AbstractMPO, MPSTensor, MPOTensor, CanonicalMPS, CanonicalMPO,
	MPO, MPOHamiltonian, Orthogonalize,
	AbstractSparseMPOTensor, SparseMPOTensor, SchurMPOTensor, SparseMPOHamiltonian,
	OpTerm, OpSum, term, tompotensors,
	todense, vectorize, devectorize, superoperator,
	ExpDecayOpTerm, ExpDecayOpSum,
	isleftcanonical, isrightcanonical, iscanonical, canonicalize!, canonicalize,
	space_l, space_r, phydim, ophydim, iphydim, phydims,
	ophydims, iphydims, bonddim, bonddims,
	scaling, setscaling!, svectors_uninitialized, unset_svectors!, normalize_tr!,
	expectation, expectationvalue, entanglement_entropy, entanglement_spectrum, schmidt_values,
	prodmps, randommps, DensityOperator, infinite_temperature_state,
	changebond!, truncate!, identitympo, prodmpo, randommpo, hadamard, ⊙,
	copyphydims,
	tomps, tompo
# algorithms
export mult, add, compress, compress!,
	svdguess_mult, svdguess_add, svdguess_compress, svdguess_hadamard,
	mult!, add!, linsolve!, hadamard!,
	ground_state, ground_state!, excited_state, excited_state!,
	leftsweep!, rightsweep!, sweep!,
	SVDCompression, DMRG1, DMRG2, TDVP1, DefaultMultAlg, Defaults, IterativeMPSAlgorithm,
        SingleSiteUpdate, TwoSiteUpdate, ALSLinSolve2,
        Seq2Seq, ALSLinSolve, DefaultLinearSolver,
	ac_prime, c_prime, Heff,
	FirstOrderStepper, SecondOrderStepper,
	iterative_compute!, AddCache, MultCache, HadamardCache, LinsolveCache, Seq2SeqCache,
	WI, WII, ComplexStepper, complex_stepper, timeevompo, timeevolve!,
	linsolve, seq2seq, seq2seq!, sample, amplitude,
	DMRGCache, ExcitedStateCache, OverlapCache, recalculate!,
	TraceCache,
	# experimental
        PDMRG, CADMRG, PositiveMPA, ThermalDMRGCache,
        randompmpa, thermalstate, thermalstate!, freeenergy, tr,
        CAMPS, CliffordGate, PauliTerm,
        ALSRecon, ALSReconCache, reconstruct, reconstruct!
# gates
export AbstractGate, UnitaryGate, GeneralGate, apply!, swap!, positions, shift

include("tensorops/tensorops.jl")
include("defaults.jl")
include("abstractdefs.jl")
include("states/states.jl")
include("operators/operators.jl")
include("algorithms/algorithms.jl")

end # module
