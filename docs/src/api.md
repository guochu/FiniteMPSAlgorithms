# API reference

## Chain structures and gauging

```@docs
CanonicalMPS
CanonicalMPO
MPO
MPOHamiltonian
OpTerm
OpSum
ExpDecayOpTerm
ExpDecayOpSum
term
tompotensors
leftorth!
rightorth!
canonicalize!
canonicalize
iscanonical
isleftcanonical
isrightcanonical
scaling
setscaling!
```

## Observables

```@docs
expectation
expectationvalue
TraceCache
```

## Initialization

```@docs
randommps
randommpo
prodmps
prodmpo
identitympo
DensityOperator
infinite_temperature_state
increase_bond!
```

## Observables

```@docs
expectation
expectationvalue
entanglement_entropy
entanglement_spectrum
schmidt_values
LinearAlgebra.tr
LinearAlgebra.norm
```

## Chain arithmetic

```@docs
mult
mult!
add
add!
compress
compress!
hadamard
linsolve
linsolve!
hadamard!
svd_add
svdguess_mult
svdguess_add
svdguess_compress
svdguess_hadamard
```

## Algorithms

```@docs
SVDCompression
DMRG1
DefaultMultAlg
Defaults
iterative_compute!
leftsweep!
rightsweep!
sweep!
```

## Ground and excited states

```@docs
ground_state
ground_state!
excited_state
excited_state!
```

## Time evolution

```@docs
TDVP1
timeevompo
timeevolve!
WI
WII
ComplexStepper
complex_stepper
```

## Gates (TEBD)

```@docs
UnitaryGate
GeneralGate
swap!
positions
operator
shift
```

## Seq2seq

```@docs
seq2seq
init_seq2seqcache
Seq2SeqCache
```

## Tensor utilities

```@docs
tsvd
tsvd!
truncate!
TruncationScheme
NoTruncation
TruncateDim
TruncateRelError
TruncateDimCutoff
truncdim
truncrelerr
truncdimcutoff
distance
distance2
renyi_entropy
```
