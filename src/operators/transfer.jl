# MPO-MPO overlap and trace transfer primitives

"""
	_updateleft(h, WA, WB) -> h′

Left-to-right MPO-MPO overlap transfer `⟨conj(WA)|h|WB⟩`, `h::AbstractMatrix`.
"""
function _updateleft(hold::AbstractMatrix, WA::MPOTensor, WB::MPOTensor)
	@tensor hnew[-1, -2] := conj(WA[1, 2, -1, 3]) * hold[1, 4] * WB[4, 2, -2, 3]
	return hnew
end

"""
	_updateright(h, WA, WB) -> h′

Right-to-left MPO-MPO overlap transfer `⟨conj(WA)|h|WB⟩`.
"""
function _updateright(hold::AbstractMatrix, WA::MPOTensor, WB::MPOTensor)
	@tensor hnew[-1, -2] := conj(WA[-1, 3, 1, 4]) * hold[1, 2] * WB[-2, 3, 2, 4]
	return hnew
end

"""
	_updatetraceleft(v, W) -> v′

Left-to-right trace transfer for `tr(MPO)`: `v::AbstractVector` carries the contracted bond.
"""
function _updatetraceleft(v::AbstractVector, W::MPOTensor)
	@tensor vnew[-1] := v[1] * W[1, 2, -1, 2]
	return vnew
end

"""
	_updatetraceright(v, W) -> v′

Right-to-left trace transfer for `tr(MPO)`.
"""
function _updatetraceright(v::AbstractVector, W::MPOTensor)
	@tensor vnew[-1] := v[1] * W[-1, 2, 1, 2]
	return vnew
end

# block-sparse MPO site tensors: sum the three-chain contraction over the non-zero operator
# blocks (scalar blocks expand to scalar·identity via the sparse-tensor getindex)
function _updateleft(hold::AbstractArray{T,3}, Aj::MPSTensor, W::AbstractSparseMPOTensor, Bj::MPSTensor) where {T}
	hnew = zeros(T, size(Aj, 3), size(W, 2), size(Bj, 3))
	for (wL, wR) in keys(W)
		O = W[wL, wR]
		hslice = hold[:, wL, :]
		@tensor hn[a, c] := conj(Aj[1, p, a]) * hslice[1, 2] * O[p, q] * Bj[2, q, c]
		hnew[:, wR, :] .+= hn
	end
	return hnew
end

function _updateright(hold::AbstractArray{T,3}, Aj::MPSTensor, W::AbstractSparseMPOTensor, Bj::MPSTensor) where {T}
	hnew = zeros(T, size(Aj, 1), size(W, 1), size(Bj, 1))
	for (wL, wR) in keys(W)
		O = W[wL, wR]
		hslice = hold[:, wR, :]
		@tensor hn[a, c] := conj(Aj[a, p, 1]) * hslice[1, 2] * O[p, q] * Bj[c, q, 2]
		hnew[:, wL, :] .+= hn
	end
	return hnew
end
