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

# block-sparse MPO site tensors: densify and take the dense path (the sparse block
# storage is a wrapper around the same 4-index layout, so this is a copy)
_updateleft(hold::AbstractArray{T,3}, Aj::MPSTensor, W::SchurMPOTensor, Bj::MPSTensor) where {T} =
	_updateleft(hold, Aj, tompotensor(W), Bj)

_updateright(hold::AbstractArray{T,3}, Aj::MPSTensor, W::SchurMPOTensor, Bj::MPSTensor) where {T} =
	_updateright(hold, Aj, tompotensor(W), Bj)
