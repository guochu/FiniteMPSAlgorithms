# environment transfer primitives (stateless @tensor contractions)

"""
	_updateleft(h, Aj, Bj) -> h′

Left-to-right overlap transfer `⟨conj(Aj)|h|Bj⟩` with `h::AbstractMatrix` (bra bond × ket bond).
"""
function _updateleft(hold::AbstractMatrix, Aj::MPSTensor, Bj::MPSTensor)
	@tensor hnew[-1, -3] := conj(Aj[1, 2, -1]) * hold[1, 3] * Bj[3, 2, -3]
	return hnew
end

"""
	_updateright(h, Aj, Bj) -> h′

Right-to-left overlap transfer `⟨conj(Aj)|h|Bj⟩`.
"""
function _updateright(hold::AbstractMatrix, Aj::MPSTensor, Bj::MPSTensor)
	@tensor hnew[-1, -2] := conj(Aj[-1, 3, 1]) * hold[1, 2] * Bj[-2, 3, 2]
	return hnew
end

"""
	_updateleft(h, Aj, W, Bj) -> h′

Left-to-right environment transfer `⟨Aj|W|Bj⟩`; `h::Array{T,3}` with indices
(bra bond, MPO bond, ket bond).
"""
function _updateleft(hold::AbstractArray{T,3}, Aj::MPSTensor, W::MPOTensor, Bj::MPSTensor) where {T}
	# bra's physical index contracts with W's OUTPUT (slot 2), ket's with W's INPUT (slot 4)
	@tensor hnew[-1, -2, -3] := conj(Aj[1, 5, -1]) * hold[1, 2, 3] * W[2, 5, -2, 4] * Bj[3, 4, -3]
	return hnew
end

"""
	_updateright(h, Aj, W, Bj) -> h′

Right-to-left environment transfer `⟨Aj|W|Bj⟩`.
"""
function _updateright(hold::AbstractArray{T,3}, Aj::MPSTensor, W::MPOTensor, Bj::MPSTensor) where {T}
	@tensor hnew[-1, -2, -3] := conj(Aj[-1, 5, 1]) * hold[1, 2, 3] * W[-2, 5, 2, 4] * Bj[-3, 4, 3]
	return hnew
end
