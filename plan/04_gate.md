# 04 gate.jl：AbstractGate / UnitaryGate / GenerateGate + apply!（基本 TEBD 构件）

TEBD 只提供最小构件：**门类型层级 + 一个 `apply!` 函数**，定义在 `src/algorithms/gate.jl`。
不定义 QuantumCircuit、Trotter 编排、Stepper、fuse、swap 等结构；时间演化由调用方循环施加门。

## 4.1 门类型层级

```julia
abstract type AbstractGate{N,T}
# 公共接口:
positions(g)::NTuple{N,Int}          # 支撑格点(构造时排序, op 相应 permute)
operator(g)::Array{T,2N}             # N 体算符的 2N 阶张量; 2 体约定 (i1', i2', i1, i2)
shift(g, i)                          # 平移 positions
Base.adjoint(g)

# 幺正门: 构造时检查输入是否幺正
struct UnitaryGate{N,T} <: AbstractGate{N,T}
    positions::NTuple{N,Int}
    op::Array{T,2N}
    # 内部构造器: 检查 op'*op ≈ I(相对容差), 否则 ArgumentError
end
UnitaryGate(positions::NTuple{N,Int}, op::Array{T,2N})
UnitaryGate(positions::Pair{Int,Int}, op::AbstractMatrix)   # 便利: d²×d² 矩阵 → reshape(d,d,d,d)
Base.adjoint(g::UnitaryGate) = UnitaryGate(g.positions, permutedims(conj(g.op), ...))

# 生成元门(生成元生成的演化门): 不检查幺正
struct GenerateGate{N,T} <: AbstractGate{N,T}
    positions::NTuple{N,Int}
    op::Array{T,2N}
end
GenerateGate(positions::NTuple{N,Int}, op::Array{T,2N})
GenerateGate(positions::NTuple{N,Int}, gen::Array{T,2N}, dt::Number)   # op = exp(gen*dt)
GenerateGate(positions::Pair{Int,Int}, gen::AbstractMatrix, dt::Number)
# 典型: GenerateGate(1=>2, -im*h_local, dt) ≈ e^{-i·h_local·dt}(数值上可能轻微非幺正, 故不做检查)
```

## 4.2 施加门与位点交换

```julia
apply!(g::AbstractGate{2}, ψ::CanonicalMPS; trunc=DefaultTruncation)
#   仅支持近邻(相邻两格点) 2-体门, 对 AbstractGate 统一分派。
#   把左键 Schmidt 谱 psi.s[i] 收缩进两站点张量⊗gate, tsvd! 截断后回写,
#   新键谱写入 psi.s[i+1] —— 保持右正则形式, 避免显式移动规范中心;
#   若 ψ.s 未初始化, 先 canonicalize!。

swap!(ψ::CanonicalMPS, i::Integer; trunc=DefaultTruncation)
#   交换相邻位点 i 与 i+1(保持右正则 + psi.s):
#   左键谱 s[i] 收缩进两站点张量, tsvd! 分组 (1,3)|(2,4) 截断,
#   新键谱写入 s[i+1]; 截断误差折算入 scaling。非近邻操作 = 序列化 swap!。
```

## 4.3 典型用法（TEBD 时间演化，调用方编排）

```julia
g_even = GenerateGate(1=>2, -im * h_even, dt)      # e^{-i·h_even·dt}
g_odd  = GenerateGate(2=>3, -im * h_odd,  dt)

for t in 0:dt:T                                    # 一阶 Trotter 示例
    apply!(g_even, ψ; trunc)
    apply!(g_odd,  ψ; trunc)
end
```

## 数值检验清单（M5 完成判据）

1. `UnitaryGate` 对非幺正输入抛 `ArgumentError`；对幺正输入正常构造。
2. 无截断单门 `apply!`（两种门）与稠密计算（`reshape` + 矩阵乘）一致。
3. 截断误差 ≤ trunc 给出的上界；保持右正则与 `psi.s` 一致性。
4. 手动循环施加门做小系统演化，与 TDVP / `exp(-i·H·t)` 精确演化对比（D 充足时差异 < 1e-6）。
