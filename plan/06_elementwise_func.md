# 06 algorithms/elementwise.jl：函数对 MPS 的逐点作用 elementwise（经典 PDE 扩展）

**动机**（来自 MPSFluidDynamics reproduce 实践）：非线性双曲/守恒律的 MPS 求解需要
**逐点非线性函数** `g(ψ)`——`ψ²`、`|ψ|`、`exp(ψ)`、限制器、源项反应项等。
它与逐元素除法（05，Newton–Schulz 特例）同属"无法用 MPO 表达的逐元素算术"。

**与"乘以已知场"的区分**（实现完全不同，勿混淆）：
- `out[𝐱] = g[𝐱]·ψ[𝐱]`，g 为**已知**空间函数/掩码 → 直接 `hadamard`（g 的 MPS 一次构造；
  g 逐腿 rank-1 时可直接缩放 core，秩保持精确，如 ttt 的 `tt_angle_mul`）；
- `out[𝐱] = g(ψ[𝐱])`，g **非线性依赖 ψ 的值** → 本文档的主题。量化编码下"场值"分布在
  全部 L 个量子比特上，g(ψ) 是全局张量函数，不存在逐 core 的独立更新。

**位置约定**：
- 算法类型 `Chebyshev`（及预留 `Cross`）→ `src/algorithms/algdefs.jl`；
- `elementwise` → `src/algorithms/elementwise.jl`（唯一导出接口）；
- 底层复用 `hadamard`（05 / states/linalg.jl 的 `⊙`）、`add`、`prodmps`、`todense`；
- TT-cross 黑盒路线仅保留 alg 分派位，首版不实现。

## 6.1 算法配置（追加到 `src/algorithms/algdefs.jl`）

```julia
# Chebyshev 级数路线（默认）
struct Chebyshev
    nterms::Int           # 级数项数（一般 20~40 达机器精度, 光滑 g）
    D::Int                # 每步截断键维上限
    ϵ::Float64            # 相对截断阈值
    dom::Union{Nothing,Tuple{Float64,Float64}}   # ψ 值域 (a,b); nothing → 自动估计
end
Chebyshev(; nterms=30, D=Defaults.D, ϵ=Defaults.tol, dom=nothing)
```

注：**稠密化路线（todense → 逐点求值 → 重编码）不作为算法提供**——调用方需要时
可直接用 `todense` + 构造器自行复合（见 6.7 陷阱 2 的适用范围说明）。

## 6.2 接口（`src/algorithms/elementwise.jl`，唯一导出接口）

```julia
# out[𝐱] = g(ψ[𝐱]), g::逐点标量函数
elementwise(g, ψ::CanonicalMPS, alg::Chebyshev = Chebyshev())
    -> (out::CanonicalMPS, err::Float64)

# 黑盒 cross 构造（预留分派位, 首版不实现）:
elementwise(g, ψ::CanonicalMPS, alg::Cross)
```

命名：`elementwise`（逐点作用，与 `hadamard`（乘法）、`hadamard_inv/div`（除法）并列）。
弃用 `apply`（与 Gate `apply!` 冲突）、`map`（与 Base 冲突）、`hadamard_func`（冗长且语义含混）。

## 6.3 实现路线一：Chebyshev 级数（默认，推荐）

**原理**：g 在 [a,b] 上光滑 ⇒ Chebyshev 截断一致逼近（误差随 nₜₑᵣₘₛ 指数衰减）；
切比雪夫多项式有三项递推 `Tₖ₊₁(x) = 2x·Tₖ(x) − Tₖ₋₁(x)`，其中"×x"与"减法"都是
MPS 原语（`⊙` 与 `+`），故整条递推只需逐元素算术。

```
1. 值域: (a, b) = alg.dom 或 todense(ψ) 的 extrema (自动估计)
   线性映射 ψ̃ = (ψ − c)/h,  c = (a+b)/2,  h = (b−a)/2      (精确线性运算)
2. 递推: T₀ = 1链 (prodmps),  T₁ = ψ̃
   Tₖ₊₁ = 2·(ψ̃ ⊙ Tₖ) − Tₖ₋₁          (每次 ⊙ 与 + 后按 truncdimcutoff(D, ϵ; add_back=1) 截断)
3. 求和: out = c₀/2·T₀ + Σₖ₌₁ cₖ·Tₖ,  cₖ = (2/nterms)·Σⱼ g(cos(πj/nterms))·cos(πjk/nterms)
4. err: 级数尾项 |cₖ| (nₜₑᵣₘₛ 之后若干项) 的理论界, 或最后两次 nₜₑᵣₘₛ 的差
```

**要点**：
- ψ̃ 的构造是**精确线性**运算（`+`/标量乘），不引入近似；
- 值域界必须**保守**（包含 |min|,|max|）：截断在 [−1,1] 外的 Chebyshev 级数发散；
- 每步递推的截断误差被下一步的 `⊙ ψ̃` 放大（|ψ̃|≤1 抑制），D 需留裕量；
- `g` 只需支持标量求值（系数表用），递推本身只用 `⊙`/`+`。

## 6.4 特殊函数的专用迭代（复用 05，首版仅逆）

| 函数 | 方法 | 状态 |
|---|---|---|
| 1/ψ | Newton–Schulz（= 05 的 `hadamard_inv`） | 05 计划 |
| √ψ | Denman–Beavers 型迭代或 1/(2s) 标度 + Padé | 预留 |
| exp(ψ) | scaling-and-squaring + Padé（⊗ 平方 = hadamard 自乘） | 预留 |

这些是 Chebyshev 之外的"结构化迭代"：收敛快/秩增长可控，但每种函数需单独推导——
按需逐个补充，不进首版。

## 6.5 TT-cross 黑盒构造（预留分派位）

`elementwise(g, ψ, alg::Cross)`：把 f(𝐱) = g(ψ[𝐱]) 当黑盒，自适应交叉插值直接构造
g(ψ) 的 TT（TT-Toolbox `tt_func`/AMEN-cross 的思路）。优点：免值域估计、免级数推导；
缺点：需实现 cross（FMA 无）、误差非单调。首版不实现，仅保留分派位。

## 6.6 测试要点

- **正确性**：随机小链，g ∈ {ψ², exp, sin, 1/(2+ψ)}，`todense(elementwise(...))` 与
  `g.(todense(ψ))` 对比（Chebyshev nterms=30 → ~1e-10）。
- **nterms 收敛**：解析 g 误差随 nterms 指数下降；|g'| 大的 g 需要更大值域裕量。
- **值域**：dom=nothing 的自动估计与手工 dom 结果一致；ψ 含离群大值时自动估计仍收敛。
- **键维**：bondmax ≤ alg.D；光滑 g 的秩增长温和（每步 ×(D_ψ̃=2) 后截断回 D）。
- **scaling 纪律**：返回链 `scaling == 1`；线性映射 ψ̃ 的标量乘折叠正确。
- **回归**：tdjm 的 `kin = ½(u⊙u + v⊙v)` 用 hadamard 幂的结果作为 ψ² 特例基准；
  mpslbm 的 1/ρ（二阶 Taylor）与 `elementwise(x -> 1/x, ρ, Chebyshev(...))` 对比。

## 6.7 已知陷阱（来自 reproduce 实践）

1. **值域界不足 ⇒ 发散**：Chebyshev 级数在 [−1,1] 外发散；自动估计必须用
   `extrema(todense(ψ))` 并留裕量（或调用方显式给 dom）。
2. **递推的秩漂移**：Tₖ 的有效秩随 k 增长（×ψ̃ 每步翻倍再截断），D 需按目标精度
   预留；对非光滑 g（|ψ|、minmod），级数收敛慢/不收敛——限制器类操作建议保持
   sfv_tt 式稠密混合格式（在调用方稠密求值后重编码），不提供专门算法。
3. **截断 × 递推的误差累积**：与 05 的 Newton–Schulz 同理，每步截断误差被后续
   递推部分抵消（切比雪夫多项式在 [−1,1] 上有界），但 D 不足时误差放大是主失效模式。
4. **g 的解析性**：Chebyshev 逼近率由 g 在椭圆收敛域内的解析性决定；g 仅连续
   （如 |x|）时收敛退化为代数——文档必须写明适用范围。
