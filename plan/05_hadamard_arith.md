# 05 algorithms/hadamard.jl：逐元素除法 hadamard_inv / hadamard_div（Newton–Schulz）

**动机**（来自 MPSFluidDynamics reproduce 实践）：经典 PDE 的 MPS 求解中，非线性项频繁出现
**逐元素除法**——守恒变量恢复 `u = (ρu)/ρ`、`1/ρ`、隐式步的对角解 `x = a/B`。
它无法用 MPO 表达（非线性），是线性代数原语（mult/add/compress）之外唯一
不可回避的逐元素算术。reproduce 中出现过三种实现：二阶 Taylor（mpslbm）、
Newton–Schulz 迭代（tdjm，推荐）、稠密逐点求解（ttt，等价于逐点精确但要求可稠密化）。

**位置约定**：
- 算法类型 `NewtonSchulz` → `src/algorithms/algdefs.jl`；
- `hadamard_inv` / `hadamard_inv!` / `hadamard_div` / `hadamard_div!` →
  `src/algorithms/hadamard.jl`（唯一导出接口）；
- 逐元素乘直接复用 `states/linalg.jl` 的精确 `⊙` 与 `algorithms/hadamard.jl` 的
  `hadamard(ψA, ψB, alg::SVDCompression)`；本文件不重复实现乘法；
- 病态兜底（同一位运算的变分路线）`linsolve(mps2mpo(B), A)` 属 `linsolve` 的对角特化，
  本文档只规定接口语义，不强制实现。

## 5.1 算法配置（追加到 `src/algorithms/algdefs.jl`）

```julia
struct NewtonSchulz
    D::Int                # 每次截断的键维上限
    ϵ::Float64            # 相对截断阈值（truncdimcutoff(D, ϵ; add_back=1)）
    maxiter::Int          # 最大迭代数
    tol::Float64          # 相对残差容差 ‖1 − B⊙r‖ / ‖1‖ ≤ tol 提前退出
    rescale::Bool         # 均值预缩放（冷启动收敛域预处理），默认 true
end
NewtonSchulz(; D=Defaults.D, ϵ=Defaults.tol, maxiter=5, tol=1e-12, rescale=true)
```

`warm`（热启动链）不进 struct，也**不走关键字参数**——它与 `initguess` 同理属于
**调用点的状态**，通过 in-place 版本的首参数传递（见 5.2）。

## 5.2 接口（`src/algorithms/hadamard.jl`，唯一导出接口）

```julia
# allocating 版（内部冷启动）:
hadamard_inv(B::CanonicalMPS, alg::NewtonSchulz = NewtonSchulz())
    -> (r::CanonicalMPS, res::Float64)
hadamard_div(A::CanonicalMPS, B::CanonicalMPS, alg::NewtonSchulz = NewtonSchulz())
    -> (out::CanonicalMPS, res::Float64)

# in-place 版（热启动 = 第一个参数，同时是输出缓冲，内容会被覆盖）:
hadamard_inv!(r::CanonicalMPS, B::CanonicalMPS, alg::NewtonSchulz = NewtonSchulz())
    -> (r, res)                     # r: 入参 = 上一步的 B⁻¹ (热启动), 出参 = 新的 B⁻¹
hadamard_div!(out::CanonicalMPS, A::CanonicalMPS, B::CanonicalMPS,
              alg::NewtonSchulz = NewtonSchulz())
    -> (out, res)                   # out: 输出缓冲; 内部逆为冷启动
```

命名决定：`hadamard_inv` / `hadamard_div`（与既有 `hadamard` 同前缀）。
弃用 `divide`（与标量/矩阵除歧义）、`rdivide`（与 LinearAlgebra 语义冲突）、`hadiv`（不可读）。
**不提供 `⊘` 或 `Base.:/(A, B)` 运算符重载**：`/` 在 Julia 里是"右除线性系统"
（`A/B` 解 `X·B = A`），对裸 `CanonicalMPS` 定义逐元素商会造成类型派发歧义；
需要运算符形态时，应限定到专门的 wrapper 类型。

**热启动的推荐用法**（时间演化循环，逆随步更新并复用）：

```julia
hadamard_inv!(st.rhoinv, st.U[1], alg)             # 逆原地更新（热启动 = 旧逆）
u = hadamard(st.U[2], st.rhoinv, alg)              # 商 = hadamard 复合
eT = hadamard(st.U[4], st.rhoinv, alg)
```

`hadamard_div` / `hadamard_div!` 是一步式便利接口（内部逆为冷启动，仅 `hadamard_div!`
省去商的分配）；需要跨步热启动时走 `hadamard_inv!` + `hadamard` 复合。

## 5.3 数学原理

逐元素（Hadamard）逆 `r = B⁻¹` 满足 `B ⊙ r = 1`。对标量方程 `b·r = 1` 的牛顿迭代

```
r_{k+1} = r_k · (2 − b·r_k)
```

逐元素推广到链（`⊙` 为精确逐元素乘、加法走精确 `+`，每步后按 `truncdimcutoff(D, ϵ; add_back=1)`
截断）：**二阶收敛**（e_{k+1} ≈ C·e_k²），收敛域 |1 − b·r₀| < 1（逐点）。

**均值预缩放**（`rescale=true`）：冷启动 `r₀ = 1` 只覆盖 b ∈ (0,2)。取 `m = ⟨1, B⟩/d^L`
（`dot(ones_mps, B) / d^L`），解 `r̃ = (B/m)⁻¹`，返回 `r = r̃/m`——覆盖一切
`b/m ∈ (0,2)` 的正场。`m == 0`（零场）→ 直接返回零链（`add_back=1` 语义）。

**热启动**：`warm` 链已在前一收敛域内，`maxiter` 可取小值（实践：冷 5 次 / 热 2 次即可，
tdjm 生产数据）。每次迭代的截断误差随 e_k² 一起衰减，热启动下截断损失不累积。

**截断纪律**：每次 `hadamard`/`+` 走 `SVDCompression` 路线；返回链 `scaling == 1`
（与 fma 纪律一致：逆与商的量级全部在 data 中）。残差

```
res = ‖1 − B⊙r‖ / ‖1‖      (每次迭代后计算, 1 为全 1 链)
```

`res ≤ alg.tol` 提前退出；返回最后一次 `res`（注意：截断后 res 不会低于截断误差水平）。

## 5.4 实现要求（清单）

1. **复用既有原语**：迭代体 = `hadamard(r, t, SVDCompression(...))` + `+`（精确块和）+
   标量乘；不新写收缩路径。
2. **`add_back=1`**：全零/近零场（如初始 v ≈ 0）不得把链截成键 0——
   reproduce 中 mpslbm/tdjm 均踩过此坑（包默认 `add_back=0` 的教训）。
3. **scaling 纪律**：标量乘（均值缩放、×2）产生的 `scaling ≠ 1` 在每次 `hadamard`/`add`
   入口折叠（`fma` 语义），返回链必须 `scaling == 1`。
4. **warm 语义**：`hadamard_inv!` 的首参数 `r` 既是热启动初值也是输出缓冲（会被覆盖），
   因此**不得与 `B` 是同一对象（别名）**；`hadamard_inv` 分配新链，不改任何入参。
   均值预缩放只作用于 `B`，不影响 `r` 的量纲约定（`r` 与 B⁻¹ 同量纲）。
5. **复数/符号**：实现按实正场写文档语义；复数/含负场通过吸收符号到 A（`A/B = sign(B)·A/|B|`）
   的方式留给调用方或后续扩展，首版不自动处理。
6. **err/res 返回**：签名返回 `res`（残差），调用方可断言 `res < 1`（发散告警）。

## 5.5 病态兜底（变分路线，接口预留）

Newton–Schulz 要求残差收敛域（逐点 `b ≠ 0` 且预缩放后 `b/m ∈ (0,2)`）。
当 B 含近零点时，同一位运算的**变分**表述是逐元素线性求解：

```
hadamard_div(A, B, HadamardLinsolve())  :=  linsolve(Diag(B), A)
```

其中 `Diag(B)` 是 `mps2mpo(B)` 式的对角 MPO（物理维翻倍），`linsolve` 用 ALS 变分求解
`B ⊙ x = A`。首版**不实现** `HadamardLinsolve`（依赖 linsolve 的对角特化与全局收敛性调参），
仅保留 alg 类型分派位。

## 5.6 测试要点

- **正确性**：随机小链（L=6~8, d=2, 正元素）`todense(hadamard_inv(B)) .* todense(B) ≈ 1`
  （~1e-10，受 χ 截断限制）；`hadamard_div` 同理 `A/B ⊙ B ≈ A`。
- **均值远离 1**：B = 45 + 小扰动（模拟 ρeT），`rescale=true` 收敛、`=false` 冷启动发散。
- **热启动**：`hadamard_inv!(r, B)`（r = 热）2 次迭代与 `hadamard_inv(B)` 冷启动 5 次迭代
  结果一致（~截断水平）；`r` 的旧内容被正确覆盖；`r` 不得与 `B` 别名（可加断言）。
- **零场**：B = 0 链 → 返回零链，不抛错（`add_back=1`）。
- **截断上界**：`‖hadamard_div(A,B) − A./B_dense‖` ≤ D 截断误差上界；返回链 `scaling == 1`。
- **回归**：tdjm 的 `mps_divide` 用例（16×16 pilot，Newton 逆 rel err ~1e-13）作为基准数字。

## 5.7 已知陷阱（来自 reproduce 实践）

1. `add_back=0` 会把零场截成键 0，后续收缩直接崩溃（mpslbm `build_case` 的全零掩码、
   tdjm 初始 v≈0 场都触发过）——截断必须 `add_back=1`。
2. 流动类应用对截断决策混沌敏感：同一运算换 SVD 驱动（gesdd/gesvd）都会改变轨迹，
   比较新旧实现时以 χ 收敛性为准，不要比较逐位轨迹。
3. 时间演化的热启动逆必须随状态一起存档/重载（tdjm 的 `MPSState.rhoinv`），
   否则每步冷启动会放大截断误差。
