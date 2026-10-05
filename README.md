# ml compiler

A tiny ML compiler
## The UOp node

```python
UOp(op, src=(), arg=None)
```

| field  | meaning                                                                 |
|--------|-------------------------------------------------------------------------|
| `op`   | string tag naming the operation (`"ADD"`, `"LOAD"`, `"RANGE"`, ...)      |
| `src`  | tuple of child `UOp`s                                  |
| `arg`  | op-specific payload: a constant value, a name, a shape, a dtype, ...     |



Payload classes used in `arg`:

- `ParamArg(name, dtype, size=None)` — identity of a named parameter/buffer (used by `PARAM`, `BUFFER`, `ALLOC`)
- `CallArg(name)` — function name (used by `CALL`)

## UOp spec

Conventions: `a`, `b`, `cond` are child UOps from `src`.

### Leaves — values, no computation

| op       | src | arg                     | what it is |
|----------|-----|-------------------------|------------|
| `CONST`  | —   | the value (`int`/`float`/`bool`) | Compile-time constant. dtype = Python type of `arg`; shape `()`. |
| `VAR`    | —   | name (str)              | Free symbolic variable / loop-index reference. dtype `int`, shape `()`. |
| `PARAM`  | —   | `ParamArg(name, dtype, size)` | Named kernel parameter. Looked up by name in `run_elementwise`; rendered indexed in `compile_c`. dtype from `arg`, shape `(size,)`, addrspace `ALU`. |
| `BUFFER` | —   | `ParamArg(name, dtype, size)` | Named global memory buffer (kernel input/output). dtype from `arg`, shape `(size,)`, addrspace `MEM`. |
| `ALLOC`  | —   | `ParamArg(name, dtype, size)` | Scratch buffer allocated inside the kernel body. Same dtype/shape/addrspace as `BUFFER`. |

### ALU — arithmetic / logic

All binary ops take `src = (a, b)`; unary take `src = (a,)`.

| op         | arity | what it computes | dtype |
|------------|-------|------------------|-------|
| `ADD`      | 2     | `a + b` | promoted from operands |
| `MUL`      | 2     | `a * b`; on two bools it is logical AND | promoted |
| `DIV`      | 2     | `a / b` (true division) | — |
| `RECIP`    | 1     | `1.0 / a` | — |
| `NEG`      | 1     | `-a` | inherits `a` |
| `MAX`      | 2     | `max(a, b)`; on two bools it is logical OR | promoted |
| `EXP2`     | 1     | `2 ** a` | inherits `a` |
| `LOG2`     | 1     | `log2(a)` | inherits `a` |
| `CMPNE`    | 2     | `a != b` | `bool` |
| `CMPLT`    | 2     | `a < b` | `bool` |
| `FLOORDIV` | 2     | `a // b` | promoted |
| `FLOORMOD` | 2     | `a % b` | promoted |

Elementwise `ADD`/`MUL` also require both operands to have the same shape (else `ValueError`), and take on that shape.

### Control flow

| op      | src | arg | what it does |
|---------|-----|-----|--------------|
| `WHERE` | `(cond, then, else)` | — | Ternary select: `cond ? then : else`. |
| `CAST`  | `(a,)` | target dtype (`"int"`/`"float"`/`"bool"`) | Convert `a` to `arg`. dtype = `arg`. |
| `CALL`  | `(body, arg0, arg1, ...)` | `CallArg(name)` | Function call. `body` is a UOp containing `PARAM` leaves; `run` binds each `PARAM` (in discovery order) to the matching argument. |
| `AFTER` | `(value, effect)` | — | Sequencing: evaluate `effect` (e.g. a `STORE`) first, then yield `value`. Gives stores a defined order. |

### Memory

| op      | src | what it does | dtype |
|---------|-----|--------------|-------|
| `INDEX` | `(base, idx)` | Pointer to `base[idx]` — `base` is a `BUFFER`/`ALLOC`, `idx` any int expression. Shape `()`. | element dtype of `base` |
| `LOAD`  | `(ptr,)` | Dereference an `INDEX` pointer; yields the loaded value. | inherits `ptr` |
| `STORE` | `(ptr, value)` | Write `value` into `ptr`. Shape `()`. | `void` |

### Loops

| op      | src | arg | what it does | dtype |
|---------|-----|-----|--------------|-------|
| `RANGE` | `(end,)` | loop variable name (str) | Loop from `0` to `end`. `render` uses `src[0]` as the end bound. | `int` |
| `END`   | `(carried_value, range)` | — | Closes the loop opened by `src[1]`; `src[0]` is what the loop yields (e.g. the accumulator buffer). dtype/shape inherited from `src[0]`. | from `src[0]` |

### Program structure

| op       | src | what it does | dtype |
|----------|-----|--------------|-------|
| `LINEAR` | nodes in topological order | A straight-line program: every child is emitted as its own line, in `src` order. Produced by `linearize()`. | `void` |

### Movement / shape ops

Pure view ops: they only redefine `shape` (and pass dtype through from `src[0]`) — no values are touched, and `run`/`simplify` don't handle them.

| op        | arg | what it does |
|-----------|-----|--------------|
| `RESHAPE` | new shape tuple | Reinterpret the same data with a new shape. Total size must match (else `ValueError`). |
| `SHRINK`  | per-axis `(start, end)` | Crop each axis to `[start, end)`. Axes count must match; bounds-checked (`0 <= start <= end <= size`). |
| `FLIP`    | axes to reverse | Mark those axes as reversed. Shape is unchanged. |
| `PAD`     | per-axis `(left, right)` | Grow each axis by `left + right`. |
| `STACK`   | — | Stack N same-shaped sources (all of `src`) along a new leading axis. |

