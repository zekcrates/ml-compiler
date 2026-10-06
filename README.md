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

| op | src | arg | what it does |
|----|-----|-----|--------------|
| `CONST` | — | the value (`int`/`float`/`bool`) | Compile-time constant. |
| `VAR` | — | name (str) | Free symbolic variable / loop-index reference. |
| `PARAM` | — | `ParamArg(name, dtype, size)` | Named kernel parameter. |
| `BUFFER` | — | `ParamArg(name, dtype, size)` | Named global memory buffer. |
| `ALLOC` | — | `ParamArg(name, dtype, size)` | Scratch buffer allocated in the kernel body. |
| `ADD` | `(a, b)` | — | `a + b`. |
| `SUB` | `(a, b)` | — | `a - b`. |
| `MUL` | `(a, b)` | — | `a * b`. |
| `DIV` | `(a, b)` | — | `a / b`. |
| `RECIP` | `(a,)` | — | `1.0 / a`. |
| `NEG` | `(a,)` | — | `-a`. |
| `MAX` | `(a, b)` | — | `max(a, b)`. |
| `EXP2` | `(a,)` | — | `2 ** a`. |
| `LOG2` | `(a,)` | — | `log2(a)`. |
| `CMPNE` | `(a, b)` | — | `a != b`. |
| `CMPLT` | `(a, b)` | — | `a < b`. |
| `FLOORDIV` | `(a, b)` | — | `a // b`. |
| `FLOORMOD` | `(a, b)` | — | `a % b`. |
| `WHERE` | `(cond, then, else)` | — | Ternary select: `cond ? then : else`. |
| `CAST` | `(a,)` | target dtype (`"int"`/`"float"`/`"bool"`) | Convert `a` to `arg`. |
| `CALL` | `(body, arg0, arg1, ...)` | `CallArg(name)` | Function call: binds `body`'s `PARAM`s to the arguments. |
| `AFTER` | `(value, effect)` | — | Sequencing: evaluate `effect` first, then yield `value`. |
| `INDEX` | `(base, idx)` | — | Pointer to `base[idx]`. |
| `LOAD` | `(ptr,)` | — | Dereference an `INDEX` pointer. |
| `STORE` | `(ptr, value)` | — | Write `value` into `ptr`. |
| `RANGE` | `(end,)` | loop variable name (str) | Loop from `0` to `end`. |
| `END` | `(carried_value, range)` | — | Closes the loop opened by `src[1]`. |
| `LINEAR` | nodes in topological order | — | Straight-line program: each child on its own line, in `src` order. |
| `RESHAPE` | `(x,)` | new shape tuple | Reinterpret the same data with a new shape. |
| `SHRINK` | `(x,)` | per-axis `(start, end)` | Crop each axis to `[start, end)`. |
| `FLIP` | `(x,)` | axes to reverse | Reverse the given axes. |
| `PAD` | `(x,)` | per-axis `(left, right)` | Grow each axis by `left + right`. |
| `STACK` | `(x0, x1, ...)` | — | Stack the sources along a new leading axis. |
