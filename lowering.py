from uops import ParamArg, UOp


def lower_indexed(expr, idxs):
    if len(idxs) == 1:
        i, = idxs

    if expr.op == "PARAM":
        pointer = UOp("INDEX", (expr, i))
        return UOp("LOAD", (pointer,))

    if expr.op == "NEG":
        child = lower_indexed(expr.src[0], idxs)
        return UOp("NEG", (child,))

    if expr.op in ("ADD", "SUB", "MUL", "DIV", "MAX", "CMPNE", "CMPLT", "FLOORDIV", "FLOORMOD"):
        return UOp(expr.op, tuple(lower_indexed(s, idxs) for s in expr.src))

    if expr.op in ("RECIP", "EXP2", "LOG2", "CAST"):
        return UOp(expr.op, (lower_indexed(expr.src[0], idxs),), arg=expr.arg)

    if expr.op == "WHERE":
        return UOp(
            "WHERE",
            tuple(lower_indexed(src, idxs) for src in expr.src),
        )

    if expr.op == "PAD":
        in_idx = []
        for a, (l, r) in enumerate(expr.arg):
            if l == 0:
                in_idx.append(idxs[a])
            else:
                in_idx.append(UOp("ADD", (idxs[a], UOp("CONST", arg=-l))))

        data = lower_indexed(expr.src[0], tuple(in_idx))
        result = data
        for a, (l, r) in reversed(list(enumerate(expr.arg))):
            l_const = UOp("CONST", arg=l)
            zero = UOp("CONST", arg=0)
            if r > 0:
                size = expr.src[0].shape[a]
                limit = UOp("ADD", (l_const, UOp("CONST", arg=size)))
                cond = UOp("CMPLT", (idxs[a], limit))
                result = UOp("WHERE", (cond, result, zero))
            if l > 0:
                cond = UOp("CMPLT", (idxs[a], l_const))
                result = UOp("WHERE", (cond, zero, result))
        return result

    if expr.op == "SHRINK":
        in_shape = expr.src[0].shape
        in_idx = []
        for idx, (l, r) in enumerate(expr.arg):
            if l == 0:
                in_idx.append(idxs[idx])
            else:
                const = UOp("CONST", arg=l)
                in_idx.append(UOp("ADD", (idxs[idx], const)))

        return lower_indexed(expr.src[0], tuple(in_idx))

    if expr.op == "FLIP":
        in_shape = expr.src[0].shape
        flip_set = expr.arg if expr.arg is not None else tuple(range(len(in_shape)))
        in_idx = []
        for a in range(len(in_shape)):
            if a in flip_set:
                size_sub = UOp("CONST", arg=in_shape[a] - 1)
                in_idx.append(UOp("SUB", (size_sub, idxs[a])))
            else:
                in_idx.append(idxs[a])
        return lower_indexed(expr.src[0], tuple(in_idx))

    if expr.op == "EXPAND":
        in_shape = expr.src[0].shape
        out_shape = expr.arg
        in_idx = []
        for a in range(len(out_shape)):
            if in_shape[a] == 1:
                in_idx.append(UOp("CONST", arg=0))
            else:
                in_idx.append(idxs[a])
        return lower_indexed(expr.src[0], tuple(in_idx))

    if expr.op == "REDUCE":
        op, axis = expr.arg
        n = expr.src[0].shape[0]
        allc = UOp("ALLOC", arg=ParamArg("acc", expr.src[0].dtype, 1))

        zero_const = UOp("CONST", arg=0)
        idx = UOp("INDEX", (allc, zero_const))
        old_acc = UOp("LOAD", (idx,))

        if op == "ADD":
            if expr.src[0].dtype == "float":
                start_value = UOp("CONST", arg=0.0)
            else:
                start_value = UOp("CONST", arg=0)
        elif op == "MAX":
            zero_c2 = UOp("CONST", arg=0)
            output = lower_indexed(expr.src[0], (zero_c2,))
            start_value = output
        init = UOp("STORE", (idx, start_value))

        k = UOp("VAR", arg="k")
        bound_const = UOp("CONST", arg=n)
        loop = UOp("RANGE", (bound_const,), arg="k")
        element = lower_indexed(expr.src[0], (k,))

        if op == "ADD":
            combine = UOp("ADD", (old_acc, element))
        elif op == "MAX":
            combine = UOp("MAX", (old_acc, element))

        update = UOp("STORE", (idx, combine))
        end = UOp("END", (allc, loop))
        return UOp("LINEAR", (allc, init, loop, update, end))

    if expr.op == "PERMUTE":
        in_idx = [None] * len(expr.src[0].shape)
        for k, a in enumerate(expr.arg):
            in_idx[a] = idxs[k]

        return lower_indexed(expr.src[0], tuple(in_idx))

    if expr.op == "STACK":
        stack_index = idxs[0]
        inner_idxs = tuple(idxs[1:])
        result = lower_indexed(expr.src[-1], inner_idxs)
        for pos in reversed(range(len(expr.src) - 1)):
            candidate = lower_indexed(expr.src[pos], inner_idxs)
            condition = UOp(
                "CMPLT",
                (stack_index, UOp("CONST", arg=pos + 1)),
            )
            result = UOp(
                "WHERE",
                (condition, candidate, result),
            )
        return result

    if expr.op == "RESHAPE":
        shape = expr.arg
        flat = idxs[-1]
        stride = 1
        for a in reversed(range(len(shape) - 1)):
            stride *= shape[a + 1]
            row_const = UOp("CONST", arg=stride)
            mul_i_row = UOp("MUL", (idxs[a], row_const))
            flat = UOp("ADD", (mul_i_row, flat))
        base = expr.src[0]
        while base.op == "RESHAPE":
            base = base.src[0]
        ptr = UOp("INDEX", (base, flat))
        return UOp("LOAD", (ptr,))
    return expr


def lower(expr):
    return lower_indexed(expr, (UOp("VAR", arg="i"),))


def lower_program_reduce(expr):
    op, axis = expr.arg
    in_shape = expr.src[0].shape
    out_shape = expr.shape
    dtype = expr.src[0].dtype

    names = ["i", "j", "k", "l"][:len(out_shape)]
    idxs = tuple(UOp("VAR", arg=name) for name in names)

    n = 1
    for s in out_shape:
        n *= s
    out = UOp("ALLOC", arg=ParamArg("out", dtype, n))

    kname = "k" if "k" not in names else "m"
    k = UOp("VAR", arg=kname)
    in_idxs = idxs[:axis] + (k,) + idxs[axis:]
    element = lower_indexed(expr.src[0], in_idxs)

    acc = UOp("ALLOC", arg=ParamArg("acc", dtype, 1))
    acc_ptr = UOp("INDEX", (acc, UOp("CONST", arg=0)))
    old_acc = UOp("LOAD", (acc_ptr,))

    if op == "ADD":
        init_val = UOp("CONST", arg=0.0 if dtype == "float" else 0)
        combine = UOp("ADD", (old_acc, element))
    elif op == "MAX":
        first = lower_indexed(expr.src[0], tuple(UOp("CONST", arg=0) for _ in in_idxs))
        init_val = first
        combine = UOp("MAX", (old_acc, element))
    else:
        raise ValueError(f"unsupported reduce op {op}")

    init = UOp("STORE", (acc_ptr, init_val))
    loop = UOp("RANGE", (UOp("CONST", arg=in_shape[axis]),), arg=kname)
    update = UOp("STORE", (acc_ptr, combine))
    end = UOp("END", (acc, loop))
    store_val = UOp("LOAD", (UOp("INDEX", (acc, UOp("CONST", arg=0))),))

    flat = idxs[-1]
    stride = 1
    for a in reversed(range(len(out_shape) - 1)):
        stride *= out_shape[a + 1]
        flat = UOp("ADD", (UOp("MUL", (idxs[a], UOp("CONST", arg=stride))), flat))
    store = UOp("STORE", (UOp("INDEX", (out, flat)), store_val))

    lines = [out]
    ranges = []
    for name, size in zip(names, out_shape):
        loop_o = UOp("RANGE", (UOp("CONST", arg=size),), arg=name)
        ranges.append(loop_o)
        lines.append(loop_o)
    lines.extend([acc, init, loop, update, end, store])
    for loop_o in reversed(ranges):
        lines.append(UOp("END", (out, loop_o)))
    return UOp("LINEAR", tuple(lines))


def lower_program(expr):
    if expr.op == "REDUCE":
        return lower_program_reduce(expr)
    shape = expr.shape

    names = ["i", "j", "k", "l"][:len(shape)]
    idxs = tuple(UOp("VAR", arg=name) for name in names)

    n = 1
    for s in shape:
        n *= s
    out = UOp("ALLOC", arg=ParamArg("out", "int", n))

    value = lower_indexed(expr, idxs)

    flat = idxs[-1]
    stride = 1
    for a in reversed(range(len(shape) - 1)):
        stride *= shape[a + 1]
        flat = UOp("ADD", (UOp("MUL", (idxs[a], UOp("CONST", arg=stride))), flat))
    store = UOp("STORE", (UOp("INDEX", (out, flat)), value))

    lines = [out]
    ranges = []
    for name, idx, size in zip(names, idxs, shape):
        loop = UOp("RANGE", (UOp("CONST", arg=size),), arg=name)
        ranges.append(loop)
        lines.append(loop)
    lines.append(store)
    for loop in reversed(ranges):
        lines.append(UOp("END", (out, loop)))
    return UOp("LINEAR", tuple(lines))


def matmul(A, B):
    M, K = A.shape
    K2, N = B.shape
    if K2 != K:
        raise ValueError("matmul: inner dimensions don't match")

    a3 = UOp("EXPAND", (UOp("RESHAPE", (A,), arg=(M, 1, K)),), arg=(M, N, K))
    bt = UOp("PERMUTE", (UOp("RESHAPE", (B,), arg=(1, N, K)),), arg=(0, 2, 1))
    b3 = UOp("EXPAND", (bt,), arg=(M, N, K))

    return UOp("REDUCE", (mul(a3, b3),), arg=("ADD", 2))


def broadcast(a, b):
    sa, sb = a.shape, b.shape
    if len(sa) != len(sb):
        raise ValueError("broadcast: rank mismatch")

    out = []
    for x, y in zip(sa, sb):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            raise ValueError(f"broadcast mismatch: {sa} vs {sb}")

    out_shape = tuple(out)
    if sa != out_shape:
        a = UOp("EXPAND", (a,), arg=out_shape)
    if sb != out_shape:
        b = UOp("EXPAND", (b,), arg=out_shape)
    return a, b


def add(a, b):
    return UOp("ADD", broadcast(a, b))


def sub(a, b):
    return UOp("SUB", broadcast(a, b))


def mul(a, b):
    return UOp("MUL", broadcast(a, b))


def rangeify(expr):
    if expr.op == "REDUCE":
        op, axis = expr.arg
        out_shape = expr.shape
        reduce_size = expr.src[0].shape[axis]

        names = ["i", "j", "k", "m"][:len(out_shape)]
        out_idxs = tuple(UOp("VAR", arg=name) for name in names)
        out_ranges = tuple(
            UOp("RANGE", (UOp("CONST", arg=size),), arg=name)
            for name, size in zip(names, out_shape)
        )
        k = UOp("VAR", arg="k")
        reduce_range = UOp(
            "RANGE",
            (UOp("CONST", arg=reduce_size),),
            arg="k",
        )
        in_idxs = out_idxs[:axis] + (k,) + out_idxs[axis:]

        element = lower_indexed(expr.src[0], in_idxs)
        body = UOp("REDUCE", (element,), arg=(op, axis))

        return out_ranges + (reduce_range,), body

    shape = expr.shape
    names = ["i", "j", "k", "l"][:len(shape)]
    idxs = tuple(UOp("VAR", arg=name) for name in names)
    ranges = tuple(
        UOp("RANGE", (UOp("CONST", arg=size),), arg=name)
        for name, size in zip(names, shape)
    )

    body = lower_indexed(expr, idxs)
    return ranges, body


def schedule(ranges, body):
    loop = ranges[0]
    size = loop.src[0].arg
    index = UOp("VAR", arg=loop.arg)
    out = UOp(
        "ALLOC",
        arg=ParamArg("out", body.dtype, size),
    )
    output_ptr = UOp("INDEX", (out, index))
    store = UOp("STORE", (output_ptr, body))
    end = UOp("END", (out, loop))
    return out, loop, store, end
