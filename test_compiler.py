import gc
import weakref
import pytest

from compiler import *


# ============================================================
# HELPERS
# ============================================================

def C(value):
    return UOp("CONST", arg=value)


def P(name, dtype="float", size=None):
    return UOp("PARAM", arg=ParamArg(name, dtype, size))


def B(name, dtype="float", size=10):
    return UOp("BUFFER", arg=ParamArg(name, dtype, size))


def V(name):
    return UOp("VAR", arg=name)


# ============================================================
# UOP CORE + HASH CONSING
# ============================================================

def test_uop_core():
    a, b = C(2), C(3)
    x = UOp("ADD", (a, b))

    assert x.op == "ADD"
    assert x.src == (a, b)
    assert isinstance(x.src, tuple)
    assert x.arg is None

    assert C(5) == C(5)
    assert C(5) is C(5)

    assert UOp("ADD", (a, b)) is UOp("ADD", (a, b))

    assert (
        P("x", "float", 3)
        is P("x", "float", 3)
    )

    assert (
        UOp("CALL", arg=CallArg("foo"))
        is UOp("CALL", arg=CallArg("foo"))
    )


def test_hash_cons_cache_releases_unused_nodes():
    UOp._cache.clear()

    a = C(999999)
    ref = weakref.ref(a)

    assert ref() is a

    del a
    gc.collect()

    assert ref() is None


def test_paramarg():
    x = ParamArg("x", "float", 3)
    y = ParamArg("x", "float", 3)

    assert (x.name, x.dtype, x.size) == ("x", "float", 3)
    assert x == y
    assert hash(x) == hash(y)


# ============================================================
# RUN / SIMPLIFY / RENDER — CORE OPS
# ============================================================

@pytest.mark.parametrize(
    "expr,run_expected,simplify_expected,render_expected",
    [
        (UOp("ADD", (C(2), C(3))), 5, C(5), "(2 + 3)"),
        (UOp("MUL", (C(4), C(5))), 20, C(20), "(4 * 5)"),
        (UOp("DIV", (C(8.0), C(2.0))), 4.0, C(4.0), "(8.0 / 2.0)"),
        (UOp("RECIP", (C(4.0),)), 0.25, C(0.25), "(1.0 / 4.0)"),
        (UOp("NEG", (C(5),)), -5, C(-5), "(-5)"),
        (UOp("MAX", (C(5), C(3))), 5, C(5), "max(5, 3)"),
        (UOp("CMPNE", (C(5), C(3))), True, C(True), "(5 != 3)"),
        (UOp("CMPLT", (C(3), C(5))), True, C(True), "(3 < 5)"),
        (UOp("FLOORDIV", (C(7), C(3))), 2, C(2), "(7 // 3)"),
        (UOp("FLOORMOD", (C(7), C(3))), 1, C(1), "(7 % 3)"),
        (UOp("EXP2", (C(3),)), 8, C(8), "exp2(3)"),
        (UOp("LOG2", (C(8),)), 3, C(3), "log2(8)"),
    ],
)
def test_core_ops(expr, run_expected, simplify_expected, render_expected):
    assert run(expr) == run_expected
    assert simplify(expr) == simplify_expected
    assert render(expr) == render_expected


@pytest.mark.parametrize(
    "value,dtype",
    [
        (5, "int"),
        (5.5, "float"),
        (True, "bool"),
    ],
)
def test_const_dtype(value, dtype):
    assert C(value).dtype == dtype


# ============================================================
# BOOLEAN OPS
# ============================================================

@pytest.mark.parametrize(
    "op,a,b,expected",
    [
        ("MUL", True, True, True),
        ("MUL", True, False, False),
        ("MUL", False, False, False),

        ("MAX", False, False, False),
        ("MAX", False, True, True),
        ("MAX", True, True, True),

        ("CMPNE", False, False, False),
        ("CMPNE", False, True, True),
        ("CMPNE", True, False, True),
        ("CMPNE", True, True, False),
    ],
)
def test_bool_ops(op, a, b, expected):
    assert run(UOp(op, (C(a), C(b)))) is expected


# ============================================================
# WHERE / CAST
# ============================================================

@pytest.mark.parametrize(
    "cond,expected",
    [
        (True, 10),
        (False, 20),
    ],
)
def test_where(cond, expected):
    expr = UOp("WHERE", (C(cond), C(10), C(20)))

    assert run(expr) == expected
    assert simplify(expr) == C(expected)


@pytest.mark.parametrize(
    "value,dtype,expected,render_expected",
    [
        (5.7, "int", 5, "((int)5.7)"),
        (5, "float", 5.0, "((float)5)"),
        (1, "bool", True, "((bool)1)"),
    ],
)
def test_cast(value, dtype, expected, render_expected):
    expr = UOp("CAST", (C(value),), arg=dtype)

    assert run(expr) == expected
    assert simplify(expr) == C(expected)
    assert render(expr) == render_expected


# ============================================================
# NESTED + SYMBOLIC SIMPLIFICATION
# ============================================================

@pytest.mark.parametrize(
    "expr,expected",
    [
        (UOp("ADD", (V("x"), C(0))), V("x")),
        (UOp("ADD", (C(0), V("x"))), V("x")),
        (UOp("MUL", (V("x"), C(1))), V("x")),
        (UOp("MUL", (C(1), V("x"))), V("x")),
        (UOp("MUL", (V("x"), C(0))), C(0)),
        (UOp("MUL", (C(0), V("x"))), C(0)),
        (UOp("DIV", (V("x"), C(1))), V("x")),
        (UOp("NEG", (UOp("NEG", (V("x"),)),)), V("x")),
    ],
)
def test_simplify_identities(expr, expected):
    assert simplify(expr) == expected


def test_nested_constant_folding():
    expr = UOp(
        "MUL",
        (
            UOp("ADD", (C(2), C(3))),
            C(4),
        ),
    )

    assert run(expr) == 20
    assert simplify(expr) == C(20)


def test_nonconstant_ops_survive_simplify():
    x, y = V("x"), V("y")

    assert simplify(UOp("DIV", (x, y))) == UOp("DIV", (x, y))
    assert simplify(UOp("RECIP", (x,))) == UOp("RECIP", (x,))


def test_recip_zero_raises():
    with pytest.raises(ZeroDivisionError):
        simplify(UOp("RECIP", (C(0),)))


# ============================================================
# RECURSIVE PROPERTIES
# ============================================================

@pytest.mark.parametrize(
    "expr,dtype,shape,addrspace",
    [
        (C(5), "int", (), "ALU"),
        (P("x", "float", 3), "float", (3,), "ALU"),
        (V("i"), "int", (), "ALU"),
        (B("a", "float", 10), "float", (10,), "MEM"),
        (
            UOp("ALLOC", arg=ParamArg("tmp", "float", 10)),
            "float",
            (10,),
            "MEM",
        ),
    ],
)
def test_recursive_properties(expr, dtype, shape, addrspace):
    assert expr.dtype == dtype
    assert expr.shape == shape
    assert expr.addrspace == addrspace


@pytest.mark.parametrize(
    "expr,expected",
    [
        (UOp("ADD", (C(2), C(3))), "int"),
        (UOp("ADD", (C(2), C(3.5))), "float"),
        (UOp("MUL", (C(True), C(4))), "int"),
        (UOp("CMPNE", (C(1), C(2))), "bool"),
        (UOp("CMPLT", (C(1), C(2))), "bool"),
    ],
)
def test_dtype_propagation(expr, expected):
    assert expr.dtype == expected


def test_elementwise_shape_propagation():
    a = P("a", "float", 3)
    b = P("b", "float", 3)

    assert UOp("ADD", (a, b)).shape == (3,)
    assert UOp("MUL", (a, b)).shape == (3,)


def test_elementwise_shape_mismatch():
    a = P("a", "float", 3)
    b = P("b", "float", 4)

    with pytest.raises(ValueError):
        _ = UOp("ADD", (a, b)).shape


# ============================================================
# MEMORY IR
# ============================================================

def test_memory_ir():
    buf = B("a", "float", 10)
    out = B("out", "float", 10)
    idx = C(3)

    ptr = UOp("INDEX", (buf, idx))
    load = UOp("LOAD", (ptr,))
    store = UOp(
        "STORE",
        (
            UOp("INDEX", (out, idx)),
            load,
        ),
    )

    assert ptr.dtype == "float"
    assert ptr.shape == ()

    assert load.dtype == "float"
    assert load.shape == ()

    assert store.dtype == "void"
    assert store.shape == ()

    assert render(ptr) == "a[3]"
    assert render(load) == "a[3]"
    assert render(store) == "out[3] = a[3];"


def test_alloc_render():
    x = UOp("ALLOC", arg=ParamArg("tmp", "float", 10))

    assert x.dtype == "float"
    assert x.shape == (10,)
    assert x.addrspace == "MEM"
    assert render(x) == "float tmp[10];"


def test_after():
    buf = B("a", "float", 3)

    store = UOp(
        "STORE",
        (
            UOp("INDEX", (buf, C(0))),
            C(5.0),
        ),
    )

    expr = UOp("AFTER", (buf, store))

    assert render(expr) == "a[0] = 5.0;\na"


# ============================================================
# STACK
# ============================================================

def test_stack():
    a = B("a", "float", 3)
    b = B("b", "float", 3)

    s = UOp("STACK", (a, b))

    assert s.shape == (2, 3)
    assert s.dtype == "float"

    bad = UOp(
        "STACK",
        (
            B("x", "float", 3),
            B("y", "float", 4),
        ),
    )

    with pytest.raises(ValueError):
        _ = bad.shape


# ============================================================
# ELEMENTWISE INTERPRETER
# ============================================================

@pytest.mark.parametrize(
    "op,a,b,expected",
    [
        ("ADD", [1, 2, 3], [10, 20, 30], [11, 22, 33]),
        ("MUL", [1, 2, 3], [4, 5, 6], [4, 10, 18]),
    ],
)
def test_elementwise_binary(op, a, b, expected):
    expr = UOp(
        op,
        (
            P("a", "int", len(a)),
            P("b", "int", len(b)),
        ),
    )

    assert run_elementwise(expr, {"a": a, "b": b}) == expected


def test_elementwise_nested():
    expr = UOp(
        "MUL",
        (
            UOp(
                "ADD",
                (
                    P("a", "int", 3),
                    P("b", "int", 3),
                ),
            ),
            C(2),
        ),
    )

    assert run_elementwise(
        expr,
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        },
    ) == [10, 14, 18]


# ============================================================
# RENDERER
# ============================================================

def test_render_memory_expression():
    a = B("a", "float", 10)
    out = B("out", "float", 10)
    i = V("i")

    load = UOp(
        "LOAD",
        (
            UOp("INDEX", (a, i)),
        ),
    )

    value = UOp("ADD", (load, C(2.0)))

    store = UOp(
        "STORE",
        (
            UOp("INDEX", (out, i)),
            value,
        ),
    )

    assert render(store) == "out[i] = (a[i] + 2.0);"


def test_render_symbolic_index_offset():
    a = B("a", "float", 10)
    i = V("i")

    idx = UOp("ADD", (i, C(1)))

    assert render(UOp("INDEX", (a, idx))) == "a[(i + 1)]"


# ============================================================
# CALL / PARAM
# ============================================================

def test_call():
    x = P("x", "int")
    y = P("y", "int")

    body = UOp(
        "MUL",
        (
            UOp("ADD", (x, y)),
            C(2),
        ),
    )

    call = UOp(
        "CALL",
        (
            body,
            C(5),
            C(7),
        ),
        arg=CallArg("foo"),
    )

    assert run(call) == 24


def test_render_call():
    expr = UOp(
        "CALL",
        (
            C(2),
            C(3),
        ),
        arg=CallArg("foo"),
    )

    assert render(expr) == "foo(2, 3)"


# ============================================================
# C RENDERING
# ============================================================

def test_render_c_function():
    a = B("a", "float", 10)
    out = B("out", "float", 10)
    i = P("i", "int", None)

    value = UOp(
        "ADD",
        (
            UOp(
                "LOAD",
                (
                    UOp("INDEX", (a, i)),
                ),
            ),
            C(2.0),
        ),
    )

    store = UOp(
        "STORE",
        (
            UOp("INDEX", (out, i)),
            value,
        ),
    )

    code = render_function(
        "kernel",
        store,
        params=(a, out, i),
    )

    assert code == (
        "void kernel(float *a, float *out, int i) {\n"
        "    out[i] = (a[i] + 2.0);\n"
        "}\n"
    )


def test_compile_c_indexes_params():
    a = P("a", "int", 3)
    b = P("b", "int", 3)

    expr = UOp("ADD", (a, b))

    code = compile_c(expr, 3)

    assert "out[i] = (a[i] + b[i]);" in code


# ============================================================
# RANGE / END — WEEK 3 START
# ============================================================

def test_range_and_end_render():
    r = UOp(
        "RANGE",
        (
            C(0),
            C(3),
        ),
        arg="i",
    )

    assert render(r) == "for (int i = 0; i < 3; i++)"
    assert render(UOp("END")) == "}"


def test_render_loop():
    r = UOp(
        "RANGE",
        (
            C(0),
            C(3),
        ),
        arg="i",
    )

    out = B("out", "float", 3)

    body = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    out,
                    V("i"),
                ),
            ),
            C(1.0),
        ),
    )

    code = "\n".join(
        [
            render(r),
            "    " + render(body),
            render(UOp("END")),
        ]
    )

    assert code == (
        "for (int i = 0; i < 3; i++)\n"
        "    out[i] = 1.0;\n"
        "}"
    )


# ============================================================
# MOVEMENT OPS
# ============================================================

def test_reshape():
    x = P("x", "float", 6)

    assert UOp("RESHAPE", (x,), arg=(2, 3)).shape == (2, 3)

    bad = UOp("RESHAPE", (x,), arg=(4, 2))

    with pytest.raises(ValueError):
        _ = bad.shape


@pytest.mark.parametrize(
    "base_shape,arg,expected",
    [
        ((10,), ((2, 6),), (4,)),
        ((5, 6), ((1, 4), (2, 5)), (3, 3)),
    ],
)
def test_shrink(base_shape, arg, expected):
    size = 1

    for x in base_shape:
        size *= x

    base = P("x", "float", size)

    if len(base_shape) > 1:
        base = UOp("RESHAPE", (base,), arg=base_shape)

    assert UOp("SHRINK", (base,), arg=arg).shape == expected


@pytest.mark.parametrize(
    "arg",
    [
        ((6, 2),),
        ((2, 12),),
    ],
)
def test_shrink_invalid(arg):
    x = P("x", "float", 10)

    with pytest.raises(ValueError):
        _ = UOp("SHRINK", (x,), arg=arg).shape


def test_shrink_wrong_axes():
    x = UOp(
        "RESHAPE",
        (
            P("x", "float", 30),
        ),
        arg=(5, 6),
    )

    with pytest.raises(ValueError):
        _ = UOp(
            "SHRINK",
            (x,),
            arg=((1, 4),),
        ).shape


@pytest.mark.parametrize(
    "shape,axes",
    [
        ((5,), (0,)),
        ((3, 4), (1,)),
    ],
)
def test_flip(shape, axes):
    size = 1

    for x in shape:
        size *= x

    base = P("x", "float", size)

    if len(shape) > 1:
        base = UOp("RESHAPE", (base,), arg=shape)

    flipped = UOp("FLIP", (base,), arg=axes)

    assert flipped.shape == shape
    assert flipped.dtype == "float"


@pytest.mark.parametrize(
    "shape,padding,expected",
    [
        ((5,), ((2, 3),), (10,)),
        ((3, 4), ((1, 2), (3, 1)), (6, 8)),
        ((2, 3, 4), ((1, 1), (2, 0), (0, 3)), (4, 5, 7)),
    ],
)
def test_pad(shape, padding, expected):
    size = 1

    for x in shape:
        size *= x

    base = P("x", "float", size)

    if len(shape) > 1:
        base = UOp("RESHAPE", (base,), arg=shape)

    padded = UOp("PAD", (base,), arg=padding)

    assert padded.shape == expected
    assert padded.dtype == "float"


# ============================================================
# TOPOSORT + LINEAR
# ============================================================

def test_toposort_and_linear():
    a = C(2)
    b = C(3)
    c = UOp("ADD", (a, b))
    d = UOp("MUL", (c, a))

    order = toposort(d)

    assert order.index(a) < order.index(c)
    assert order.index(b) < order.index(c)
    assert order.index(c) < order.index(d)

    # shared node only appears once
    assert order.count(a) == 1

    linear = linearize(d)

    assert linear.op == "LINEAR"
    assert linear.src[-1] is d
    assert linear.dtype == "void"
    assert linear.shape == ()


def test_linear_render_order():
    a = B("a", "float", 3)

    store1 = UOp(
        "STORE",
        (
            UOp("INDEX", (a, C(0))),
            C(5.0),
        ),
    )

    store2 = UOp(
        "STORE",
        (
            UOp("INDEX", (a, C(0))),
            C(7.0),
        ),
    )

    linear = UOp(
        "LINEAR",
        (
            store1,
            store2,
        ),
    )

    assert render(linear) == (
        "a[0] = 5.0;\n"
        "a[0] = 7.0;"
    )


def test_linearized_render():
    a = B("a", "float", 3)

    store = UOp(
        "STORE",
        (
            UOp("INDEX", (a, C(0))),
            C(5.0),
        ),
    )

    code = render(linearize(store))

    assert code.endswith("a[0] = 5.0;")



def test_range_basic():
    r = UOp("RANGE", (C(3),), arg="i")

    assert r.dtype == "int"
    assert r.shape == ()
    assert render(r) == "for (int i = 0; i < 3; i++)"


def test_end_basic():
    loop = UOp(
        "RANGE",
        (
            UOp("CONST", arg=3),
        ),
        arg="i",
    )

    value = UOp("CONST", arg=7)

    end = UOp(
        "END",
        (
            value,
            loop,
        ),
    )

    assert end.src == (value, loop)
    assert end.dtype == value.dtype
    assert end.shape == value.shape
    assert render(end) == "}"


def test_render_loop_with_end():
    loop = UOp(
        "RANGE",
        (
            UOp("CONST", arg=3),
        ),
        arg="i",
    )

    out = UOp(
        "BUFFER",
        arg=ParamArg("out", "float", 3),
    )

    store = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    out,
                    UOp("VAR", arg="i"),
                ),
            ),
            UOp("CONST", arg=1.0),
        ),
    )

    end = UOp(
        "END",
        (
            out,
            loop,
        ),
    )

    code = "\n".join([
        render(loop),
        "    " + render(store),
        render(end),
    ])

    assert code == (
        "for (int i = 0; i < 3; i++)\n"
        "    out[i] = 1.0;\n"
        "}"
    )


def test_nested_loops_render():
    outer = UOp(
        "RANGE",
        (UOp("CONST", arg=2),),
        arg="i",
    )

    inner = UOp(
        "RANGE",
        (UOp("CONST", arg=3),),
        arg="j",
    )

    end_inner = UOp(
        "END",
        (
            UOp("CONST", arg=0),
            inner,
        ),
    )

    end_outer = UOp(
        "END",
        (
            UOp("CONST", arg=0),
            outer,
        ),
    )

    code = "\n".join([
        render(outer),
        "    " + render(inner),
        "    " + render(end_inner),
        render(end_outer),
    ])

    assert code == (
        "for (int i = 0; i < 2; i++)\n"
        "    for (int j = 0; j < 3; j++)\n"
        "    }\n"
        "}"
    )


def test_render_reduction_update():
    a = UOp(
        "BUFFER",
        arg=ParamArg("a", "float", 3),
    )

    acc = UOp(
        "ALLOC",
        arg=ParamArg("acc", "float", 1),
    )

    k = UOp(
        "RANGE",
        (
            UOp("CONST", arg=3),
        ),
        arg="k",
    )

    update = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp(
                "ADD",
                (
                    UOp(
                        "LOAD",
                        (
                            UOp(
                                "INDEX",
                                (
                                    acc,
                                    UOp("CONST", arg=0),
                                ),
                            ),
                        ),
                    ),
                    UOp(
                        "LOAD",
                        (
                            UOp(
                                "INDEX",
                                (
                                    a,
                                    UOp("VAR", arg="k"),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    assert render(update) == "acc[0] = (acc[0] + a[k]);"



def test_render_full_reduction():
    a = UOp(
        "BUFFER",
        arg=ParamArg("a", "float", 3),
    )

    acc = UOp(
        "ALLOC",
        arg=ParamArg("acc", "float", 1),
    )

    init = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp("CONST", arg=0.0),
        ),
    )

    loop = UOp(
        "RANGE",
        (
            UOp("CONST", arg=3),
        ),
        arg="k",
    )

    update = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp(
                "ADD",
                (
                    UOp(
                        "LOAD",
                        (
                            UOp(
                                "INDEX",
                                (
                                    acc,
                                    UOp("CONST", arg=0),
                                ),
                            ),
                        ),
                    ),
                    UOp(
                        "LOAD",
                        (
                            UOp(
                                "INDEX",
                                (
                                    a,
                                    UOp("VAR", arg="k"),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    end = UOp(
        "END",
        (
            acc,
            loop,
        ),
    )

    code = "\n".join([
        render(acc),
        render(init),
        render(loop),
        "    " + render(update),
        render(end),
    ])

    assert code == (
        "float acc[1];\n"
        "acc[0] = 0.0;\n"
        "for (int k = 0; k < 3; k++)\n"
        "    acc[0] = (acc[0] + a[k]);\n"
        "}"
    )


def test_render_gemm_one_cell():
    A = UOp(
        "BUFFER",
        arg=ParamArg("A", "float", 4),
    )

    B = UOp(
        "BUFFER",
        arg=ParamArg("B", "float", 4),
    )

    acc = UOp(
        "ALLOC",
        arg=ParamArg("acc", "float", 1),
    )

    init = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp("CONST", arg=0.0),
        ),
    )

    k = UOp(
        "RANGE",
        (
            UOp("CONST", arg=2),
        ),
        arg="k",
    )

    a_index = UOp(
        "ADD",
        (
            UOp("CONST", arg=0),
            UOp("VAR", arg="k"),
        ),
    )

    b_index = UOp(
        "MUL",
        (
            UOp("VAR", arg="k"),
            UOp("CONST", arg=2),
        ),
    )

    update = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp(
                "ADD",
                (
                    UOp(
                        "LOAD",
                        (
                            UOp(
                                "INDEX",
                                (
                                    acc,
                                    UOp("CONST", arg=0),
                                ),
                            ),
                        ),
                    ),
                    UOp(
                        "MUL",
                        (
                            UOp(
                                "LOAD",
                                (
                                    UOp(
                                        "INDEX",
                                        (
                                            A,
                                            a_index,
                                        ),
                                    ),
                                ),
                            ),
                            UOp(
                                "LOAD",
                                (
                                    UOp(
                                        "INDEX",
                                        (
                                            B,
                                            b_index,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    end = UOp(
        "END",
        (
            acc,
            k,
        ),
    )

    code = "\n".join([
        render(acc),
        render(init),
        render(k),
        "    " + render(update),
        render(end),
    ])

    assert code == (
        "float acc[1];\n"
        "acc[0] = 0.0;\n"
        "for (int k = 0; k < 2; k++)\n"
        "    acc[0] = (acc[0] + (A[(0 + k)] * B[(k * 2)]));\n"
        "}"
    )


def test_render_gemm_2x2():
    A = UOp(
        "BUFFER",
        arg=ParamArg("A", "float", 4),
    )

    B = UOp(
        "BUFFER",
        arg=ParamArg("B", "float", 4),
    )

    C = UOp(
        "BUFFER",
        arg=ParamArg("C", "float", 4),
    )

    acc = UOp(
        "ALLOC",
        arg=ParamArg("acc", "float", 1),
    )

    i = UOp(
        "RANGE",
        (UOp("CONST", arg=2),),
        arg="i",
    )

    j = UOp(
        "RANGE",
        (UOp("CONST", arg=2),),
        arg="j",
    )

    k = UOp(
        "RANGE",
        (UOp("CONST", arg=2),),
        arg="k",
    )

    init = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp("CONST", arg=0.0),
        ),
    )

    a_index = UOp(
        "ADD",
        (
            UOp(
                "MUL",
                (
                    UOp("VAR", arg="i"),
                    UOp("CONST", arg=2),
                ),
            ),
            UOp("VAR", arg="k"),
        ),
    )

    b_index = UOp(
        "ADD",
        (
            UOp(
                "MUL",
                (
                    UOp("VAR", arg="k"),
                    UOp("CONST", arg=2),
                ),
            ),
            UOp("VAR", arg="j"),
        ),
    )

    update = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp(
                "ADD",
                (
                    UOp(
                        "LOAD",
                        (
                            UOp(
                                "INDEX",
                                (
                                    acc,
                                    UOp("CONST", arg=0),
                                ),
                            ),
                        ),
                    ),
                    UOp(
                        "MUL",
                        (
                            UOp(
                                "LOAD",
                                (
                                    UOp(
                                        "INDEX",
                                        (
                                            A,
                                            a_index,
                                        ),
                                    ),
                                ),
                            ),
                            UOp(
                                "LOAD",
                                (
                                    UOp(
                                        "INDEX",
                                        (
                                            B,
                                            b_index,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    c_index = UOp(
        "ADD",
        (
            UOp(
                "MUL",
                (
                    UOp("VAR", arg="i"),
                    UOp("CONST", arg=2),
                ),
            ),
            UOp("VAR", arg="j"),
        ),
    )

    store_c = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    C,
                    c_index,
                ),
            ),
            UOp(
                "LOAD",
                (
                    UOp(
                        "INDEX",
                        (
                            acc,
                            UOp("CONST", arg=0),
                        ),
                    ),
                ),
            ),
        ),
    )

    end_k = UOp("END", (acc, k))
    end_j = UOp("END", (C, j))
    end_i = UOp("END", (C, i))

    code = "\n".join([
        render(acc),
        render(i),
        "    " + render(j),
        "        " + render(init),
        "        " + render(k),
        "            " + render(update),
        "        " + render(end_k),
        "        " + render(store_c),
        "    " + render(end_j),
        render(end_i),
    ])

    assert code == (
        "float acc[1];\n"
        "for (int i = 0; i < 2; i++)\n"
        "    for (int j = 0; j < 2; j++)\n"
        "        acc[0] = 0.0;\n"
        "        for (int k = 0; k < 2; k++)\n"
        "            acc[0] = (acc[0] + (A[((i * 2) + k)] * B[((k * 2) + j)]));\n"
        "        }\n"
        "        C[((i * 2) + j)] = acc[0];\n"
        "    }\n"
        "}"
    )


def test_render_conv1d():
    x = UOp(
        "BUFFER",
        arg=ParamArg("x", "float", 4),
    )

    w = UOp(
        "BUFFER",
        arg=ParamArg("w", "float", 3),
    )

    out = UOp(
        "BUFFER",
        arg=ParamArg("out", "float", 2),
    )

    acc = UOp(
        "ALLOC",
        arg=ParamArg("acc", "float", 1),
    )

    i = UOp(
        "RANGE",
        (UOp("CONST", arg=2),),
        arg="i",
    )

    k = UOp(
        "RANGE",
        (UOp("CONST", arg=3),),
        arg="k",
    )

    init = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp("CONST", arg=0.0),
        ),
    )

    x_index = UOp(
        "ADD",
        (
            UOp("VAR", arg="i"),
            UOp("VAR", arg="k"),
        ),
    )

    update = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp(
                "ADD",
                (
                    UOp(
                        "LOAD",
                        (
                            UOp(
                                "INDEX",
                                (
                                    acc,
                                    UOp("CONST", arg=0),
                                ),
                            ),
                        ),
                    ),
                    UOp(
                        "MUL",
                        (
                            UOp(
                                "LOAD",
                                (
                                    UOp(
                                        "INDEX",
                                        (
                                            x,
                                            x_index,
                                        ),
                                    ),
                                ),
                            ),
                            UOp(
                                "LOAD",
                                (
                                    UOp(
                                        "INDEX",
                                        (
                                            w,
                                            UOp("VAR", arg="k"),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    store_out = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    out,
                    UOp("VAR", arg="i"),
                ),
            ),
            UOp(
                "LOAD",
                (
                    UOp(
                        "INDEX",
                        (
                            acc,
                            UOp("CONST", arg=0),
                        ),
                    ),
                ),
            ),
        ),
    )

    end_k = UOp("END", (acc, k))
    end_i = UOp("END", (out, i))

    code = "\n".join([
        render(acc),
        render(i),
        "    " + render(init),
        "    " + render(k),
        "        " + render(update),
        "    " + render(end_k),
        "    " + render(store_out),
        render(end_i),
    ])

    assert code == (
        "float acc[1];\n"
        "for (int i = 0; i < 2; i++)\n"
        "    acc[0] = 0.0;\n"
        "    for (int k = 0; k < 3; k++)\n"
        "        acc[0] = (acc[0] + (x[(i + k)] * w[k]));\n"
        "    }\n"
        "    out[i] = acc[0];\n"
        "}"
    )


def test_render_conv2d():
    inp = UOp(
        "BUFFER",
        arg=ParamArg("inp", "float", 9),   # 3x3
    )

    ker = UOp(
        "BUFFER",
        arg=ParamArg("ker", "float", 4),   # 2x2
    )

    out = UOp(
        "BUFFER",
        arg=ParamArg("out", "float", 4),   # 2x2 output
    )

    acc = UOp(
        "ALLOC",
        arg=ParamArg("acc", "float", 1),
    )

    y = UOp("RANGE", (UOp("CONST", arg=2),), arg="y")
    x = UOp("RANGE", (UOp("CONST", arg=2),), arg="x")
    ky = UOp("RANGE", (UOp("CONST", arg=2),), arg="ky")
    kx = UOp("RANGE", (UOp("CONST", arg=2),), arg="kx")

    init = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp("CONST", arg=0.0),
        ),
    )

    input_index = UOp(
        "ADD",
        (
            UOp(
                "MUL",
                (
                    UOp(
                        "ADD",
                        (
                            UOp("VAR", arg="y"),
                            UOp("VAR", arg="ky"),
                        ),
                    ),
                    UOp("CONST", arg=3),
                ),
            ),
            UOp(
                "ADD",
                (
                    UOp("VAR", arg="x"),
                    UOp("VAR", arg="kx"),
                ),
            ),
        ),
    )

    kernel_index = UOp(
        "ADD",
        (
            UOp(
                "MUL",
                (
                    UOp("VAR", arg="ky"),
                    UOp("CONST", arg=2),
                ),
            ),
            UOp("VAR", arg="kx"),
        ),
    )

    update = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    acc,
                    UOp("CONST", arg=0),
                ),
            ),
            UOp(
                "ADD",
                (
                    UOp(
                        "LOAD",
                        (
                            UOp(
                                "INDEX",
                                (
                                    acc,
                                    UOp("CONST", arg=0),
                                ),
                            ),
                        ),
                    ),
                    UOp(
                        "MUL",
                        (
                            UOp(
                                "LOAD",
                                (
                                    UOp(
                                        "INDEX",
                                        (
                                            inp,
                                            input_index,
                                        ),
                                    ),
                                ),
                            ),
                            UOp(
                                "LOAD",
                                (
                                    UOp(
                                        "INDEX",
                                        (
                                            ker,
                                            kernel_index,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    output_index = UOp(
        "ADD",
        (
            UOp(
                "MUL",
                (
                    UOp("VAR", arg="y"),
                    UOp("CONST", arg=2),
                ),
            ),
            UOp("VAR", arg="x"),
        ),
    )

    store_out = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    out,
                    output_index,
                ),
            ),
            UOp(
                "LOAD",
                (
                    UOp(
                        "INDEX",
                        (
                            acc,
                            UOp("CONST", arg=0),
                        ),
                    ),
                ),
            ),
        ),
    )

    end_kx = UOp("END", (acc, kx))
    end_ky = UOp("END", (acc, ky))
    end_x = UOp("END", (out, x))
    end_y = UOp("END", (out, y))

    code = "\n".join([
        render(acc),
        render(y),
        "    " + render(x),
        "        " + render(init),
        "        " + render(ky),
        "            " + render(kx),
        "                " + render(update),
        "            " + render(end_kx),
        "        " + render(end_ky),
        "        " + render(store_out),
        "    " + render(end_x),
        render(end_y),
    ])

    assert code == (
        "float acc[1];\n"
        "for (int y = 0; y < 2; y++)\n"
        "    for (int x = 0; x < 2; x++)\n"
        "        acc[0] = 0.0;\n"
        "        for (int ky = 0; ky < 2; ky++)\n"
        "            for (int kx = 0; kx < 2; kx++)\n"
        "                acc[0] = (acc[0] + (inp[(((y + ky) * 3) + (x + kx))] * ker[((ky * 2) + kx)]));\n"
        "            }\n"
        "        }\n"
        "        out[((y * 2) + x)] = acc[0];\n"
        "    }\n"
        "}"
    )


def _assert_pad_lowered(arg, expected):
    a = P("a", "int", 5)
    padded = UOp("PAD", (a,), arg=arg)
    assert render(lower(padded)) == expected


def test_lower_pad():
    # formula: (i < left) ? 0 : (i < left + size) ? a[i - left] : 0
    _assert_pad_lowered(((2, 0),), "((i < 2) ? 0 : a[(i + -2)])")
    _assert_pad_lowered(((2, 2),), "((i < 2) ? 0 : ((i < (2 + 5)) ? a[(i + -2)] : 0))")
    _assert_pad_lowered(((0, 2),), "((i < (0 + 5)) ? a[i] : 0)")


def test_lower_shrink():
    a = P("a", "int", 10)

    shrunk = UOp("SHRINK", (a,), arg=((2, 7),))

    assert render(lower(shrunk)) == "a[(i + 2)]"

    a = P("a", "int", 10)
    shrunk = UOp("SHRINK", (a,), arg=((0, 10),))
    assert render(lower(shrunk)) == "a[i]"

    a = P("a", "int", 10)
    shrunk = UOp("SHRINK", (a,), arg=((4, 9),))
    assert render(lower(shrunk)) == "a[(i + 4)]"

    a = P("a", "int", 10)
    b = P("b", "int", 10)
    expr = UOp("ADD", (
        UOp("SHRINK", (a,), arg=((2, 7),)),
        UOp("SHRINK", (b,), arg=((1, 6),)),
    ))
    assert render(lower(expr)) == "(a[(i + 2)] + b[(i + 1)])"


def test_lower_flip():
    a = P("a", "int", 5)

    flipped = UOp("FLIP", (a,))

    assert render(lower(flipped)) == "a[(4 - i)]"


       # the 2 must come from the input's shape, not be hardcoded
    a = P("a", "int", 3)
    flipped = UOp("FLIP", (a,))
    assert render(lower(flipped)) == "a[(2 - i)]"

    a = P("a", "int", 5)
    b = P("b", "int", 10)
    expr = UOp("ADD", (
        UOp("FLIP", (a,)),
        UOp("SHRINK", (b,), arg=((1, 6),)),
    ))
    assert render(lower(expr)) == "(a[(4 - i)] + b[(i + 1)])"


def test_lower_expand():
    a = P("a", "int", 1)                          

    expanded = UOp("EXPAND", (a,), arg=(10,))     

    assert render(lower(expanded)) == "a[0]"


    a = P("a", "int", 1)
    expanded = UOp("EXPAND", (a,), arg=(4,))
    assert render(lower(expanded)) == "a[0]"

def test_lower_program_elementwise():
    a = P("a", "int", 5)
    b = P("b", "int", 5)

    expr = UOp("ADD", (a, b))

    prog = lower_program(expr)

    assert render(prog) == (
        "int out[5];\n"
        "for (int i = 0; i < 5; i++)\n"
        "out[i] = (a[i] + b[i]);\n"
        "}"
    )




def test_lower_reduce_sum():
    a = P("a", "float", 5)
    red = UOp("REDUCE", (a,), arg=("ADD" , 0))
    lowered = lower(red)

    assert render(lowered) == (
        "float acc[1];\n"
        "acc[0] = 0.0;\n"
        "for (int k = 0; k < 5; k++)\n"
        "acc[0] = (acc[0] + a[k]);\n"
        "}"
    )

    a = P("a", "float", 5)

    red = UOp("REDUCE", (a,), arg=("MAX", 0))

    lowered = lower(red)

    assert render(lowered) == (
        "float acc[1];\n"
        "acc[0] = a[0];\n"                    
        "for (int k = 0; k < 5; k++)\n"
        "acc[0] = max(acc[0], a[k]);\n"       
        "}"
    )


def test_reduce_composes():
    a = P("a", "int", 10)

    shrunk = UOp("SHRINK", (a,), arg=((2, 7),))     
    red = UOp("REDUCE", (shrunk,), arg=("ADD", 0))  

    lowered = lower(red)

    assert render(lowered) == (
        "int acc[1];\n"
        "acc[0] = 0;\n"
        "for (int k = 0; k < 5; k++)\n"              
        "acc[0] = (acc[0] + a[(k + 2)]);\n"          
        "}"
    )

def test_lower_permute():
    a = UOp("RESHAPE", (P("a", "int", 6),), arg=(2, 3))
    permuted = UOp("PERMUTE", (a, ), arg=(1,0))
    assert permuted.shape == (3,2)

    i, j = V("i"), V("j")
    lowered = lower_indexed(permuted, (i, j ))
    assert render(lowered) == "a[((j * 3) + i)]"

def test_lower_expand_identity():
    a = P("a", "int", 3)
    expanded = UOp("EXPAND", (a, ), arg=(3,))
    assert render(lower(expanded)) == "a[i]"


def test_broadcast_shape_propagation():
    a = P("a", "int", 5)
    b = P("b", "int", 1)

    expr = UOp("ADD", (a, b))
    assert expr.shape == (5,)

    expr2 = UOp("MUL", (b, a))          
    assert expr2.shape == (5,)



def test_lower_stack():
    a = P("a","int", 3  )
    b = P("b", "int", 3)
    stacked = UOp("STACK", (a,b))
    assert render(lower(stacked)) == "((i < 3) ? a[(i % 3)] : b[(i % 3)])"



def eval_lowered_at(expr, arrays, i):
    if expr.op == "VAR" and expr.arg == "i":
        return i
    if expr.op == "INDEX":
        base = expr.src[0]
        idx = eval_lowered_at(expr.src[1], arrays, i)
        return arrays[base.arg.name][idx]
    if expr.op == "LOAD":
        return eval_lowered_at(expr.src[0], arrays, i)
    if expr.op == "WHERE":                     
        cond = eval_lowered_at(expr.src[0], arrays, i)
        taken = expr.src[1] if cond else expr.src[2]
        return eval_lowered_at(taken, arrays, i)
    const = lambda v: UOp("CONST", arg=v)
    rebuilt = UOp(expr.op, tuple(const(eval_lowered_at(s, arrays, i)) for s in expr.src), arg=expr.arg)
    return run(rebuilt)
def test_end_to_end_1d():
    a_vals = [1, 2, 3, 4, 5]
    b_vals = [10, 10, 10, 10, 10]

    a = P("a", "int", 5)
    b = P("b", "int", 5)

    expr = UOp("ADD", (
        UOp("PAD", (UOp("FLIP", (a,)),), arg=((1, 0),)),
        b,
    ))

    n = 5

    expected = run_elementwise(expr, {"a": a_vals, "b": b_vals})
    assert expected == [10, 15, 14, 13, 12]   

    lowered = lower(expr)
    actual = [eval_lowered_at(lowered, {"a": a_vals, "b": b_vals}, i) for i in range(n)]
    assert actual == expected


def test_lower_reshape_1d():
    a = P("a", "int", 5)
    reshaped = UOp("RESHAPE", (a,), arg=(5,))       
    assert render(lower(reshaped)) == "a[i]" 


def test_lower_reshape_2d():
    a = P("a", "int", 6)
    reshaped = UOp("RESHAPE", (a,), arg=(2,3))
    i,j = V("i"), V("j")
    lowered = lower_indexed(reshaped, (i,j))
    assert render(lowered) == "a[((i * 3) + j)]"

def test_lower_reshape_3d():
    a = P("a", "int", 24)
    reshaped = UOp("RESHAPE", (a,), arg=(2, 3, 4))

    i, j, k = V("i"), V("j"), V("k")
    lowered = lower_indexed(reshaped, (i, j, k))
    # slab i, row j, col k -> flat i*12 + j*4 + k
    assert render(lowered) == "a[((i * 12) + ((j * 4) + k))]"

def test_lower_reshape_4d():
    a = P("a", "int", 120)
    reshaped = UOp("RESHAPE", (a,), arg=(2, 3, 4, 5))

    i, j, k, l = V("i"), V("j"), V("k"), V("l")
    lowered = lower_indexed(reshaped, (i, j, k, l))
    # strides: 60, 20, 5, 1 -> flat i*60 + j*20 + k*5 + l
    assert render(lowered) == "a[((i * 60) + ((j * 20) + ((k * 5) + l)))]"


def test_lower_permute_1d():
    a = UOp("RESHAPE", (P("a", "int", 5),), arg=(5,))
    permuted = UOp("PERMUTE", (a,), arg=(0,))
    assert permuted.shape == (5,)

    assert render(lower(permuted)) == "a[i]"        


def test_lower_permute_2d():
    a = UOp("RESHAPE", (P("a", "int", 6),), arg=(2, 3))
    permuted = UOp("PERMUTE", (a,), arg=(1, 0))
    assert permuted.shape == (3, 2)

    i, j = V("i"), V("j")
    # out (i, j) -> in (j, i) -> flat j*3 + i
    assert render(lower_indexed(permuted, (i, j))) == "a[((j * 3) + i)]"


def test_lower_permute_3d():
    a = UOp("RESHAPE", (P("a", "int", 24),), arg=(2, 3, 4))
    permuted = UOp("PERMUTE", (a,), arg=(2, 0, 1))
    # out axis k <- in axis 2, out axis i <- in axis 0, out axis j <- in axis 1
    assert permuted.shape == (4, 2, 3)

    i, j, k = V("i"), V("j"), V("k")
    # out (i, j, k) -> in (j, k, i) -> strides of (2,3,4) are (12, 4, 1)
    # flat = j*12 + k*4 + i
    assert render(lower_indexed(permuted, (i, j, k))) == "a[((j * 12) + ((k * 4) + i))]"


def test_lower_permute_4d():
    a = UOp("RESHAPE", (P("a", "int", 120),), arg=(2, 3, 4, 5))
    permuted = UOp("PERMUTE", (a,), arg=(3, 1, 0, 2))
    assert permuted.shape == (5, 3, 2, 4)

    i, j, k, l = V("i"), V("j"), V("k"), V("l")
    # out (i, j, k, l) -> in (k, j, l, i) -> strides (60, 20, 5, 1)
    # flat = k*60 + j*20 + l*5 + i
    assert render(lower_indexed(permuted, (i, j, k, l))) == "a[((k * 60) + ((j * 20) + ((l * 5) + i)))]"


def test_lower_flip_2d_axis0():
    a = UOp("RESHAPE", (P("a", "int", 12),), arg=(3, 4))
    flipped = UOp("FLIP", (a,), arg=(0,))          
    assert flipped.shape == (3, 4)

    i, j = V("i"), V("j")
    # in tuple = ((2 - i), j) -> flat (2 - i)*4 + j
    assert render(lower_indexed(flipped, (i, j))) == "a[(((2 - i) * 4) + j)]"


def test_lower_flip_3d():
    a = UOp("RESHAPE", (P("a", "int", 24),), arg=(2, 3, 4))
    flipped = UOp("FLIP", (a,), arg=(0, 2))        
    assert flipped.shape == (2, 3, 4)

    i, j, k = V("i"), V("j"), V("k")
    # in tuple = ((1 - i), j, (3 - k)) -> flat (1-i)*12 + j*4 + (3-k)
    assert render(lower_indexed(flipped, (i, j, k))) == "a[(((1 - i) * 12) + ((j * 4) + (3 - k)))]"


def test_lower_flip_4d():
    a = UOp("RESHAPE", (P("a", "int", 120),), arg=(2, 3, 4, 5))
    flipped = UOp("FLIP", (a,), arg=(1, 3))      
    assert flipped.shape == (2, 3, 4, 5)

    i, j, k, l = V("i"), V("j"), V("k"), V("l")
    # in tuple = (i, (2 - j), k, (4 - l)) -> flat i*60 + (2-j)*20 + k*5 + (4-l)
    assert render(lower_indexed(flipped, (i, j, k, l))) == "a[((i * 60) + (((2 - j) * 20) + ((k * 5) + (4 - l))))]"


def test_lower_shrink_2d_axis0():
    a = UOp("RESHAPE", (P("a", "int", 12),), arg=(3, 4))
    shrunk = UOp("SHRINK", (a,), arg=((1, 3), (0, 4)))   
    assert shrunk.shape == (2, 4)

    i, j = V("i"), V("j")
    # in tuple = (i + 1, j) -> flat (i + 1)*4 + j
    assert render(lower_indexed(shrunk, (i, j))) == "a[(((i + 1) * 4) + j)]"


def test_lower_shrink_2d_axis1():
    a = UOp("RESHAPE", (P("a", "int", 12),), arg=(3, 4))
    shrunk = UOp("SHRINK", (a,), arg=((0, 3), (2, 4)))  
    assert shrunk.shape == (3, 2)

    i, j = V("i"), V("j")
    # in tuple = (i, j + 2) -> flat i*4 + (j + 2)
    assert render(lower_indexed(shrunk, (i, j))) == "a[((i * 4) + (j + 2))]"


def test_lower_shrink_3d():
    a = UOp("RESHAPE", (P("a", "int", 24),), arg=(2, 3, 4))
    shrunk = UOp("SHRINK", (a,), arg=((1, 2), (0, 3), (2, 4)))
    # out shape: (1, 3, 2)
    assert shrunk.shape == (1, 3, 2)

    i, j, k = V("i"), V("j"), V("k")
    # in tuple = (i + 1, j, k + 2) -> flat (i+1)*12 + j*4 + (k+2)
    assert render(lower_indexed(shrunk, (i, j, k))) == "a[(((i + 1) * 12) + ((j * 4) + (k + 2)))]"


def test_lower_shrink_4d():
    a = UOp("RESHAPE", (P("a", "int", 120),), arg=(2, 3, 4, 5))
    shrunk = UOp("SHRINK", (a,), arg=((0, 2), (1, 3), (0, 4), (3, 5)))
    # out shape: (2, 2, 4, 2)
    assert shrunk.shape == (2, 2, 4, 2)

    i, j, k, l = V("i"), V("j"), V("k"), V("l")
    # in tuple = (i, j + 1, k, l + 3) -> flat i*60 + (j+1)*20 + k*5 + (l+3)
    assert render(lower_indexed(shrunk, (i, j, k, l))) == "a[((i * 60) + (((j + 1) * 20) + ((k * 5) + (l + 3))))]"



def test_lower_pad_2d_axis0():
    a = UOp("RESHAPE", (P("a", "int", 12),), arg=(3, 4))
    padded = UOp("PAD", (a,), arg=((2, 0), (0, 0)))     
    assert padded.shape == (5, 4)

    i, j = V("i"), V("j")
    assert render(lower_indexed(padded, (i, j))) == (
        "((i < 2) ? 0 : a[(((i + -2) * 4) + j)])"
    )

def test_lower_pad_3d():
    a = UOp("RESHAPE", (P("a", "int", 24),), arg=(2, 3, 4))
    padded = UOp("PAD", (a,), arg=((1, 1), (0, 0), (0, 2)))
    # out shape: (4, 3, 6)
    assert padded.shape == (4, 3, 6)

    i, j, k = V("i"), V("j"), V("k")
    assert render(lower_indexed(padded, (i, j, k))) == (
        "((i < 1) ? 0 : ((i < (1 + 2)) ? ((k < (0 + 4)) ? a[(((i + -1) * 12) + ((j * 4) + k))] : 0) : 0))"
    )


def test_lower_pad_4d():
    a = UOp("RESHAPE", (P("a", "int", 24),), arg=(2, 3, 4))
    padded = UOp("PAD", (a,), arg=((0, 0), (2, 1), (0, 0)))
    # out shape: (2, 6, 4)
    assert padded.shape == (2, 6, 4)

    i, j, k = V("i"), V("j"), V("k")
    assert render(lower_indexed(padded, (i, j, k))) == (
        "((j < 2) ? 0 : ((j < (2 + 3)) ? a[((i * 12) + (((j + -2) * 4) + k))] : 0))"
    )



def test_lower_expand_2d():
    a = UOp("RESHAPE", (P("a", "int", 3),), arg=(3, 1))
    expanded = UOp("EXPAND", (a,), arg=(3,4))
    assert expanded.shape == (3,4)
    i, j = V("i"), V("j")
    assert render(lower_indexed(expanded, (i, j))) == "a[((i * 1) + 0)]"


def test_lower_expand_3d():
    a = UOp("RESHAPE", (P("a", "int", 6),), arg=(2, 3, 1))
    expanded = UOp("EXPAND", (a,), arg=(2, 3, 4))     
    assert expanded.shape == (2, 3, 4)

    i, j, k = V("i"), V("j"), V("k")
    # in tuple = (i, j, 0) -> strides of (2,3,1) are (3, 1, 1)
    # flat = i*3 + j*1 + 0
    assert render(lower_indexed(expanded, (i, j, k))) == "a[((i * 3) + ((j * 1) + 0))]"


def test_lower_expand_3d_middle():
    a = UOp("RESHAPE", (P("a", "int", 8),), arg=(2, 1, 4))
    expanded = UOp("EXPAND", (a,), arg=(2, 3, 4))     # broadcast the MIDDLE axis
    assert expanded.shape == (2, 3, 4)

    i, j, k = V("i"), V("j"), V("k")
    # in tuple = (i, 0, k) -> flat i*12 + 0*4 + k
    assert render(lower_indexed(expanded, (i, j, k))) == "a[((i * 4) + ((0 * 4) + k))]"


def test_lower_expand_4d():
    a = UOp("RESHAPE", (P("a", "int", 8),), arg=(2, 1, 4, 1))
    expanded = UOp("EXPAND", (a,), arg=(2, 3, 4, 5))  # two broadcast axes: 1 and 3
    assert expanded.shape == (2, 3, 4, 5)

    i, j, k, l = V("i"), V("j"), V("k"), V("l")
    # in tuple = (i, 0, k, 0) -> strides (60, 20, 5, 1)
    # flat = i*60 + 0*20 + k*5 + 0
    assert render(lower_indexed(expanded, (i, j, k, l))) == "a[((i * 4) + ((0 * 4) + ((k * 1) + 0)))]"


def test_broadcast_lowered():
    a = P("a", "int", 5)
    b = P("b", "int", 1)
    expr = UOp("ADD", broadcast(a, b))
    assert render(lower(expr)) == "(a[i] + b[0])"


def test_broadcast_2d():
    a = UOp("RESHAPE", (P("a", "int", 2),), arg=(2, 1))
    b = UOp("RESHAPE", (P("b", "int", 3),), arg=(1, 3))
    assert UOp("ADD", broadcast(a, b)).shape == (2, 3)
    assert UOp("ADD", (a, b)).shape == (2, 3)


def test_broadcast_mismatch_2d():
    a = UOp("RESHAPE", (P("a", "int", 6),), arg=(2, 3))
    b = UOp("RESHAPE", (P("b", "int", 4),), arg=(2, 2))
    with pytest.raises(ValueError):
        _ = UOp("ADD", (a, b)).shape



def test_lower_program_2d():
    a = UOp("RESHAPE", (P("a", "int", 6),), arg=(2, 3))
    b = UOp("RESHAPE", (P("b", "int", 6),), arg=(2, 3))
    expr = add(a, b)

    prog = lower_program(expr)

    assert render(prog) == (
        "int out[6];\n"
        "for (int i = 0; i < 2; i++)\n"
        "for (int j = 0; j < 3; j++)\n"
        "out[((i * 3) + j)] = (a[((i * 3) + j)] + b[((i * 3) + j)]);\n"
        "}\n"
        "}"
    )

def test_lower_program_2d():
    a = UOp("RESHAPE", (P("a", "int", 6),), arg=(2, 3))
    b = UOp("RESHAPE", (P("b", "int", 6),), arg=(2, 3))
    expr = add(a, b)

    prog = lower_program(expr)

    assert render(prog) == (
        "int out[6];\n"
        "for (int i = 0; i < 2; i++)\n"
        "for (int j = 0; j < 3; j++)\n"
        "out[((i * 3) + j)] = (a[((i * 3) + j)] + b[((i * 3) + j)]);\n"
        "}\n"
        "}"
    )


def test_lower_program_3d():
    a = UOp("RESHAPE", (P("a", "int", 24),), arg=(2, 3, 4))
    b = UOp("RESHAPE", (P("b", "int", 24),), arg=(2, 3, 4))
    expr = add(a, b)

    prog = lower_program(expr)

    assert render(prog) == (
        "int out[24];\n"
        "for (int i = 0; i < 2; i++)\n"
        "for (int j = 0; j < 3; j++)\n"
        "for (int k = 0; k < 4; k++)\n"
        "out[((i * 12) + ((j * 4) + k))] = (a[((i * 12) + ((j * 4) + k))] + b[((i * 12) + ((j * 4) + k))]);\n"
        "}\n"
        "}\n"
        "}"
    )


def test_lower_program_4d():
    a = UOp("RESHAPE", (P("a", "int", 120),), arg=(2, 3, 4, 5))
    b = UOp("RESHAPE", (P("b", "int", 120),), arg=(2, 3, 4, 5))
    expr = add(a, b)

    prog = lower_program(expr)

    assert render(prog) == (
        "int out[120];\n"
        "for (int i = 0; i < 2; i++)\n"
        "for (int j = 0; j < 3; j++)\n"
        "for (int k = 0; k < 4; k++)\n"
        "for (int l = 0; l < 5; l++)\n"
        "out[((i * 60) + ((j * 20) + ((k * 5) + l)))] = (a[((i * 60) + ((j * 20) + ((k * 5) + l)))] + b[((i * 60) + ((j * 20) + ((k * 5) + l)))]);\n"
        "}\n"
        "}\n"
        "}\n"
        "}"
    )


def test_matmul_2x2_lowered():
    A = UOp("RESHAPE", (P("A", "float", 4),), arg=(2, 2))
    B = UOp("RESHAPE", (P("B", "float", 4),), arg=(2, 2))

    C = matmul(A, B)
    assert C.shape == (2, 2)

    prog = lower_program(C)

    # semantic match with the hand-written test_render_gemm_2x2:
    # acc[0] = (acc[0] + (A[i*2+k] * B[k*2+j])) inside i/j loops
    assert render(prog) == (
        "float out[4];\n"
        "for (int i = 0; i < 2; i++)\n"
        "for (int j = 0; j < 2; j++)\n"
        "float acc[1];\n"
        "acc[0] = 0.0;\n"
        "for (int k = 0; k < 2; k++)\n"
        "acc[0] = (acc[0] + (A[((i * 2) + ((0 * 2) + k))] * B[((0 * 4) + ((k * 2) + j))]));\n"
        "}\n"
        "out[((i * 2) + j)] = acc[0];\n"
        "}\n"
        "}"
    )



def test_rangeify_test_param():
    a = P("a", "int", 3)
    ranges,body  = rangeify(a)

    assert len(ranges) ==1 
    assert ranges[0].op == "RANGE"
    assert ranges[0].arg == "i"
    assert run(ranges[0].src[0]) == 3

    assert body.op == "LOAD"
    assert render(body) == "a[i]"


def test_schedule_rangeified_param():
    a = P("a", "int", 3)
    ranges, body = rangeify(a)

    scheduled = schedule(ranges, body)
    out, loop, store, end = scheduled

    assert out.op == "ALLOC"
    assert out.arg == ParamArg("out", "int", 3)
    assert loop is ranges[0]

    assert store.op == "STORE"
    pointer, value = store.src
    assert pointer.op == "INDEX"
    assert pointer.src == (out, V("i"))
    assert value is body

    assert end.op == "END"
    assert end.src == (out, loop)
def test_rangeify_neg():
    a = P("a", "int", 3)
    expr = UOp("NEG", (a,))
    ranges,body = rangeify(expr)
    assert len(ranges) == 1
    assert ranges[0].op == "RANGE"
    assert ranges[0].arg == "i"
    assert run(ranges[0].src[0]) == 3

    assert body.op == "NEG"
    assert render(body) == "(-a[i])"


def test_schedule_neg():
    a  = P("a","int", 3)
    expr =UOp("NEG", (a,))
    ranges,body = rangeify(expr)
    scheduled = schedule(ranges, body)
    out,loop, store, end = scheduled
    assert out.op == "ALLOC"
    assert out.arg == ParamArg("out", "int", 3)

    assert loop is ranges[0]

    assert store.op == "STORE"

    pointer, value = store.src

    assert pointer.op == "INDEX"
    assert pointer.src == (out, V("i"))
    assert value is body

    assert end.op == "END"
    assert end.src == (out, loop)


def test_rangeify_reshape():
    a = P("a", "int", 6)
    expr = UOp("RESHAPE", (a,), arg=(2,3))
    ranges,body = rangeify(expr)
    assert len(ranges) == 2

    assert tuple(r.arg for r in ranges) == ("i", "j")
    assert tuple(run(r.src[0]) for r in ranges) == (2, 3)

    assert body.op == "LOAD"
    assert render(body) == "a[((i * 3) + j)]"


def test_rangeify_permute():
    a = P("a", "int", 6)
    reshaped = UOp("RESHAPE", (a,), arg=(2,3))
    expr = UOp("PERMUTE", (reshaped, ), arg=(1,0))
    ranges,body = rangeify(expr)
    assert len(ranges) == 2
    assert tuple(r.arg for r in ranges) == ("i", "j")
    assert tuple(run(r.src[0]) for r in ranges) == (3, 2)

    assert body.op == "LOAD"
    assert render(body) == "a[((j * 3) + i)]"


def test_rangeify_expand():
    a  = P("a", "int", 1)
    expr = UOp("EXPAND", (a,), arg=(3,))
    ranges,body = rangeify(expr)
    assert len(ranges) == 1
    assert ranges[0].arg == "i"
    assert run(ranges[0].src[0]) == 3

    assert body.op == "LOAD"
    assert render(body) == "a[0]"


def test_rangeify_flip():
    a = P("a", "int", 3)
    expr = UOp("FLIP", (a,))

    ranges, body = rangeify(expr)

    assert len(ranges) == 1
    assert ranges[0].arg == "i"
    assert run(ranges[0].src[0]) == 3

    assert body.op == "LOAD"
    assert render(body) == "a[(2 - i)]"

def test_rangeify_shrink():
    a = P("a", "int", 5)
    expr = UOp("SHRINK", (a,), arg=((2, 5),))

    ranges, body = rangeify(expr)

    assert len(ranges) == 1
    assert ranges[0].arg == "i"
    assert run(ranges[0].src[0]) == 3

    assert body.op == "LOAD"
    assert render(body) == "a[(i + 2)]"


def test_rangeify_add():
    a = P("a", "int", 3)
    b = P("b", "int", 3)
    expr = UOp("ADD", (a, b))

    ranges, body = rangeify(expr)

    assert len(ranges) == 1
    assert ranges[0].arg == "i"
    assert run(ranges[0].src[0]) == 3
    assert body.op == "ADD"
    assert render(body) == "(a[i] + b[i])"


def test_rangeify_pad():
    a = P("a", "int", 5)
    expr = UOp("PAD", (a,), arg=((1, 0),))

    ranges, body = rangeify(expr)

    assert len(ranges) == 1
    assert ranges[0].arg == "i"
    assert run(ranges[0].src[0]) == 6
    assert body.op == "WHERE"
    assert render(body) == "((i < 1) ? 0 : a[(i + -1)])"


def test_rangeify_reshape_3d():
    a = P("a", "int", 24)
    expr = UOp("RESHAPE", (a,), arg=(2, 3, 4))

    ranges, body = rangeify(expr)

    assert len(ranges) == 3
    assert tuple(r.arg for r in ranges) == ("i", "j", "k")
    assert tuple(run(r.src[0]) for r in ranges) == (2, 3, 4)
    assert body.op == "LOAD"
    assert render(body) == "a[((i * 12) + ((j * 4) + k))]"


def test_rangeify_permute_3d():
    a = UOp("RESHAPE", (P("a", "int", 24),), arg=(2, 3, 4))
    expr = UOp("PERMUTE", (a,), arg=(2, 0, 1))

    ranges, body = rangeify(expr)

    assert len(ranges) == 3
    assert tuple(r.arg for r in ranges) == ("i", "j", "k")
    assert tuple(run(r.src[0]) for r in ranges) == (4, 2, 3)
    assert body.op == "LOAD"
    assert render(body) == "a[((j * 12) + ((k * 4) + i))]"


def test_rangeify_sub():
    a = P("a", "int", 3)
    b = P("b", "int", 3)
    expr = UOp("SUB", (a, b))

    ranges, body = rangeify(expr)

    assert len(ranges) == 1
    assert ranges[0].arg == "i"
    assert run(ranges[0].src[0]) == 3
    assert body.op == "SUB"
    assert render(body) == "(a[i] - b[i])"


def test_rangeify_mul():
    a = P("a", "int", 3)
    b = P("b", "int", 3)
    expr = UOp("MUL", (a, b))

    ranges, body = rangeify(expr)

    assert len(ranges) == 1
    assert ranges[0].arg == "i"
    assert run(ranges[0].src[0]) == 3
    assert body.op == "MUL"
    assert render(body) == "(a[i] * b[i])"


def test_rangeify_expand_2d():
    a = UOp("RESHAPE", (P("a", "int", 3),), arg=(1, 3))
    expr = UOp("EXPAND", (a,), arg=(2, 3))

    ranges, body = rangeify(expr)

    assert tuple(r.arg for r in ranges) == ("i", "j")
    assert tuple(run(r.src[0]) for r in ranges) == (2, 3)
    assert body.op == "LOAD"
    assert render(body) == "a[((0 * 3) + j)]"


def test_rangeify_flip_2d():
    a = UOp("RESHAPE", (P("a", "int", 6),), arg=(2, 3))
    expr = UOp("FLIP", (a,), arg=(0, 1))

    ranges, body = rangeify(expr)

    assert tuple(r.arg for r in ranges) == ("i", "j")
    assert tuple(run(r.src[0]) for r in ranges) == (2, 3)
    assert body.op == "LOAD"
    assert render(body) == "a[(((1 - i) * 3) + (2 - j))]"


def test_rangeify_shrink_2d():
    a = UOp("RESHAPE", (P("a", "int", 12),), arg=(3, 4))
    expr = UOp("SHRINK", (a,), arg=((1, 3), (0, 4)))

    ranges, body = rangeify(expr)

    assert tuple(r.arg for r in ranges) == ("i", "j")
    assert tuple(run(r.src[0]) for r in ranges) == (2, 4)
    assert body.op == "LOAD"
    assert render(body) == "a[(((i + 1) * 4) + j)]"


def test_rangeify_composed_views():
    a = P("a", "int", 12)

    reshaped = UOp("RESHAPE", (a,), arg=(3, 4))
    permuted = UOp("PERMUTE", (reshaped,), arg=(1, 0))
    expr = UOp("FLIP", (permuted,), arg=(0,))

    ranges, body = rangeify(expr)

    assert tuple(r.arg for r in ranges) == ("i", "j")
    assert tuple(run(r.src[0]) for r in ranges) == (4, 3)
    assert body.op == "LOAD"
    assert render(body) == "a[((j * 4) + (3 - i))]"


def test_rangeify_elementwise_view():
    a = P("a", "int", 1)
    b = P("b", "int", 3)

    expanded = UOp("EXPAND", (a,), arg=(3,))
    expr = UOp("ADD", (expanded, b))

    ranges, body = rangeify(expr)

    assert len(ranges) == 1
    assert ranges[0].arg == "i"
    assert run(ranges[0].src[0]) == 3
    assert body.op == "ADD"
    assert render(body) == "(a[0] + b[i])"
