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