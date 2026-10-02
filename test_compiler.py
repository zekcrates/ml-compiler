import pytest
from compiler import *


def C(value):
    return UOp("CONST", arg=value)


def P(name, dtype="float", size=None):
    return UOp("PARAM", arg=ParamArg(name, dtype, size))


def B(name, dtype="float", size=10):
    return UOp("BUFFER", arg=ParamArg(name, dtype, size))


# ============================================================
# UOP CORE
# ============================================================

def test_uop_core_structure():
    a, b = C(2), C(3)
    x = UOp("ADD", [a, b])

    assert x.op == "ADD"
    assert x.src == (a, b)
    assert isinstance(x.src, tuple)
    assert x.arg is None


def test_uop_structural_equality():
    assert C(5) == C(5)
    assert UOp("ADD", (C(2), C(3))) == UOp("ADD", (C(2), C(3)))


def test_paramarg():
    arg = ParamArg("x", "float", 3)

    assert (arg.name, arg.dtype, arg.size) == ("x", "float", 3)


# ============================================================
# RUN / EVALUATION
# ============================================================

@pytest.mark.parametrize(
    "op,a,b,expected",
    [
        ("ADD", 2, 3, 5),
        ("MUL", 4, 5, 20),
        ("DIV", 8.0, 2.0, 4.0),
    ],
)
def test_run_binary(op, a, b, expected):
    assert run(UOp(op, (C(a), C(b)))) == expected


def test_run_nested():
    expr = UOp("MUL", (UOp("ADD", (C(1), C(2))), C(3)))
    assert run(expr) == 9


def test_run_recip():
    assert run(UOp("RECIP", (C(4.0),))) == 0.25


@pytest.mark.parametrize("cond,expected", [(True, 10), (False, 20)])
def test_run_where(cond, expected):
    expr = UOp("WHERE", (C(cond), C(10), C(20)))
    assert run(expr) == expected


def test_after_returns_first_value():
    assert run(UOp("AFTER", (C(5), C(10)))) == 5


# ============================================================
# GRAPH WALKING
# ============================================================

@pytest.mark.parametrize(
    "expr,expected",
    [
        (C(5), []),
        (UOp("ADD", (C(2), C(3))), ["ADD"]),
        (
            UOp(
                "MUL",
                (
                    UOp("ADD", (C(1), C(2))),
                    UOp("SUB", (C(5), C(3))),
                ),
            ),
            ["ADD", "SUB", "MUL"],
        ),
    ],
)
def test_ops_dependency_order(expr, expected):
    assert ops(expr) == expected


# ============================================================
# REWRITES
# ============================================================

@pytest.mark.parametrize(
    "op,left,right,expected",
    [
        ("ADD", UOp("VAR", arg="x"), C(0), UOp("VAR", arg="x")),
        ("ADD", C(0), UOp("VAR", arg="x"), UOp("VAR", arg="x")),
        ("MUL", UOp("VAR", arg="x"), C(1), UOp("VAR", arg="x")),
        ("MUL", C(1), UOp("VAR", arg="x"), UOp("VAR", arg="x")),
        ("MUL", UOp("VAR", arg="x"), C(0), C(0)),
        ("MUL", C(0), UOp("VAR", arg="x"), C(0)),
        ("DIV", UOp("VAR", arg="x"), C(1.0), UOp("VAR", arg="x")),
    ],
)
def test_simplify_identities(op, left, right, expected):
    assert simplify(UOp(op, (left, right))) == expected


@pytest.mark.parametrize(
    "expr,expected",
    [
        (UOp("ADD", (C(2), C(3))), C(5)),
        (UOp("MUL", (C(5), C(4))), C(20)),
        (UOp("DIV", (C(8.0), C(2.0))), C(4.0)),
        (UOp("RECIP", (C(4.0),)), C(0.25)),
    ],
)
def test_constant_folding(expr, expected):
    assert simplify(expr) == expected


def test_nested_constant_folding():
    expr = UOp("MUL", (UOp("ADD", (C(2), C(3))), C(4)))
    assert simplify(expr) == C(20)


@pytest.mark.parametrize("cond,expected", [(True, C(10)), (False, C(20))])
def test_simplify_where_constant_condition(cond, expected):
    expr = UOp("WHERE", (C(cond), C(10), C(20)))
    assert simplify(expr) == expected


# ============================================================
# RECURSIVE PROPERTIES
# ============================================================

@pytest.mark.parametrize(
    "value,expected",
    [(5, "int"), (5.3, "float"), (True, "bool")],
)
def test_const_dtype(value, expected):
    assert C(value).dtype == expected


@pytest.mark.parametrize(
    "expr,expected",
    [
        (UOp("ADD", (C(2), C(3))), "int"),
        (UOp("ADD", (C(2), C(3.5))), "float"),
        (UOp("MUL", (C(True), C(4))), "int"),
    ],
)
def test_dtype_propagation(expr, expected):
    assert expr.dtype == expected


def test_param_properties():
    x = P("x", "float", 3)

    assert x.dtype == "float"
    assert x.shape == (3,)


def test_var_properties():
    i = UOp("VAR", arg="i")

    assert i.dtype == "int"
    assert i.shape == ()
    assert i.addrspace == "ALU"


def test_elementwise_shape_propagation():
    a = P("a", "float", 3)
    b = P("b", "float", 3)

    assert UOp("ADD", (a, b)).shape == (3,)
    assert UOp("MUL", (a, b)).shape == (3,)


def test_shape_mismatch():
    a = P("a", "float", 3)
    b = P("b", "float", 4)

    with pytest.raises(ValueError):
        _ = UOp("ADD", (a, b)).shape


@pytest.mark.parametrize("op", ["CONST", "ADD", "MUL"])
def test_alu_addrspace(op):
    expr = C(1) if op == "CONST" else UOp(op, (C(1), C(2)))
    assert expr.addrspace == "ALU"


@pytest.mark.parametrize("op", ["BUFFER", "ALLOC"])
def test_memory_object_properties(op):
    x = UOp(op, arg=ParamArg("a", "float", 10))

    assert x.addrspace == "MEM"
    assert x.dtype == "float"
    assert x.shape == (10,)


# ============================================================
# MEMORY IR
# ============================================================

def test_index_load_store_properties():
    buf = B("a", "float", 10)
    out = B("out", "float", 10)
    idx = C(3)

    ptr = UOp("INDEX", (buf, idx))
    load = UOp("LOAD", (ptr,))
    store = UOp("STORE", (UOp("INDEX", (out, idx)), load))

    assert ptr.dtype == "float"
    assert ptr.shape == ()
    assert load.dtype == "float"
    assert load.shape == ()
    assert store.dtype == "float"
    assert store.shape == ()


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
    expr = UOp(op, (P("a", "int", len(a)), P("b", "int", len(b))))
    assert run_elementwise(expr, {"a": a, "b": b}) == expected


def test_elementwise_nested():
    expr = UOp(
        "MUL",
        (
            UOp("ADD", (P("a", "int", 3), P("b", "int", 3))),
            C(2),
        ),
    )

    assert run_elementwise(
        expr,
        {"a": [1, 2, 3], "b": [4, 5, 6]},
    ) == [10, 14, 18]


# ============================================================
# RENDERER
# ============================================================

@pytest.mark.parametrize(
    "expr,expected",
    [
        (C(5), "5"),
        (UOp("VAR", arg="i"), "i"),
        (UOp("ADD", (C(2), C(3))), "(2 + 3)"),
        (UOp("MUL", (C(2), C(3))), "(2 * 3)"),
        (UOp("DIV", (C(8.0), C(2.0))), "(8.0 / 2.0)"),
        (UOp("RECIP", (C(4.0),)), "(1.0 / 4.0)"),
    ],
)
def test_render_expression(expr, expected):
    assert render(expr) == expected


def test_render_symbolic_memory_expression():
    a = B("a", "float", 10)
    out = B("out", "float", 10)
    i = UOp("VAR", arg="i")

    value = UOp(
        "ADD",
        (
            UOp("LOAD", (UOp("INDEX", (a, i)),)),
            C(2.0),
        ),
    )
    store = UOp("STORE", (UOp("INDEX", (out, i)), value))

    assert render(store) == "out[i] = (a[i] + 2.0);"


def test_render_symbolic_index_offset():
    a = B("a", "float", 10)
    i = UOp("VAR", arg="i")
    idx = UOp("ADD", (i, C(1)))

    assert render(UOp("INDEX", (a, idx))) == "a[(i + 1)]"


def test_render_call():
    expr = UOp("CALL", (C(2), C(3)), arg=CallArg("foo"))
    assert render(expr) == "foo(2, 3)"


# ============================================================
# LOOPLESS C FUNCTION
# ============================================================

def test_render_c_function():
    a = B("a", "float", 10)
    out = B("out", "float", 10)
    i = P("i", "int", None)

    value = UOp(
        "ADD",
        (
            UOp("LOAD", (UOp("INDEX", (a, i)),)),
            C(2.0),
        ),
    )
    store = UOp("STORE", (UOp("INDEX", (out, i)), value))

    code = render_function("kernel", store, params=(a, out, i))

    assert code == (
        "void kernel(float *a, float *out, int i) {\n"
        "    out[i] = (a[i] + 2.0);\n"
        "}\n"
    )
