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



def test_range_render():
    i = UOp(
        "RANGE",
        (
            UOp("CONST", arg=0),
            UOp("CONST", arg=10),
        ),
        arg="i",
    )

    assert render(i) == "for (int i = 0; i < 10; i++)"



def test_end_render():
    end = UOp("END")

    assert render(end) == "}"



def test_render_loop():
    i = UOp(
        "RANGE",
        (
            UOp("CONST", arg=0),
            UOp("CONST", arg=3),
        ),
        arg="i",
    )

    body = UOp(
        "STORE",
        (
            UOp(
                "INDEX",
                (
                    UOp("BUFFER", arg=ParamArg("out", "float", 3)),
                    UOp("VAR", arg="i"),
                ),
            ),
            UOp("CONST", arg=1.0),
        ),
    )

    end = UOp("END")

    code = "\n".join([
        render(i),
        "    " + render(body),
        render(end),
    ])

    assert code == (
        "for (int i = 0; i < 3; i++)\n"
        "    out[i] = 1.0;\n"
        "}"
    )


def test_reshape_shape():
    x = UOp(
        "PARAM", 
        arg=ParamArg("x", "float" , 6), 
    )

    y = UOp("RESHAPE",  (x,), arg=(2,3),)

    assert y.shape == (2,3)
    assert y.dtype == "float"


def test_reshape_size_mismatch():
    x = UOp(
        "PARAM",
        arg=ParamArg("x", "float", 6),
    )

    y = UOp(
        "RESHAPE",
        (x,),
        arg=(4, 2),
    )

    with pytest.raises(ValueError):
        _ = y.shape


def test_shrink_1d_shape():
    x = UOp(
        "PARAM",
        arg=ParamArg("x", "float", 10),
    )
    y = UOp(
        "SHRINK",(x,) , arg=((2,6),),

    )
    assert y.shape == (4,)
    assert y.dtype== "float"


def test_shrink_2d_shape():
    x = UOp(
        "RESHAPE",
        (
            UOp(
                "PARAM",
                arg=ParamArg("x", "float", 30),
            ),
        ),
        arg=(5, 6),
    )

    y = UOp(
        "SHRINK",
        (x,),
        arg=((1, 4), (2, 5)),
    )

    assert y.shape == (3, 3)
    assert y.dtype == "float"


def test_shrink_wrong_number_of_axes():
    x = UOp(
        "RESHAPE",
        (
            UOp(
                "PARAM",
                arg=ParamArg("x", "float", 30),
            ),
        ),
        arg=(5, 6),
    )

    y = UOp(
        "SHRINK",
        (x,),
        arg=((1, 4),),
    )

    with pytest.raises(ValueError):
        _ = y.shape


def test_shrink_invalid_range():
    x = UOp(
        "PARAM",
        arg=ParamArg("x", "float", 10),
    )

    y = UOp(
        "SHRINK",
        (x,),
        arg=((6, 2),),
    )

    with pytest.raises(ValueError):
        _ = y.shape

def test_shrink_out_of_bounds():
    x = UOp(
        "PARAM",
        arg=ParamArg("x", "float", 10),
    )

    y = UOp(
        "SHRINK",
        (x,),
        arg=((2, 12),),
    )

    with pytest.raises(ValueError):
        _ = y.shape




def test_flip_shape():
    x = UOp(
        "PARAM",
        arg=ParamArg("x", "float", 5),
    )

    y = UOp(
        "FLIP",
        (x,),
        arg=(0,),
    )

    assert y.shape == (5,)
    assert y.dtype == "float"


def test_flip_2d_shape():
    x = UOp(
        "RESHAPE",
        (
            UOp(
                "PARAM",
                arg=ParamArg("x", "float", 12),
            ),
        ),
        arg=(3, 4),
    )

    y = UOp(
        "FLIP",
        (x,),
        arg=(1,),
    )

    assert y.shape == (3, 4)
    assert y.dtype == "float"


def test_pad_1d_shape():
    x = UOp(
        "PARAM",
        arg=ParamArg("x", "float", 5),
    )

    y = UOp(
        "PAD",
        (x,),
        arg=((2, 3),),
    )

    assert y.shape == (10,)
    assert y.dtype == "float"


def test_pad_2d_shape():
    x = UOp(
        "RESHAPE",
        (
            UOp(
                "PARAM",
                arg=ParamArg("x", "float", 12),
            ),
        ),
        arg=(3, 4),
    )

    y = UOp(
        "PAD",
        (x,),
        arg=((1, 2), (3, 1)),
    )

    assert y.shape == (6, 8)
    assert y.dtype == "float"


def test_pad_3d_shape():
    x = UOp(
        "RESHAPE",
        (
            UOp(
                "PARAM",
                arg=ParamArg("x", "float", 24),
            ),
        ),
        arg=(2, 3, 4),
    )

    y = UOp(
        "PAD",
        (x,),
        arg=((1, 1), (2, 0), (0, 3)),
    )

    assert y.shape == (4, 5, 7)
    assert y.dtype == "float"



def test_run_neg_int():
    assert run(UOp("NEG", (C(5),))) == -5
    assert run(UOp("NEG", (C(2.5),))) == -2.5
    assert simplify(UOp("NEG", (C(-5),))) == C(5)
    assert simplify(UOp("NEG", (C(5),))) == C(-5)
    assert render(UOp("NEG", (C(5),))) == "(-5)"

def test_run_max():
    assert run(
        UOp(
            "MAX",
            (
                UOp("CONST", arg=5), 
                UOp("CONST", arg=3), 
            ), 
        )
    )  == 5 

    assert simplify(
        UOp(
            "MAX",
            (
                UOp("CONST", arg=5),
                UOp("CONST", arg=3),
            ),
        )
    ) == UOp("CONST", arg=5)

    assert render(
        UOp(
            "MAX",
            (
                UOp("CONST", arg=5),
                UOp("CONST", arg=3),
            ),
        )
    ) == "max(5, 3)"



def test_run_cmpne():
    assert run(
        UOp("" \
        "CMPNE", 
            (
                UOp("CONST", arg=5), 
                UOp("CONST", arg=3), 
            )
        )
    ) is True 

    assert run(
        UOp(
            "CMPNE",
            (
                UOp("CONST", arg=5),
                UOp("CONST", arg=5),
            ),
        )
    ) is False

    assert simplify(
        UOp(
            "CMPNE",
            (
                UOp("CONST", arg=5),
                UOp("CONST", arg=3),
            ),
        )
    ) == UOp("CONST", arg=True)

    assert render(
        UOp(
            "CMPNE",
            (
                UOp("CONST", arg=5),
                UOp("CONST", arg=3),
            ),
        )
    ) == "(5 != 3)"



def test_run_cmplt():
    assert run(
        UOp(
            "CMPLT",
            (
                UOp("CONST", arg=3),
                UOp("CONST", arg=5),
            ),
        )
    ) is True

    assert run(
        UOp(
            "CMPLT",
            (
                UOp("CONST", arg=5),
                UOp("CONST", arg=3),
            ),
        )
    ) is False

    assert simplify(
        UOp(
            "CMPLT",
            (
                UOp("CONST", arg=3),
                UOp("CONST", arg=5),
            ),
        )
    ) == UOp("CONST", arg=True)

    assert render(
        UOp(
            "CMPLT",
            (
                UOp("CONST", arg=3),
                UOp("CONST", arg=5),
            ),
        )
    ) == "(3 < 5)"



def test_run_floordiv():
    assert run(
        UOp(
            "FLOORDIV",
            (
                UOp("CONST", arg=7),
                UOp("CONST", arg=3),
            ),
        )
    ) == 2

    assert simplify(
        UOp(
            "FLOORDIV",
            (
                UOp("CONST", arg=7),
                UOp("CONST", arg=3),
            ),
        )
    ) == UOp("CONST", arg=2)

    assert render(
        UOp(
            "FLOORDIV",
            (
                UOp("CONST", arg=7),
                UOp("CONST", arg=3),
            ),
        )
    ) == "(7 // 3)"

def test_run_floormod():
    assert run(
        UOp(
            "FLOORMOD",
            (
                UOp("CONST", arg=7),
                UOp("CONST", arg=3),
            ),
        )
    ) == 1

    assert simplify(
        UOp(
            "FLOORMOD",
            (
                UOp("CONST", arg=7),
                UOp("CONST", arg=3),
            ),
        )
    ) == UOp("CONST", arg=1)


    assert render(
        UOp(
            "FLOORMOD",
            (
                UOp("CONST", arg=7),
                UOp("CONST", arg=3),
            ),
        )
    ) == "(7 % 3)"



def test_run_cast():
    assert run(
        UOp(
            "CAST",
            (UOp("CONST", arg=5.7),),
            arg="int",
        )
    ) == 5

    assert run(
        UOp(
            "CAST",
            (UOp("CONST", arg=5),),
            arg="float",
        )
    ) == 5.0

    assert simplify(
        UOp(
            "CAST",
            (UOp("CONST", arg=5.7),),
            arg="int",
        )
    ) == UOp("CONST", arg=5)

    assert render(
        UOp(
            "CAST",
            (UOp("CONST", arg=5),),
            arg="float",
        )
    ) == "((float)5)"



def test_run_exp2():
    assert run(
        UOp(
            "EXP2",
            (UOp("CONST", arg=3),),
        )
    ) == 8

    assert simplify(
        UOp(
            "EXP2",
            (UOp("CONST", arg=3),),
        )
    ) == UOp("CONST", arg=8)

    assert render(
        UOp(
            "EXP2",
            (UOp("CONST", arg=3),),
        )
    ) == "exp2(3)"



def test_run_log2():
    assert run(
        UOp(
            "LOG2",
            (UOp("CONST", arg=8),),
        )
    ) == 3

    assert simplify(
        UOp(
            "LOG2",
            (UOp("CONST", arg=8),),
        )
    ) == UOp("CONST", arg=3)

    assert render(
        UOp(
            "LOG2",
            (UOp("CONST", arg=8),),
        )
    ) == "log2(8)"

def test_uop_hash_consing():
    a = UOp("CONST", arg=5)
    b = UOp("CONST", arg=5)

    assert a == b
    assert a is b


def test_uop_hash_consing_compound():
    a = UOp("CONST", arg=2)
    b = UOp("CONST", arg=3)

    x = UOp("ADD", (a, b))
    y = UOp("ADD", (a, b))

    assert x is y

def test_simplify_recip_nonconstant():
    x = UOp("VAR", arg="x")

    result = simplify(UOp("RECIP", (x,)))

    assert result == UOp("RECIP", (x,))


def test_simplify_recip_zero_raises():
    with pytest.raises(ZeroDivisionError):
        simplify(UOp("RECIP", (C(0),)))



def test_simplify_div_nonconstant():
    x = UOp("VAR", arg="x")
    y = UOp("VAR", arg="y")

    result = simplify(UOp("DIV", (x, y)))

    assert result == UOp("DIV", (x, y))

def test_bool_mul_is_and():
    assert run(UOp("MUL", (C(True), C(True)))) is True
    assert run(UOp("MUL", (C(True), C(False)))) is False
    assert run(UOp("MUL", (C(False), C(False)))) is False


def test_bool_max_is_or():
    assert run(UOp("MAX", (C(False), C(False)))) is False
    assert run(UOp("MAX", (C(False), C(True)))) is True
    assert run(UOp("MAX", (C(True), C(True)))) is True


def test_bool_cmpne_is_xor():
    assert run(UOp("CMPNE", (C(False), C(False)))) is False
    assert run(UOp("CMPNE", (C(False), C(True)))) is True
    assert run(UOp("CMPNE", (C(True), C(False)))) is True
    assert run(UOp("CMPNE", (C(True), C(True)))) is False

def test_param_hash_consing():
    a = UOp("PARAM", arg=ParamArg("x", "float", 3))
    b = UOp("PARAM", arg=ParamArg("x", "float", 3))

    assert a is b

def test_callarg_hash_consing():
    a = UOp("CALL", arg=CallArg("foo"))
    b = UOp("CALL", arg=CallArg("foo"))

    assert a is b

def test_run_call():
    x = UOp("PARAM", arg=ParamArg("x", "int"))
    body = UOp("ADD", (x, C(1)))

    call = UOp(
        "CALL",
        (body, C(5)),
        arg=CallArg("foo"),
    )

    assert run(call) == 6


def test_run_call_two_params():
    x = UOp("PARAM", arg=ParamArg("x", "int"))
    y = UOp("PARAM", arg=ParamArg("y", "int"))

    body = UOp("ADD", (x, y))

    call = UOp(
        "CALL",
        (body, C(5), C(7)),
        arg=CallArg("foo"),
    )

    assert run(call) == 12


def test_run_call_nested_body():
    x = UOp("PARAM", arg=ParamArg("x", "int"))
    y = UOp("PARAM", arg=ParamArg("y", "int"))

    body = UOp(
        "MUL",
        (
            UOp("ADD", (x, y)),
            C(2),
        ),
    )

    call = UOp(
        "CALL",
        (body, C(5), C(7)),
        arg=CallArg("foo"),
    )

    assert run(call) == 24


def test_compile_c_indexes_params():
    a = UOp("PARAM", arg=ParamArg("a", "int", 3))
    b = UOp("PARAM", arg=ParamArg("b", "int", 3))

    expr = UOp("ADD", (a, b))

    code = compile_c(expr, 3)

    assert "out[i] = (a[i] + b[i]);" in code