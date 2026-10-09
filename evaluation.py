import math

from uops import UOp


def run(expr):
    if expr.op == "CONST":
        return expr.arg
    if expr.op == "EXP2":
        return 2 ** run(expr.src[0])
    if expr.op == "AFTER":
        a = expr.src[0]
        b = expr.src[1]
        run(b)
        return run(a)
    if expr.op == "ADD":
        a = run(expr.src[0])
        b = run(expr.src[1])
        return a + b

    if expr.op == "SUB":
        a = run(expr.src[0])
        b = run(expr.src[1])
        return a - b

    if expr.op == "MUL":
        a = run(expr.src[0])
        b = run(expr.src[1])
        if isinstance(a, bool) and isinstance(b, bool):
            return a and b
        return a * b

    if expr.op == "DIV":
        a = run(expr.src[0])
        b = run(expr.src[1])
        return a / b
    if expr.op == "RECIP":
        a = run(expr.src[0])
        return 1.0 / a
    if expr.op == "WHERE":
        cond = run(expr.src[0])
        left = run(expr.src[1])
        right = run(expr.src[2])
        if cond:
            return left
        return right

    if expr.op == "NEG":
        a = run(expr.src[0])
        return -a

    if expr.op == "MAX":
        a = run(expr.src[0])
        b = run(expr.src[1])
        if isinstance(a, bool) and isinstance(b, bool):
            return a or b
        return max(a, b)

    if expr.op == "CMPNE":
        a = run(expr.src[0])
        b = run(expr.src[1])
        return not a == b

    if expr.op == "CMPLT":
        a = run(expr.src[0])
        b = run(expr.src[1])
        return a < b

    if expr.op == "FLOORDIV":
        a = run(expr.src[0])
        b = run(expr.src[1])
        return a // b
    if expr.op == "FLOORMOD":
        a = run(expr.src[0])
        b = run(expr.src[1])
        return a % b

    if expr.op == "CAST":
        value = run(expr.src[0])
        if expr.arg == "int":
            return int(value)
        if expr.arg == "float":
            return float(value)
        if expr.arg == "bool":
            return bool(value)

    if expr.op == "LOG2":
        return math.log2(run(expr.src[0]))

    if expr.op == "CALL":
        body = expr.src[0]
        args = expr.src[1:]

        params = []

        def collect(node):
            if node.op == "PARAM":
                if node.arg.name not in [p.arg.name for p in params]:
                    params.append(node)
                return

            for child in node.src:
                collect(child)

        collect(body)

        env = {
            param.arg.name: run(arg)
            for param, arg in zip(params, args)
        }

        return eval_call_body(body, env)


def eval_call_body(node, env):
    if node.op == "PARAM":
        return env[node.arg.name]

    if node.op == "CONST":
        return node.arg

    if node.op == "ADD":
        return eval_call_body(node.src[0], env) + eval_call_body(node.src[1], env)

    if node.op == "MUL":
        return eval_call_body(node.src[0], env) * eval_call_body(node.src[1], env)

    raise NotImplementedError(node.op)


def simplify(expr):
    if expr.op == "PARAM" or expr.op == "CONST" or expr.op == "VAR":
        return expr

    if expr.op == "ADD":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if a.op == "CONST" and a.arg == 0:
            return b
        elif b.op == "CONST" and b.arg == 0:
            return a

        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=a.arg + b.arg)
        return UOp("ADD", src=(a, b))

    if expr.op == "SUB":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if b.op == "CONST" and b.arg == 0:
            return a

        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=a.arg - b.arg)
        return UOp("SUB", (a, b))

    if expr.op == "MUL":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])

        if a.op == "CONST" and a.arg == 0:
            return a
        if b.op == "CONST" and b.arg == 0:
            return b
        if a.op == "CONST" and a.arg == 1:
            return b
        if b.op == "CONST" and b.arg == 1:
            return a

        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=a.arg * b.arg)

        return UOp("MUL", (a, b))

    if expr.op == "DIV":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if b.op == "CONST" and b.arg == 1:
            return a
        if a.op == "CONST" and a.arg == 0:
            return a
        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=a.arg / b.arg)
        return UOp("DIV", (a, b))

    if expr.op == "RECIP":
        a = simplify(expr.src[0])
        if a.op == "CONST":
            return UOp("CONST", arg=1 / a.arg)
        return UOp("RECIP", (a,))

    if expr.op == "WHERE":
        cond = simplify(expr.src[0])
        a = simplify(expr.src[1])
        b = simplify(expr.src[2])
        if cond.op == "CONST":
            return a if cond.arg else b
        return UOp("WHERE", (cond, a, b))

    if expr.op == "NEG":
        a = simplify(expr.src[0])
        if a.op == "CONST":
            return UOp("CONST", arg=-a.arg)
        if a.op == "NEG":
            return a.src[0]
        return UOp("NEG", (a,))

    if expr.op == "MAX":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=max(a.arg, b.arg))
        return UOp("MAX", (a, b))

    if expr.op == "CMPNE":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=not a == b)
        return UOp("CMPNE", (a, b))

    if expr.op == "CMPLT":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=a.arg < b.arg)
        return UOp("CMPLT", (a, b))

    if expr.op == "FLOORDIV":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=a.arg // b.arg)
        return UOp("FLOORDIV", (a, b))

    if expr.op == "FLOORMOD":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=a.arg % b.arg)
        return UOp("FLOORMOD", (a, b))

    if expr.op == "CAST":
        a = simplify(expr.src[0])
        if a.op == "CONST":
            if expr.arg == "int":
                return UOp("CONST", arg=int(a.arg))
            if expr.arg == "float":
                return UOp("CONST", arg=float(a.arg))
            if expr.arg == "bool":
                return UOp("CONST", arg=bool(a.arg))
        return UOp("CAST", (a,), arg=expr.arg)

    if expr.op == "EXP2":
        a = simplify(expr.src[0])
        if a.op == "CONST":
            return UOp("CONST", arg=2 ** a.arg)
        return UOp("EXP2", (a,))

    if expr.op == "LOG2":
        a = simplify(expr.src[0])
        if a.op == "CONST":
            return UOp("CONST", arg=math.log2(a.arg))
        return UOp("LOG2", (a,))


def run_elementwise(expr, inputs):
    if expr.op == "PARAM":
        return inputs[expr.arg.name]

    if expr.op == "CONST":
        return expr.arg

    if expr.op in ("FLIP", "PAD", "SHRINK", "EXPAND", "RESHAPE"):
        data = run_elementwise(expr.src[0], inputs)
        if expr.op == "FLIP":
            return data[::-1]
        elif expr.op == "PAD":
            ((l, r),) = expr.arg
            return [0] * l + data + [0] * r
        if expr.op == "SHRINK":
            ((l, r),) = expr.arg
            return data[l:r]
        if expr.op == "EXPAND":
            return data * (expr.arg[0] // len(data)) if len(data) == 1 else list(data)
        if expr.op == "RESHAPE":
            return list(data)

    op, src = expr.op, expr.src
    left = src[0]
    right = src[1]

    left = run_elementwise(left, inputs)
    right = run_elementwise(right, inputs)

    if isinstance(left, list) and isinstance(right, list):
        if op == "ADD":
            return [a + b for a, b in zip(left, right)]
        if op == "MUL":
            return [a * b for a, b in zip(left, right)]
        if op == "DIV":
            return [a / b for a, b in zip(left, right)]

    if isinstance(left, list) and isinstance(right, int):
        if op == "ADD":
            return [l + right for l in left]
        if op == "MUL":
            return [l * right for l in left]
        if op == "DIV":
            return [l / right for l in left]

    if isinstance(left, int) and isinstance(right, list):
        if op == "ADD":
            return [left + a for a in right]
        if op == "MUL":
            return [left * a for a in right]
        if op == "DIV":
            return [left / a for a in right]
