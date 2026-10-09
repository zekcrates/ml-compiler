def render(expr, indexed=False):
    if expr.op == "CONST":
        return str(expr.arg)

    if expr.op == "EXP2":
        return f"exp2({render(expr.src[0])})"
    if expr.op == "LOG2":
        return f"log2({render(expr.src[0])})"
    if expr.op == "VAR":
        return str(expr.arg)
    if expr.op == "PARAM":
        if indexed:
            return f"{expr.arg.name}[i]"
        return expr.arg.name

    if expr.op == "NEG":
        a = render(expr.src[0])
        return f"(-{a})"

    if expr.op == "END":
        return "}"
    if expr.op == "BUFFER":
        return str(expr.arg.name)

    if expr.op == "INDEX":
        base = expr.src[0]
        idx = render(expr.src[1])
        if base.op in ("BUFFER", "ALLOC"):
            return f"{base.arg.name}[{idx}]"
        return f"{render(base)}[{idx}]"
    if expr.op == "LOAD":
        return render(expr.src[0])

    if expr.op == "STORE":
        ptr = render(expr.src[0])
        value = render(expr.src[1])
        return f"{ptr} = {value};"

    if expr.op == "RECIP":
        return f"(1.0 / {render(expr.src[0])})"

    if expr.op == "WHERE":
        cond = render(expr.src[0])
        left = render(expr.src[1])
        right = render(expr.src[2])
        return f"({cond} ? {left} : {right})"

    if expr.op == "CALL":
        args = ", ".join(render(x) for x in expr.src)
        return f"{expr.arg.name}({args})"

    if expr.op == "CAST":
        return f"(({expr.arg}){render(expr.src[0])})"
    if expr.op == "RANGE":
        end = render(expr.src[-1])
        return f"for (int {expr.arg} = 0; {expr.arg} < {end}; {expr.arg}++)"

    if expr.op == "ALLOC":
        name = expr.arg.name
        size = expr.arg.size
        ty = expr.arg.dtype
        return f"{ty} {name}[{size}];"

    if expr.op == "AFTER":
        store = render(expr.src[1])
        buf = render(expr.src[0])
        return f"{store}\n{buf}"

    if expr.op == "LINEAR":
        return "\n".join(render(x) for x in expr.src)

    left = render(expr.src[0], indexed=indexed)
    right = render(expr.src[1], indexed=indexed)

    if expr.op == "ADD":
        return f"({left} + {right})"

    if expr.op == "SUB":
        return f"({left} - {right})"

    if expr.op == "MUL":
        return f"({left} * {right})"

    if expr.op == "DIV":
        return f"({left} / {right})"

    if expr.op == "MAX":
        left = render(expr.src[0])
        right = render(expr.src[1])
        return f"max({left}, {right})"

    if expr.op == "CMPNE":
        left = render(expr.src[0])
        right = render(expr.src[1])
        return f"({left} != {right})"

    if expr.op == "CMPLT":
        left = render(expr.src[0])
        right = render(expr.src[1])
        return f"({left} < {right})"

    if expr.op == "FLOORDIV":
        left = render(expr.src[0])
        right = render(expr.src[1])
        return f"({left} // {right})"

    if expr.op == "FLOORMOD":
        left = render(expr.src[0])
        right = render(expr.src[1])
        return f"({left} % {right})"


def compile_c(expr, n):
    output = f"for (int i = 0; i < {n}; i++) {{"
    output += "out[i] = "
    output += render(expr, indexed=True)
    output += ";}"
    return output


def compile_c_program(expr, inputs):
    output = "#include <stdio.h>\n"
    output += "int main() {\n"

    for name, values in inputs.items():
        vals = ", ".join(str(x) for x in values)
        output += f"int {name}[] = {{{vals}}};\n"

    n = len(next(iter(inputs.values())))
    output += f"int out[{n}];\n"
    output += compile_c(expr, n)

    output += "\nfor (int i = 0; i < " + str(n) + "; i++) {\n"
    output += 'printf("%d\\n", out[i]);\n'
    output += "}\n"
    output += "return 0;\n"
    output += "}\n"
    return output


def render_function(name, body, params):
    rendered_params = []

    for p in params:
        if p.op == "BUFFER":
            rendered_params.append(f"{p.arg.dtype} *{p.arg.name}")
        elif p.op == "PARAM":
            rendered_params.append(f"{p.arg.dtype} {p.arg.name}")

    signature = f"void {name}(" + ", ".join(rendered_params) + ")"
    body_code = render(body)

    return (
        signature
        + " {\n"
        + "    "
        + body_code
        + "\n"
        + "}\n"
    )
