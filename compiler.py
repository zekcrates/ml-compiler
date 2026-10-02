class Var:
    def __init__(self, name):
        self.name = name


class UOp:
    def __init__(self, op, src=(), arg=None ):
        self.op = op
        self.src = tuple(src)
        self.arg = arg 

    def __eq__(self, other):
        if not isinstance(other, UOp):return False
        if self.op != other.op : return False 
        if self.arg != other.arg : return False 
        if self.src != other.src : return False 
        return True 

    @property
    def dtype(self):
        if self.op == "CONST": return type(self.arg).__name__
        if self.op in ("ADD", "MUL") : return promote_dtype(self.src[0].dtype, self.src[1].dtype)
        if self.op == "PARAM": return self.arg.dtype 
        if self.op in ("BUFFER", "ALLOC") : return self.arg.dtype 
        if self.op in ("INDEX" ,"LOAD", "STORE") : return self.src[0].dtype 
        if self.op == "VAR" : return "int"
    @property
    def shape(self):
        if self.op == "CONST": return ()
        if self.op in ("ADD", "MUL")  : 
            if self.src[0].shape != self.src[1].shape :
                raise ValueError("Shapes don't match")
            return self.src[0].shape 
        if self.op == "PARAM": return (self.arg.size,) 
        if self.op in ("BUFFER", "ALLOC"):
            return (self.arg.size, ) 

        if self.op in  ( "INDEX","LOAD", "STORE", "VAR"):
            return ()

    @property
    def addrspace(self):
        if self.op in ("CONST", "PARAM", "ADD", "MUL", "VAR"):
            return "ALU"
        if self.op in ("BUFFER", "ALLOC"): return "MEM"
class ParamArg:
    def __init__(self, name, dtype,size=None):
        self.name = name 
        self.dtype = dtype 
        self.size = size 

def run(expr):
    if expr.op == "CONST":
        return expr.arg 

    if expr.op == "AFTER":
        a = expr.src[0]
        b = expr.src[1]
        run(b)
        return run(a)
    if expr.op == "ADD":
        a = run(expr.src[0])
        b = run(expr.src[1])    
        return a+b 

    if expr.op == "MUL":
        a = run(expr.src[0])
        b = run(expr.src[1])
        return a *b 

    if expr.op == "DIV":
        a = run(expr.src[0])
        b = run(expr.src[1])
        return a/b 
    if expr.op == "RECIP":
        a = run(expr.src[0])
        return 1.0/a 
    if expr.op == "WHERE":
        cond = run(expr.src[0])
        left = run(expr.src[1])
        right = run(expr.src[2])
        if cond:
            return left 
        else:
            return right 
def ops(expr):
    if expr.op == "CONST" or expr.op == "PARAM":
        return []


    output = []

    if isinstance(expr, UOp):
        for child in expr.src:
            output.extend(ops(child))

        output.append(expr.op)

    return output


def simplify(expr):
    if expr.op == "PARAM" or expr.op == "CONST" or expr.op == "VAR":
        return expr 
    
    if expr.op == "ADD":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        # const 0  + param 
        if a.op == "CONST" and a.arg == 0 :
            return b 
        elif b.op == "CONST" and b.arg == 0 :
            return a 

        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=a.arg + b.arg )
        return UOp("ADD", src=(a,b))
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
        if b.op == "CONST" and b.arg == 1 :
            return a 
        if a.op == "CONST" and a.arg == 0 :
            return a 
        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=a.arg /b.arg )

    if expr.op == "RECIP":
        a = simplify(expr.src[0])
        if a.op == "CONST" and a.arg == 0 :
            return a 

        if a.op == "CONST"  :
            return UOp("CONST", arg= 1/a.arg)

    if expr.op == "WHERE":
        cond = simplify(expr.src[0])
        a = simplify(expr.src[1])
        b = simplify(expr.src[2])
        if cond.op == "CONST":
            return a if cond.arg else b

        return UOp("WHERE", (cond, a, b))
def run_elementwise(expr, inputs):

    if expr.op == "PARAM":
        return inputs[expr.arg.name]

    if expr.op == "CONST":
        return expr.arg 
    op,src = expr.op, expr.src 
    left = src[0]
    right = src[1]


    left = run_elementwise(left, inputs)
    right = run_elementwise(right, inputs)


    # list +/* list 
    if isinstance(left, list) and isinstance(right,list):
        if op == "ADD":
            return [a+ b for a,b in zip(left, right)]

        if op == "MUL":
            return [a * b for a,b in zip(left, right)]

        if op == "DIV":
            return [a/b for a,b in zip(left, right)]

    # list +/ scalar 
    if isinstance(left, list) and isinstance(right, int):
        if op  == "ADD":
            return [l + right for  l in left]
        if op == "MUL":
            return [l * right for l in  left ]
        if op == "DIV":
            return [l / right for l in left ]

    # scalar +/* list 
    if isinstance(left, int) and isinstance(right, list):
        if op == "ADD":
            return [left + a for a in right]

        if op == "MUL":
            return [left * a  for a in right]

        if op == "DIV":
            return [left / a for a  in right ]
def render(expr, indexed=False):
    if expr.op == "CONST":
        return str(expr.arg)

    if expr.op == "VAR":
        return str(expr.arg)
    if expr.op == "PARAM":
    
        return expr.arg.name

    if expr.op == "BUFFER":
        return str(expr.arg.name)

    if expr.op == "INDEX":
        base = render(expr.src[0])
        idx = render(expr.src[1])
        return f"{base}[{idx}]"
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
    left = render(expr.src[0], indexed=indexed)
    right = render(expr.src[1], indexed=indexed)

    if expr.op == "ADD":
        return f"({left} + {right})"

    if expr.op == "MUL":
        return f"({left} * {right})"

    if expr.op == "DIV":
        return f"({left} / {right})"



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


def promote_dtype(a, b):
    order = {
        "bool": 0,
        "int": 1,
        "float": 2,
    }

    return a if order[a] >= order[b] else b





class CallArg:
    def __init__(self, name):
        self.name = name 



def render_function(name, body, params):
    rendered_params = []

    for p in params:
        if p.op == "BUFFER":
            rendered_params.append(
                f"{p.arg.dtype} *{p.arg.name}"
            )

        elif p.op == "PARAM":
            rendered_params.append(
                f"{p.arg.dtype} {p.arg.name}"
            )

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