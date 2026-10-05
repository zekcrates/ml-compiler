import math
import weakref 
class Var:
    def __init__(self, name):
        self.name = name


class UOp:

    _cache = weakref.WeakValueDictionary()

    def __new__(cls, op,src=(), arg=None):
        src = tuple(src)
        key = (op,src,type(arg) , arg)
        if key in cls._cache:
            return cls._cache[key]

        obj = super().__new__(cls)
        cls._cache[key] = obj 
        return obj 
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
    def __hash__(self):
        return hash((self.op, self.src, self.arg))
    @property
    def dtype(self):
        if self.op == "CONST": return type(self.arg).__name__
        if self.op in ("ADD", "MUL", "MAX", "FLOORDIV", "FLOORMOD") : return promote_dtype(self.src[0].dtype, self.src[1].dtype)
        if self.op == "PARAM": return self.arg.dtype 
        if self.op in ("BUFFER", "ALLOC") : return self.arg.dtype 
        if self.op in ("INDEX" ,"LOAD", "RESHAPE", "SHRINK", "FLIP", "PAD", "NEG", "EXP2", "LOG2", "STACK") : return self.src[0].dtype 
        if self.op == "VAR" : return "int"
        if self.op in ("CMPNE", "CMPLT"): return "bool"
        if self.op  == "CAST": return self.arg 
        if self.op == "LOG2":  return "float"
        if self.op in ( "STORE", "LINEAR") : return "void"

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

        if self.op in  ( "INDEX","LOAD", "STORE", "VAR", "LINEAR"):
            return ()

        if self.op == "RESHAPE":
            old_shape = self.src[0].shape 
            new_shape = self.arg 
            old_size =1 
            for x in old_shape:
                old_size *= x 
            new_size = 1 
            for x in new_shape:
                new_size *= x 
            if new_size != old_size :
                raise ValueError("reshape mismatch")

            return new_shape

        if self.op == "SHRINK":
            output = []
            old_shape = self.src[0].shape

            if len(old_shape) != len(self.arg):
                raise ValueError("axes dont match")

            for axis,pair in enumerate(self.arg):
                start, end = pair
                size = old_shape[axis]


                if start < 0 :
                    raise ValueError("start out of bounds")

                if start > end:
                    raise ValueError("invalid range")

                if end > size :
                    raise ValueError("end out of bounds")

                output.append(end-start)

            return tuple(output)
        

        if self.op == "FLIP":
            return self.src[0].shape

        if self.op == "PAD":
            old_shape = self.src[0].shape 
            new_shape =  list(old_shape) 
            for axis, pair in enumerate(self.arg):
                left, right = pair 
                size = old_shape[axis]
                total = left + size + right
                new_shape[axis] = total 

            return tuple(new_shape)

        if self.op == "STACK":
            old_shape = self.src[0].shape 
            if not all(src.shape == old_shape for src in self.src):
                raise ValueError("all STACK sources must have the same shape")
            n = len(self.src)
            new_shape =  (n,) + old_shape
            return tuple(new_shape)

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

    def __eq__(self, other):
        return (isinstance(other, ParamArg) and self.name == other.name and self.dtype == other.dtype and self.size == other.size)

    def __hash__(self):
        return hash((self.name, self.dtype, self.size))
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
        return a+b 

    if expr.op == "MUL":
        a = run(expr.src[0])
        b = run(expr.src[1])
        if isinstance(a, bool) and isinstance(b, bool):
            return a and b 
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
        return  a // b 
    if expr.op == "FLOORMOD":
        a =run(expr.src[0])
        b = run(expr.src[1])
        return a % b 

    if expr.op == "CAST":
        value = run(expr.src[0])
        if expr.arg == "int": return int(value)
        if expr.arg == "float": return float(value)
        if expr.arg == "bool" : return bool(value)


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
def toposort(expr):
    seen = set() 
    out = []
    def walk(node):
        if node in seen : return 
        for child in node.src : walk(child)
        seen.add(node)
        out.append(node)

    walk(expr)
    return tuple(out)


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
        return UOp("DIV", (a,b))
    if expr.op == "RECIP":
        a = simplify(expr.src[0])
        
        if a.op == "CONST"  :
            return UOp("CONST", arg= 1/a.arg)
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
            return UOp("CONST", arg=max(a.arg,b.arg))

        return UOp("MAX", (a,b))

    if expr.op == "CMPNE":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg= not a==b)
        return UOp("CMPNE", (a, b))

    if expr.op == "CMPLT":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg= a.arg < b.arg )
        return UOp("CMPLT", (a,b))


    if expr.op == "FLOORDIV":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if a.op == "CONST"  and b.op == "CONST":
            return UOp("CONST", arg=a.arg // b.arg )
        return UOp("FLOORDIV", (a,b))


    if expr.op == "FLOORMOD":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg= a.arg % b.arg )
        return UOp("FLOORMOD", (a,b))

    if expr.op == "CAST":
        a  = simplify(expr.src[0])
        if a.op == "CONST":
            if expr.arg == "int": return UOp("CONST",arg=int(a.arg))
            if expr.arg == "float": return UOp("CONST", arg=float(a.arg))
            if expr.arg == "bool" : return UOp("CONST", arg=bool(a.arg))

        return UOp("CAST", (a,), arg=expr.arg)

    if expr.op == "EXP2":
        a =simplify(expr.src[0])
        if a.op == "CONST":
            return UOp("CONST", arg= 2** a.arg)
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

    if expr.op == "CAST":
        return f"(({expr.arg}){render(expr.src[0])})"
    if expr.op == "RANGE":
        start = render(expr.src[0])
        end = render(expr.src[1])
        return f"for (int {expr.arg} = {start}; {expr.arg} < {end}; {expr.arg}++)"


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

    if expr.op == "MUL":
        return f"({left} * {right})"

    if expr.op == "DIV":
        return f"({left} / {right})"

    if expr.op == "MAX":
        left= render(expr.src[0])
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

    def __eq__(self, other):
        return (isinstance(other, CallArg) and self.name == other.name )
    def __hash__(self):
        return hash(self.name)
    


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


def linearize(expr):
    return UOp("LINEAR", toposort(expr))