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
    def __repr__(self):
        arg = f", arg={self.arg!r}" if self.arg is not None else ""
        if not self.src:
            return f"UOp({self.op}{arg})"
        srcs = ", ".join(repr(s) for s in self.src)
        return f"UOp({self.op}, src=({srcs}){arg})"

    def tree(self, indent=0):
        line = "  " * indent + self.op
        if self.arg is not None:
            line += f" {self.arg!r}"
        for child in self.src:
            line += "\n" + child.tree(indent + 1)
        return line
    @property
    def dtype(self):
        if self.op == "CONST": return type(self.arg).__name__
        if self.op in ("ADD", "SUB", "MUL", "MAX", "FLOORDIV", "FLOORMOD") : return promote_dtype(self.src[0].dtype, self.src[1].dtype)
        if self.op == "PARAM": return self.arg.dtype 
        if self.op == "EXPAND": return self.src[0].dtype
        if self.op in ("BUFFER", "ALLOC") : return self.arg.dtype 
        if self.op in ("INDEX" ,"LOAD", "RESHAPE", "SHRINK", "FLIP", "PAD", "NEG", "EXP2", "LOG2", "STACK", "END", "PERMUTE", "REDUCE") : return self.src[0].dtype 
        if self.op in ("VAR", "RANGE")  : return "int"
        if self.op in ("CMPNE", "CMPLT"): return "bool"
        if self.op  == "CAST": return self.arg 
        if self.op == "LOG2":  return "float"
        if self.op in ( "STORE", "LINEAR") : return "void"


    @property
    def shape(self):
        if self.op == "CONST": return ()
        if self.op in ("ADD", "SUB", "MUL")  : 
            sa, sb = self.src[0].shape, self.src[1].shape
            if len(sa) != len(sb):
                raise ValueError("Shapes don't match")
            out = []
            for x, y in zip(sa, sb):
                if x == y:
                    out.append(x)
                elif y == 1:
                    out.append(x)
                elif x == 1:
                    out.append(y)
                else:
                    raise ValueError("Shapes don't match")
            return tuple(out)
        if self.op == "PARAM": return (self.arg.size,) 
        if self.op in ("BUFFER", "ALLOC"):
            return (self.arg.size, ) 

        if self.op in  ( "INDEX","LOAD", "STORE", "VAR", "LINEAR", "RANGE"):
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

        if self.op == "END":
            return self.src[0].shape 


        if self.op == "EXPAND":
            return self.arg 

        if self.op == "PERMUTE":
            return tuple(self.src[0].shape[i] for  i in self.arg )

        if self.op == "REDUCE":
            return tuple(size for axis,size in enumerate(self.src[0].shape) if axis not in self.arg )
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
    def __repr__(self):
        return f"{self.name}: {self.dtype}" + (f"[{self.size}]" if self.size is not None else "")
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

    if expr.op == "SUB":
        a = run(expr.src[0])
        b = run(expr.src[1])
        return a-b

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
    if expr.op == "SUB":
        a = simplify(expr.src[0])
        b = simplify(expr.src[1])
        # x - 0 -> x
        if b.op == "CONST" and b.arg == 0:
            return a

        if a.op == "CONST" and b.op == "CONST":
            return UOp("CONST", arg=a.arg - b.arg)
        return UOp("SUB", (a,b))
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

    if expr.op in  ("FLIP", "PAD", "SHRINK", "EXPAND", "RESHAPE"):
        data = run_elementwise(expr.src[0], inputs)
        if expr.op == "FLIP":
            return data[::-1]
        elif expr.op == "PAD":
            ((l, r),) = expr.arg
            return [0] * l + data + [0]* r 
        if expr.op == "SHRINK":
            ((l, r),) = expr.arg

            return data[l:r]
        if expr.op == "EXPAND":
            return data * (expr.arg[0] // len(data)) if len(data) == 1 else list(data)

        if expr.op == "RESHAPE":
            return list(data)

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
    def __repr__(self):
        return self.name
    


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


def lower_indexed(expr, idxs):
    if len(idxs) ==1 :
        i, = idxs 

    if expr.op == "PARAM":
        pointer = UOp("INDEX", (expr, i))
        return UOp("LOAD", (pointer,))

    if expr.op in ("ADD", "SUB", "MUL"):
        return UOp(expr.op, tuple(lower_indexed(s, idxs) for s in expr.src))

    if expr.op == "PAD":
        in_idx = []
        for a , (l,r) in enumerate(expr.arg):
            if l==0: in_idx.append(idxs[a])
            else: 
                in_idx.append(UOp("ADD", (idxs[a], UOp("CONST", arg=-l))))

        data = lower_indexed(expr.src[0], tuple(in_idx))
        result = data 
        for a, (l,r) in reversed(list(enumerate(expr.arg))): 
            l_const = UOp("CONST", arg=l)
            zero = UOp("CONST", arg=0)
            if r > 0 :
                size = expr.src[0].shape[a]
                limit = UOp("ADD", (l_const, UOp("CONST", arg=size)))  
                cond = UOp("CMPLT", (idxs[a], limit))           
                result = UOp("WHERE", (cond, result, zero))
            if l> 0 :
                cond = UOp("CMPLT", (idxs[a], l_const))
                result =  UOp("WHERE", (cond, zero, result))
        return result 
    if expr.op == "SHRINK": 
        in_shape = expr.src[0].shape 
        in_idx = []
        for idx, (l, r)  in enumerate(expr.arg) :
            if l ==0 :
                in_idx.append(idxs[idx])
            else :

                const = UOp("CONST", arg=l)
                in_idx.append(UOp("ADD", (idxs[idx], const)))

        return lower_indexed(expr.src[0], tuple(in_idx))
    if expr.op == "FLIP":
        in_shape = expr.src[0].shape 
        flip_set = expr.arg if expr.arg is not None else tuple(range(len(in_shape)))
        in_idx = []
        for a in  range(len(in_shape)):
            if a in flip_set:
                size_sub = UOp("CONST",arg=in_shape[a]-1)
                in_idx.append(UOp("SUB", (size_sub, idxs[a])))
            else:
                in_idx.append(idxs[a])
        return lower_indexed(expr.src[0], tuple(in_idx))

    if expr.op == "EXPAND":
        in_shape = expr.src[0].shape 
        out_shape = expr.arg 
        in_idx  = []
        for a in range(len(out_shape)):
            if in_shape[a]== 1 : in_idx.append(UOp("CONST", arg=0))
            else: 
                in_idx.append(idxs[a])
        return lower_indexed(expr.src[0], tuple(in_idx))



    if expr.op == "REDUCE":
        op,axis = expr.arg 
        n = expr.src[0].shape[0]
        allc = UOp("ALLOC", arg=ParamArg("acc", expr.src[0].dtype, 1))

        zero_const = UOp("CONST", arg=0)
        idx = UOp("INDEX", (allc,zero_const))
        old_acc = UOp("LOAD", (idx, ))

        if op == "ADD":
            if expr.src[0].dtype == "float":
                start_value = UOp("CONST", arg=0.0)
            else:
                start_value = UOp("CONST", arg=0)
        elif op == "MAX":
            zero_c2 = UOp("CONST", arg=0)
            output = lower_indexed(expr.src[0], (zero_c2, ))
            start_value = output 
        init  = UOp("STORE", (idx, start_value))

        k = UOp("VAR" , arg="k")
        bound_const = UOp("CONST", arg=n)
        loop = UOp("RANGE", (bound_const,), arg="k")
        element =lower_indexed(expr.src[0], (k, ) )

        if op == "ADD":
            combine = UOp("ADD", (old_acc, element))
        elif op == "MAX":
            combine = UOp("MAX", (old_acc, element))

        update = UOp("STORE", (idx, combine))
        end = UOp("END", (allc, loop))
        return UOp("LINEAR", (allc, init, loop, update, end))


    if expr.op == "PERMUTE":

        in_idx = [None]  * len(expr.src[0].shape)
        for k,a in enumerate(expr.arg):
            in_idx[a]  = idxs[k]

        return lower_indexed(expr.src[0], tuple(in_idx))

    if expr.op == "STACK":
        a_size = expr.src[0].shape[0] 
        b_size = expr.src[1].shape[0]
        a_const = UOp("CONST", arg=a_size)
        b_const = UOp("CONST", arg=b_size)
        i_less_a = UOp("CMPLT", (i, a_const))
        i_fd_a = UOp("FLOORMOD", (i , a_const))
        i_fd_b = UOp("FLOORMOD", (i, b_const))
        a_idx = UOp("INDEX", (expr.src[0], i_fd_a))
        b_idx = UOp("INDEX",(expr.src[1], i_fd_b))
        a_ld = UOp("LOAD", (a_idx,))
        b_ld = UOp("LOAD", (b_idx,))

        whr = UOp("WHERE", (i_less_a,a_ld,b_ld))
        return whr 
    if expr.op == "RESHAPE":
        shape = expr.arg 
        flat = idxs[-1]
        stride = 1 
        for a in reversed(range(len(shape)-1)):
            stride *= shape[a+1]
            row_const = UOp("CONST", arg=stride)
            mul_i_row = UOp("MUL", (idxs[a], row_const))
            flat  = UOp("ADD", (mul_i_row, flat))
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

    # store into out at the flattened output index
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

    # A3[i,j,k] = A[i,k] : (M,K) -> (M,1,K) -> (M,N,K)
    a3 = UOp("EXPAND", (UOp("RESHAPE", (A,), arg=(M, 1, K)),), arg=(M, N, K))
    # B3[i,j,k] = B[k,j] : (K,N) -> (1,N,K) -> permute axes -> (M,N,K) via expand
    bt = UOp("PERMUTE", (UOp("RESHAPE", (B,), arg=(1, N, K)),), arg=(0, 2, 1))
    b3 = UOp("EXPAND", (bt,), arg=(M, N, K))

    return UOp("REDUCE", (mul(a3, b3),), arg=("ADD", 2))


def broadcast(a,b):
    sa,sb = a.shape , b.shape 
    if len(sa) != len(sb):
        raise ValueError("broadcast: rank mismatch")

    out = []
    for x,y in zip(sa,sb):
        if x ==y :
            out.append(x)
        elif x==1 :
            out.append(y)
        elif y == 1:
            out.append(x)
        else :
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