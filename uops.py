import weakref


class Var:
    def __init__(self, name):
        self.name = name


class UOp:

    _cache = weakref.WeakValueDictionary()

    def __new__(cls, op, src=(), arg=None):
        src = tuple(src)
        key = (op, src, type(arg), arg)
        if key in cls._cache:
            return cls._cache[key]

        obj = super().__new__(cls)
        cls._cache[key] = obj
        return obj

    def __init__(self, op, src=(), arg=None):
        self.op = op
        self.src = tuple(src)
        self.arg = arg

    def __eq__(self, other):
        if not isinstance(other, UOp):
            return False
        if self.op != other.op:
            return False
        if self.arg != other.arg:
            return False
        if self.src != other.src:
            return False
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
        if self.op == "CONST":
            return type(self.arg).__name__
        if self.op in ("ADD", "SUB", "MUL", "MAX", "FLOORDIV", "FLOORMOD"):
            return promote_dtype(self.src[0].dtype, self.src[1].dtype)
        if self.op == "PARAM":
            return self.arg.dtype
        if self.op == "EXPAND":
            return self.src[0].dtype
        if self.op in ("BUFFER", "ALLOC"):
            return self.arg.dtype
        if self.op in (
            "INDEX", "LOAD", "RESHAPE", "SHRINK", "FLIP", "PAD", "NEG",
            "EXP2", "LOG2", "STACK", "END", "PERMUTE", "REDUCE",
        ):
            return self.src[0].dtype
        if self.op in ("VAR", "RANGE"):
            return "int"
        if self.op in ("CMPNE", "CMPLT"):
            return "bool"
        if self.op == "CAST":
            return self.arg
        if self.op == "LOG2":
            return "float"
        if self.op in ("STORE", "LINEAR"):
            return "void"

    @property
    def shape(self):
        if self.op == "CONST":
            return ()
        if self.op in (
            "ADD", "SUB", "MUL", "DIV", "MAX", "CMPNE", "CMPLT",
            "FLOORDIV", "FLOORMOD",
        ):
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

        if self.op in ("NEG", "RECIP", "EXP2", "LOG2", "CAST"):
            return self.src[0].shape

        if self.op == "WHERE":
            sa, sb = self.src[1].shape, self.src[2].shape
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

        if self.op == "PARAM":
            return (self.arg.size,)
        if self.op in ("BUFFER", "ALLOC"):
            return (self.arg.size,)

        if self.op in ("INDEX", "LOAD", "STORE", "VAR", "LINEAR", "RANGE"):
            return ()

        if self.op == "NEG":
            return self.src[0].shape

        if self.op == "RESHAPE":
            old_shape = self.src[0].shape
            new_shape = self.arg
            old_size = 1
            for x in old_shape:
                old_size *= x
            new_size = 1
            for x in new_shape:
                new_size *= x
            if new_size != old_size:
                raise ValueError("reshape mismatch")
            return new_shape

        if self.op == "SHRINK":
            output = []
            old_shape = self.src[0].shape
            if len(old_shape) != len(self.arg):
                raise ValueError("axes dont match")
            for axis, pair in enumerate(self.arg):
                start, end = pair
                size = old_shape[axis]
                if start < 0:
                    raise ValueError("start out of bounds")
                if start > end:
                    raise ValueError("invalid range")
                if end > size:
                    raise ValueError("end out of bounds")
                output.append(end - start)
            return tuple(output)

        if self.op == "FLIP":
            return self.src[0].shape

        if self.op == "PAD":
            old_shape = self.src[0].shape
            new_shape = list(old_shape)
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
            new_shape = (n,) + old_shape
            return tuple(new_shape)

        if self.op == "END":
            return self.src[0].shape

        if self.op == "EXPAND":
            return self.arg

        if self.op == "PERMUTE":
            return tuple(self.src[0].shape[i] for i in self.arg)

        if self.op == "REDUCE":
            return tuple(
                size for axis, size in enumerate(self.src[0].shape)
                if axis not in self.arg
            )

    @property
    def addrspace(self):
        if self.op in ("CONST", "PARAM", "ADD", "MUL", "VAR"):
            return "ALU"
        if self.op in ("BUFFER", "ALLOC"):
            return "MEM"


class ParamArg:
    def __init__(self, name, dtype, size=None):
        self.name = name
        self.dtype = dtype
        self.size = size

    def __eq__(self, other):
        return (
            isinstance(other, ParamArg)
            and self.name == other.name
            and self.dtype == other.dtype
            and self.size == other.size
        )

    def __hash__(self):
        return hash((self.name, self.dtype, self.size))

    def __repr__(self):
        return f"{self.name}: {self.dtype}" + (
            f"[{self.size}]" if self.size is not None else ""
        )


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
        return isinstance(other, CallArg) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name
