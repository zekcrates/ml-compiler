from uops import UOp


def toposort(expr):
    seen = set()
    out = []

    def walk(node):
        if node in seen:
            return
        for child in node.src:
            walk(child)
        seen.add(node)
        out.append(node)

    walk(expr)
    return tuple(out)


def linearize(expr):
    return UOp("LINEAR", toposort(expr))
