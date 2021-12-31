class Regularisation():
    def __init__(self) -> None:
        pass

    def l2(factor = 0.):
        return (lambda W: W * factor)

    def l1(factor = 0.):
        return (lambda W: ((W>0)-0.5)*2*factor)