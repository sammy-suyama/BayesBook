class PDF:
    """
    This class represents any probability density function,such as p(x|w), q(x|w0), r(w|s).
    """
    def __init__(self):
        self._params = set()
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
    
    def __call__(self, *input):
        return self.forward(*input)
    
    def forward(self, *input):
        raise NotImplementedError()
