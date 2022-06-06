from .pdf import PDF

class Prior(PDF):
    """
    This class represents a prior r(w|s).
    """
    def __init__(self):
        super().__init__()

class Normal(Prior):
    def __init__(self, mean, variance):
        """_summary_

        Args:
            mean (_type_): _description_
            variance (_type_): _description_
        """
        super().__init__()
        self.mean = mean
        self.variance = variance
