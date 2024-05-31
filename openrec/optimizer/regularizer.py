import paddle


class L1Decay(object):
    """
    L1 Weight Decay Regularization, which encourages the weights to be sparse.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L1Decay, self).__init__()
        self.coeff = factor

    def __call__(self):
        reg = paddle.regularizer.L1Decay(self.coeff)
        return reg


class L2Decay(object):
    """
    L2 Weight Decay Regularization, which helps to prevent the model over-fitting.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L2Decay, self).__init__()
        self.coeff = float(factor)

    def __call__(self):
        return self.coeff