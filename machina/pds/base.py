class BasePd(object):
    """
    Base class of probablistic distribution
    """

    def sample(self, params, sample_shape):
        """
        sampling
        """
        raise NotImplementedError

    def llh(self, x, params):
        """
        log liklihood
        """
        raise NotImplementedError

    def kl_pq(self, p_params, q_params):
        """
        KL divergence between p and q
        """
        raise NotImplementedError

    def ent(self, params):
        """
        entropy
        """
        raise NotImplementedError
