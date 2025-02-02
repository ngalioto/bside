from bside.ssm import SSM
from bside.models import ResidualNetwork

class SubspaceEncoder(SSM):

    """
    Implements a deep subspace encoder from the paper:
      Gerben Beintema, Roland Toth, Maarten Schoukens. 
      Nonlinear State-Space Identification using Deep Encoder Networks; 
      Proceedings of the 3rd Conference on Learning for Dynamics and Control, 
      PMLR 144:241-250, 2021.
    
    The paper can be found at:
      https://www.sciencedirect.com/science/article/pii/S0005109823003710

    The SubspaceEncoder class inherits from the SSM class.

    Attributes
    ----------
    xdim : int
        The dimension of the latent space.
    ydim : int
        The dimension of the output space.
    dynamics : torch.nn.Module
        The dynamics model.
    observations : torch.nn.Module
        The observation model.
    encoder : torch.nn.Module
        The encoder model.
    num_y_hist : int
        The number of past observations to be fed into the encoder.
    num_u_hist : int
        The number of past inputs to be fed into the encoder

    Methods
    -------
    fit(data, T=1)
        Fits the subspace encoder to the data.
    predict(x, u)
        Predicts the next state given the current state and input.
    """

    def __init__(
        self,
        xdim : int = 10, 
        ydim : int = 10,
        udim : int = 1,
        num_y_hist : int = 20, 
        num_u_hist : int = 20, 
        feedthrough : bool = False
    ):
        
        """
        Parameters
        ----------
        data : Dataset
            The dataset to be used for training the subspace encoder.
        nx : int, optional
            The dimension of the latent space, by default 10
        na : int, optional
            The number of autoregressive terms to be fed into the encoder, by default 20
        nb : int, optional
            The number of exogenous terms to be fed into the encoder, by default 20
        feedthrough : bool, optional
            Whether the system has a direct feedthrough term, by default False
        """

        encoder = ResidualNetwork(
            n_in = ydim * num_y_hist + udim * num_u_hist,
            n_out = xdim, 
            n_hidden = 64, 
            n_layers = 2
        )

        dynamics = ResidualNetwork(
            n_in = xdim + udim,
            n_out = xdim, 
            n_hidden = 64, 
            n_layers = 2
        )

        observations = ResidualNetwork(
            n_in = xdim + udim if feedthrough else xdim,
            n_out = ydim, 
            n_hidden = 64, 
            n_layers = 2
        )

        super().__init__(xdim, ydim, dynamics, observations, encoder, num_y_hist, num_u_hist)