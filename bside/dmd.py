import torch
from torch import Tensor

from typing import Union
from bside.dataset import Data, DataTrajectories

class DMD:

    """
    Class for training a DMD model.

    For more details on DMD, see
    https://epubs.siam.org/doi/abs/10.1137/15M1013857

    Attributes
    ----------
    A : Tensor
        The A matrix.
    A_rom : Tensor
        The A matrix for the reduced-order model.
    data : Dataset
        The dataset used for training the DMD model.
    rank : int
        The rank of the DMD model.
    """

    def __init__(
        self
    ):
        
        """
        Initialize the DMD model.

        Parameters
        ----------
        data : Dataset
            The dataset to be used for training the DMD model.
        rank : int, optional
            The rank of the DMD model, by default None. Uses maximum rank if None.
        """     

        self._A = None
        self._A_rom = None

    @property
    def A(
        self
    ) -> Tensor:
        
        if self._A is None:
            raise RuntimeError("The A matrix has not been computed yet. " +
                               "Please first call the method fit().")
        return self._A
    
    @A.setter
    def A(
        self, 
        value : Tensor
    ) -> Tensor:
        
        self._A = value

    @property
    def A_rom(
        self
    ) -> Tensor:
        
        if self._A_rom is None:
            raise RuntimeError("The A_rom matrix has not been computed yet. " + 
                               "Please first call the method fit(rom=True).")
        return self._A_rom
    
    @A.setter
    def A_rom(
        self, 
        value : Tensor
    ) -> Tensor:
        
        self._A_rom = value

    def form_snapshot_matrices(
            self, 
            data : Union[Data, DataTrajectories]
        ):

        """
        Form the snapshot matrices for the DMD model.
        
        Returns
        -------
        Tensor
            The input matrix.
        Tensor
            The output matrix.
        """

        # Form the snapshot matrices
        if isinstance(data, Data): # single trajectory
            input = data.y[:-1].T
            output = data.y[1:].T
        elif isinstance(data, DataTrajectories): # multiple trajectories
            input_indices = torch.cat([torch.arange(data.start_indices[ii], data.end_indices[ii]-1) for ii in range(data.num_traj)])
            output_indices = torch.cat([torch.arange(data.start_indices[ii]+1, data.end_indices[ii]) for ii in range(data.num_traj)])
            input = data.y[input_indices].T
            output = data.y[output_indices].T
        return input, output
    

    def fit(
        self, 
        data : Union[Data, DataTrajectories],
        rank : int = None,
        rom : bool = False
    ):
        
        """
        Fit the DMD model to the data.
        
        Parameters
        ----------
        rom : bool, optional
            Whether to compute the reduced-order model, by default False.
        """

        if not isinstance(data, Data) and not isinstance(data, DataTrajectories):
            raise ValueError(f"The data must be an instance of the Data class or DataTrajectories class. Instead received {data.__class__.__name__}.")
        rank = data.ydim if rank is None else rank
        
        input, output = self.form_snapshot_matrices(data)

        # Compute the SVD of the input matrix
        U, S, V = torch.linalg.svd(input, full_matrices=False)
        U = U[:, :rank]
        S = S[:rank]
        V = V[:rank, :]

        # Approximate the A matrix
        if rom:
            self.A_rom = U.T @ output @ (V.T / S)
        else:
            self.A = output @ (V.T / S) @ U.T

    def test(
        self,
        data : Union[Data, DataTrajectories]
    ) -> Tensor:
        
        if not isinstance(data, Data) and not isinstance(data, DataTrajectories):
            raise ValueError("The data must be an instance of the Data class or DataTrajectories class.")
        
        input, output = self.form_snapshot_matrices(data)

        return torch.mean((output - self.A @ input)**2)

class DMDc(DMD):

    """
    Class for training a DMDc model.

    For more details on DMDc, see
    https://epubs.siam.org/doi/abs/10.1137/15M1013857

    Attributes
    ----------
    A : Tensor
        The A matrix.
    A_rom : Tensor
        The A matrix for the reduced-order model.
    B : Tensor
        The control input matrix.
    B_rom : Tensor
        The control input matrix for the reduced-order model.
    data : Dataset
        The dataset used for training the DMDc model.
    rank : int
        The rank of the DMDc model.
    """

    def __init__(
        self
    ):
        
        """
        Initialize the DMDc model.
        
        Parameters
        ----------
        data : Dataset
            The dataset to be used for training the DMDc model.
        rank : int, optional
            The rank of the DMDc model, by default None. Uses maximum rank if None.
        """
        
        super().__init__()

        self._B = None
        self._B_rom = None

    @property
    def B(
        self
    ) -> Tensor:
        
        if self._B is None:
            raise RuntimeError("The A matrix has not been computed yet. " +
                               "Please first call the method fit().")
        return self._B
    
    @B.setter
    def B(
        self, 
        value : Tensor
    ) -> Tensor:
        
        self._B = value

    @property
    def B_rom(
        self
    ) -> Tensor:
        
        if self._B_rom is None:
            raise RuntimeError("The B_rom matrix has not been computed yet. " +
                               "Please first call the method fit(rom=True).")
        return self._B_rom
    
    @B.setter
    def B_rom(
        self, 
        value : Tensor
    ) -> Tensor:
        
        self._B_rom = value

    def form_snapshot_matrices(
            self,
            data : Union[Data, DataTrajectories]
        ):

        """
        Form the snapshot matrices for the DMDc model.

        Returns
        -------
        Tensor
            The input matrix.
        Tensor
            The output matrix.
        """

        if isinstance(data, Data):
            input = torch.cat((data.y[:-1], data.u[:-1]), dim=1).T
            output = data.y[1:,:].T
        elif isinstance(data, DataTrajectories):
            input_indices = torch.cat([torch.arange(data.start_indices[ii], data.end_indices[ii]-1) for ii in range(data.num_traj)])
            output_indices = torch.cat([torch.arange(data.start_indices[ii]+1, data.end_indices[ii]) for ii in range(data.num_traj)])
            input = torch.cat((data.y[input_indices], data.u[input_indices]), dim=1).T # this is different part
            output = data.y[output_indices].T
        return input, output
    

    def fit(
        self, 
        data : Union[Data, DataTrajectories],
        rank : int = None,
        rom : bool = False
    ) -> Tensor:
        
        """
        Fit the DMDc model to the data.
        
        Parameters
        ----------
        rom : bool, optional
            Whether to compute the reduced-order model, by default False.
        """

        if not isinstance(data, Data) and not isinstance(data, DataTrajectories):
            raise ValueError(f"The data must be an instance of the Data class or DataTrajectories class. Instead received {data.__class__.__name__}.")
        rank = data.ydim if rank is None else rank
        
        input, output = self.form_snapshot_matrices(data)

        # Compute the SVD of the input matrix
        U, S, V = torch.linalg.svd(input, full_matrices=False)
        U1 = U[:data.ydim, :rank]
        U2 = U[data.ydim:, :rank]
        S = S[:rank]
        V = V[:rank, :]

        # Approximate the A and B matrices
        if rom:
            Uhat, _, _ = torch.linalg.svd(output)
            Uhat = Uhat[:, :rank]
            # self.A_rom = Uhat.T @ output @ (V.T / S) @ U1.T @ Uhat
            # self.B_rom = Uhat.T @ output @ (V.T / S) @ U2.T
        else:
            self.A = output @ (V.T / S) @ U1.T
            self.B = output @ (V.T / S) @ U2.T
            return torch.mean((output - self.A @ input[:data.ydim] - self.B @ input[data.ydim:])**2)

    def test(
        self,
        data : Union[Data, DataTrajectories]
    ) -> Tensor:
        
        if not isinstance(data, Data) and not isinstance(data, DataTrajectories):
            raise ValueError("The data must be an instance of the Data class or DataTrajectories class.")
        
        input, output = self.form_snapshot_matrices(data)

        return torch.mean((output - self.A @ input[:data.ydim] - self.B @ input[data.ydim:])**2)
    