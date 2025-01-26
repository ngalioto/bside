import torch
from torch import Tensor

class Matrix(torch.nn.Module):

    def __init__(
        self,
        default: Tensor,
        mask: Tensor | None = None,
        indices: Tensor = torch.empty(0)
    ):
        
        super().__init__()

        if mask is not None and default.shape != mask.shape:
            raise ValueError(f'Matrix shape must match `mask` shape, but got shapes {default.shape} and {mask.shape}')
        self.default = default
        self.mask = mask
        self.indices = indices
        self.pdim = torch.unique(indices).numel()

        # initialize parameters
        self.params = torch.nn.Parameter(torch.zeros(self.pdim)) if self.pdim > 0 else None

        self._val = self.default.clone()
        self._inv_val = None
        self._inv_up_to_date = False


    @property
    def val(
        self
    ) -> Tensor:
        
        return self._val

    @val.setter
    def val(
        self,
        value: Tensor
    ) -> None:

        self._val = value

    @property
    def inv(
        self
    ) -> Tensor:
        
        if not self._inv_up_to_date:
            self._inv_val = torch.linalg.inv(self.val)
            self._inv_up_to_date = True
        return self._inv_val
    
    @inv.setter
    def inv(
        self,
        value: Tensor
    ) -> None:
        
        raise ValueError('Setting the inverse of an object `Matrix` is not allowed')
    
    def compute_inv(
        self
    ) -> None:
        
        """
        Compute the inverse of the matrix without storing it.
        """
        
        return torch.linalg.inv(self.val)

    def forward(
        self,
        params: Tensor | None = None
    ):

        p = self.params if params is None else params

        matrix = self.default.clone()
        if p is not None:
            matrix[:, self.mask] = p[self.indices]
        return matrix
    
    def update(
        self
    ) -> None:
        
        self.val = self()
    
    def __repr__(
        self
    ) -> str:
        
        return f'{self.__class__.__name__}({list(self.default.shape)})'
    
class PSDMatrix(Matrix):

    def __init__(
        self,
        default: Tensor,
        mask: Tensor | None = None,
        indices: Tensor = torch.empty(0)
    ) -> None:

        super().__init__(default, mask, indices)

        self._up_to_date = True

        self._sqrt_val = None
        self._sqrt_up_to_date = False

        self._inv_val = None
        self._inv_up_to_date = False

    def update(
        self
    ) -> None:
        
        self.val = self()
        self._up_to_date = True
        self._sqrt_up_to_date = False
        self._inv_up_to_date = False

    @property
    def val(
        self
    ) -> Tensor:
        
        if not self._up_to_date:
            self._val = self.sqrt @ self.sqrt.T
            self._up_to_date = True
        return self._val

    @val.setter
    def val(
        self,
        value: Tensor
    ) -> None:

        self._val = value
        self._up_to_date = True
        self._sqrt_up_to_date = False
        self._inv_up_to_date = False

    @property
    def sqrt(
        self
    ) -> Tensor:
        
        if not self._sqrt_up_to_date:
            self._sqrt_val = torch.linalg.cholesky(self.val, upper=False)
            self._sqrt_up_to_date = True
            
        return self._sqrt_val

    @sqrt.setter
    def sqrt(
        self,
        value: Tensor
    ) -> None:

        self._sqrt_val = value
        self._sqrt_up_to_date = True
        self._up_to_date = False
        self._inv_up_to_date = False

    @property
    def inv(
        self
    ) -> Tensor:
        
        if not self._inv_up_to_date:
            self._inv_val = torch.cholesky_inverse(self.sqrt, upper=False)
            self._inv_up_to_date = True
            
        return self._inv_val
    
    def compute_sqrt(
        self
    ) -> None:
        
        """
        Compute the Cholesky decomposition of the matrix without storing it.
        """
        
        return torch.linalg.cholesky(self.val, upper=False)
    
    def compute_inv(
        self
    ) -> None:
        
        """
        Compute the inverse of the matrix without storing it.
        """
        
        return torch.cholesky_inverse(self.compute_sqrt(), upper=False)
    
    
class ExponentialMatrix(Matrix):

    def forward(
        self,
    ) -> Tensor:

        return torch.linalg.matrix_exp(super().forward())
    
class SquaredMatrix(Matrix):

    def forward(
        self,
    ) -> Tensor:

        p = self.params ** 2
        return super().forward(p)
    

class FeedforwardNetwork(torch.nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int,
        n_layers: int,
        activation: torch.nn.Module = torch.nn.Tanh,
        batch_norm: bool = False,
        dropout: float | None = None
    ) -> None:

        super(FeedforwardNetwork, self).__init__()

        self.in_dim = n_in
        self.out_dim = n_out

        dims = [n_in] + [n_hidden] * (n_layers - 1) + [n_out]
        self.layers = torch.nn.ModuleList()
        for ii in range(n_layers-1):
            self.layers.append(torch.nn.Linear(dims[ii], dims[ii+1]))
            if batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(dims[ii+1]))
            self.layers.append(activation())
            if dropout is not None:
                self.layers.append(torch.nn.Dropout(dropout))
        self.layers.append(torch.nn.Linear(dims[-2], dims[-1]))

    def forward(
        self,
        x: Tensor,
        u: Tensor | None = None
    ) -> Tensor:

        x = torch.cat((x,u), dim=1) if u is not None else x.clone()
        for layer in self.layers:
            x = layer(x)
        return x
    

class ResidualNetwork(torch.nn.Module):

    def __init__(
        self, 
        n_in: int,
        n_out: int, 
        n_hidden: int, 
        n_layers: int, 
        activation: torch.nn.Module = torch.nn.Tanh,
        batch_norm: bool = False,
        dropout: float | None = None
    ) -> None:
        
        super(ResidualNetwork, self).__init__()

        self.in_dim = n_in
        self.out_dim = n_out

        self.residual = torch.nn.Linear(n_in, n_out)

        self.feedforward = FeedforwardNetwork(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def forward(
        self,
        x: Tensor,
        u: Tensor | None = None
    ) -> Tensor:

        x = torch.cat((x,u), dim=1) if u is not None else x.clone()

        return self.residual(x) + self.feedforward(x)