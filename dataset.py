import numpy as np
import copy
import torch
from torch import Tensor
from typing import Optional, Tuple, Union, Type, Self, Iterator, List

from numpy import ndarray

class Data:

    """
    A class for storing data in the form of a PyTorch tensor.
    
    Attributes
    ----------
    y : Tensor
        The state data.
    u : Optional[Tensor]
        The control input data. None if no control input is available.
    ydim : int
        The state dimension of the data.
    udim : int
        The control input dimension of the data. 0 if no control input is available.
    size : int
        The number of samples in the data.

    Methods
    -------
    __iter__()
        Return an iterator over the data.
    __len__()
        Return the length of the y tensor.
    __getitem__(index)
        Allow indexing into the y tensor and the u tensor if available.
    __repr__()
        Return a string representation of the Data object.
    """

    def __init__(
        self,
        y: Union[Tensor, ndarray] = None, 
        u: Optional[Union[Tensor, ndarray]] = None
    ):        
        
        """
        Parameters
        ----------
        y : Union[Tensor, ndarray], optional
            The state data, by default None
        u : Optional[Union[Tensor, ndarray]], optional
            The control input data, by default None
        
        Raises
        ------
        TypeError
            If y or u are not PyTorch tensors or numpy arrays.
        ValueError
            If the control input data is provided without the state data.    
        """

        if y is not None:
            if not isinstance(y, torch.Tensor) and not isinstance(y, ndarray):
                raise TypeError("y must be a PyTorch tensor or a numpy array.")
            self.y = torch.tensor(y, dtype=torch.float32) if isinstance(y, ndarray) else y

            if self.y.ndim == 1:
                self.y = self.y[...,None]
            
            self.ydim = self.y.shape[1]
            self.size = self.y.shape[0]

            if u is not None:
                if not isinstance(y, torch.Tensor) and not isinstance(y, ndarray):
                    raise TypeError("y must be a PyTorch tensor or a numpy array.")
                
                self.u = torch.tensor(u, dtype=torch.float32) if isinstance(u, ndarray) else u

                if self.u.ndim == 1:
                    self.u = self.u[...,None]
                self.udim = self.u.shape[1]
                if self.u.shape[0] != self.size:
                    raise ValueError("There must be exactly one fewer " +
                                        "control input sample than state sample.")
            else:
                self.u = None
                self.udim = 0
        elif u is not None:
            raise ValueError("The control input cannot be provided without the state data.")
        else:
            self.y = None
            self.u = None
            self.ydim = 0
            self.udim = 0
            self.size = 0

    def __iter__(
        self
    ) -> Union[Iterator[Tensor], Iterator[Tuple[Tensor, Tensor]]]:

        """
        Return an iterator over a combined tuple of (y, u) if u is available,

        Returns
        -------
        Union[Iterator[Tensor], Iterator[Tuple[Tensor, Tensor]]]
            An iterator over the data.
        """

        if self.y is None:
            return iter([])
        elif self.u is None:
            return iter(self.y)
        else:
            return iter(zip(self.y, self.u))

    def __len__(
        self
    ) -> int:
        
        """
        Return the length of the y tensor.

        Returns
        -------
        int
            The number of samples in the data.
        """

        return self.size

    def __getitem__(
            self, 
            index : Union[int, slice]
    ) -> Self:

        """
        Allow indexing into the y tensor and the u tensor if available.
        
        Parameters
        ----------
        index : Union[int, slice]
            The index or slice to be used for indexing.
        
        Returns
        -------
        Self
            The indexed data.

        Raises
        ------
        ValueError
            If the Data object is empty.
        """

        if self.y is None:
            raise ValueError("Cannot index an empty Data object.")
        elif self.u is None:
            return Data(self.y[index])
        else:
            return Data(self.y[index], self.u[index])
            
    def __repr__(
        self
    ) -> str:
        
        """
        Return a string representation of the Data object.

        Returns
        -------
        str
            A string representation of the Data object.
        """

        return f"Data(y={self.y},\n u={self.u})"

class DataTrajectories:

    """
    A class for storing trajectory data in the form of a PyTorch tensor.

    Attributes
    ----------
    y : Tensor
        The state data.
    u : Tensor
        The control input data.
    ydim : int
        The state dimension of the data.
    udim : int
        The control input dimension of the data.
    num_traj : int
        The number of trajectories in the dataset.
    traj_lengths : Tensor
        The lengths of the trajectories in the dataset.
    max_length : int
        The length of the longest trajectory in the dataset.
    min_length : int
        The length of the shortest trajectory in the dataset.
    start_indices : Tensor
        The start indices of the trajectories in the dataset.
    end_indices : Tensor
        The end indices of the trajectories in the dataset.
    index : int
        The current index of the trajectory iterator.

    Methods
    -------
    _init_from_filename(filename)
        Initialize the dataset from a file.
    _init_from_batch(batch)
        Initialize the dataset from a batch of Data objects.
    _build_slices_from_indices(traj_indices)
        Build slices from trajectory indices.
    partition_trajectories(T, history_length=1)
        Partition the trajectories into shorter trajectories to allow for batching.
    __iter__()
        Return an iterator over the dataset.
    __next__()
        Return the next trajectory in the dataset.
    __getitem__(index)
        Allow indexing into the dataset.
    __len__()
        Return the number of trajectories in the dataset.
    """

    def __init__(
        self,
        filename : str = None,
        batch : List[Data] = None
    ):
        
        """
        Parameters
        ----------
        filename : str, optional
            The filename of the dataset, by default None
        batch : List[Data], optional
            A list of Data objects, by default None

        Raises
        ------
        ValueError
            If both a filename and a batch are provided.
            If neither a filename nor a batch are provided.
        """

        if filename is not None and batch is not None:
            raise ValueError("Either a filename or a batch must be provided, but not both.")
        elif filename is None and batch is None:
            raise ValueError("Either a filename or a batch must be provided.")
        
        if filename is not None:
            self._init_from_filename(filename)
        elif batch is not None:
            self._init_from_batch(batch)

        self.norm = False
        self.y_mean = torch.tensor(0.)
        self.y_std = torch.tensor(1.)
        self.u_mean = torch.tensor(0.)
        self.u_std = torch.tensor(1.)

        self.index = 0


    def _init_from_filename(
        self,
        filename : str
    ): 
        
        """
        Initialize the dataset from a file.

        Parameters
        ----------
        filename : str
            The filename of the dataset.
        """
        
        data = np.load(filename)
        self.y = torch.tensor(data['cores'], dtype=torch.float32)
        self.u = torch.tensor(data['labels'], dtype=torch.float32)

        if self.u.ndim == 1:
            self.u = self.u[...,None]

        self.ydim = self.y.shape[1]
        self.udim = self.u.shape[1]

        # Extract frame numbers from paths
        frame_numbers = torch.tensor([int(ii.split("/")[-1].split(".")[0]) for ii in data["paths"]])

        self.start_indices = torch.where(frame_numbers == 0)[0]
        self.end_indices = torch.cat((self.start_indices[1:], torch.tensor([self.y.shape[0]])))

        self.num_traj = len(self.start_indices)
        self.traj_lengths = self.end_indices - self.start_indices

    def _init_from_batch(
        self,
        batch : List[Data]
    ):
        
        """
        Initialize the dataset from a list of Data objects.

        Parameters
        ----------
        batch : List[Data]
            A list of Data objects.
        """
        
        self.y = torch.cat([data.y for data in batch], dim=0)
        self.u = torch.cat([data.u for data in batch], dim=0)

        self.ydim = self.y.shape[1]
        self.udim = self.u.shape[1]

        self.num_traj = len(batch)
        self.traj_lengths = torch.tensor([data.size for data in batch])

        self.end_indices = torch.cumsum(self.traj_lengths, dim=0)
        self.start_indices = torch.cat((torch.zeros(1,dtype=int), self.end_indices[:-1]))
    
    def _build_slices_from_indices(
        self,
        traj_indices : Tensor
    ) -> List[slice]:
        
        """
        Build slices from trajectory indices.

        Parameters
        ----------
        traj_indices : Tensor
            The trajectory indices.

        Returns
        -------
        List[slice]
            A list of slices with starts and stops corresponding to the traj_indices.
        """

        return [slice(traj_indices[i], traj_indices[i+1]) for i in range(len(traj_indices)-1)]
    
    def normalize(
        self
    ):

        """
        Normalizes the data. If the data is already normalized, this function does nothing.
        """  

        if self.norm == False:
            self.y_mean = self.y.mean(dim=0)
            self.y_std = self.y.std(dim=0)
            self.u_mean = self.u.mean(dim=0)
            self.u_std = self.u.std(dim=0)
            self.y = (self.y - self.y_mean) / self.y_std
            self.u = (self.u - self.u_mean) / self.u_std
            self.norm = True

    def unnormalize(
        self
    ):
        
        """
        Unnormalizes the data. If the data is not normalized, this function does nothing.
        """
        
        if self.norm == True:
            self.y = self.y * self.y_std + self.y_mean
            self.u = self.u * self.u_std + self.u_mean
            self.norm = False

    def partition_trajectories(
        self,
        T : int,
        history_length : int = 1
    ) -> Self:
        
        """
        Partition the trajectories into shorter trajectories to allow for batching.
        
        Parameters
        ----------
        T : int
            The length of the new trajectories.
        history_length : int, optional
            The length of the history to be included in each new trajectory, by default 1
        
        Returns
        -------
        Type[Self]
            The partitioned dataset.

        Raises
        ------
        ValueError
            If the history length is greater than or equal to the minimum trajectory length.
        """

        if history_length >= self.min_length:
            raise ValueError(f"History length {history_length} is greater than or equal to the minimum trajectory length {self.min_length}.")
        
        new_data = copy.copy(self)

        # No changes are needed if the trajectories are already shorter than T + history_length
        if self.max_length > T + history_length:
            new_data.start_indices = torch.cat([
                torch.arange(
                    self.start_indices[ii], 
                    self.end_indices[ii]-history_length, 
                    T
                ) for ii in range(self.num_traj)
            ])

            new_data.end_indices = torch.cat([
                torch.cat((
                    torch.arange(
                        min(self.start_indices[ii]+history_length+T, self.end_indices[ii]),
                        self.end_indices[ii],
                        T
                        ), 
                    torch.tensor([self.end_indices[ii]])
                )) for ii in range(self.num_traj)
            ])
            
            new_data.num_traj = len(new_data.start_indices)
            new_data.traj_lengths = new_data.end_indices - new_data.start_indices

        return new_data


    def __iter__(
        self
    ) -> Self:
        
        """
        Return an iterator over the dataset.

        Returns
        -------
        Self
            An iterator over the dataset.
        """
                
        return self
    
    def __next__(
        self
    ) -> Data:
        
        """
        Return the next trajectory in the dataset.

        Returns
        -------
        Data
            The next trajectory in the dataset.

        Raises
        ------
        StopIteration
            If the end of the dataset is reached.
        """
        
        if self.index < self.num_traj:
            result = self[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration
    
    def __getitem__(
        self,
        index : Union[int, slice]
    ) -> Data:
        
        """
        Allow indexing into the dataset.

        Parameters
        ----------
        index : Union[int, slice]
            The index or slice to be used for indexing.

        Returns
        -------
        Data
            The indexed data.

        Raises
        ------
        TypeError
            If the index is not an integer or a slice.
        """
        
        if isinstance(index, int):
            y = self.y[self.start_indices[index]:self.end_indices[index]]
            u = self.u[self.start_indices[index]:self.end_indices[index]]
            return Data(y, u)
        elif isinstance(index, slice):
            # convert the trajectory slice into a range of trajectory indices
            traj_indices = range(*index.indices(self.num_traj))
            # convert trajectory indices into data indices
            data_indices = torch.cat([torch.arange(
                self.start_indices[idx], 
                self.end_indices[idx]
                ) for idx in traj_indices])
            y = self.y[data_indices]
            u = self.u[data_indices]
            return Data(y, u)
        else:
            raise TypeError("Invalid argument type.")        

    def __len__(
        self
    ) -> int:
        
        """
        Return the number of trajectories in the dataset.
        
        Returns
        -------
        int
            The number of trajectories in the dataset.
        """
        
        return self.num_traj
    
    @property
    def traj_lengths(
        self
    ) -> Tensor:
        
        """
        Return the lengths of the trajectories in the dataset.
        
        Returns
        -------
        Tensor
            The lengths of the trajectories in the dataset.
        """
        
        return self._traj_lengths
    
    @traj_lengths.setter
    def traj_lengths(
        self,
        value : Tensor
    ):
        
        """
        Set the lengths of the trajectories in the dataset and updates the max and min lengths variables.

        Parameters
        ----------
        value : Tensor
            The lengths of the trajectories in the dataset.
        """
        
        self._traj_lengths = value
        self._max_length = max(value)
        self._min_length = min(value)
    
    @property
    def max_length(
        self
    ) -> int:
        
        """
        Return the length of the longest trajectory in the dataset.
        
        Returns
        -------
        int
            The length of the longest trajectory in the dataset.
        """
        
        return self._max_length
    
    @max_length.setter
    def max_length(
        self,
        value : int
    ):
        
        """
        Settter for max_length.
        
        Parameters
        ----------
        value : int
            The maximum trajectory length.
            
        Raises
        ------
        AttributeError
            The maximum trajectory length depends on the data and cannot be set.
        """
        
        raise AttributeError("The maximum trajectory length cannot be set.")
    
    @property
    def min_length(
        self
    ) -> int:
        
        """
        Return the length of the shortest trajectory in the dataset.
        
        Returns
        -------
        int
            The length of the shortest trajectory in the dataset.
        """
        
        return self._min_length
    
    @min_length.setter
    def min_length(
        self,
        value : int
    ):
        
        """
        Setter for min_length.

        Parameters
        ----------
        value : int
            The minimum trajectory length.

        Raises
        ------
        AttributeError
            The minimum trajectory length depends on the data and cannot be set.
        """
        
        raise AttributeError("The minimum trajectory length cannot be set.")
    