from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler, SequentialSampler
import math

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, weights, run_model):
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        if not run_model:
            sampler = WeightedRandomSampler(weights, batch_size * math.ceil(self.nbr_examples / batch_size), True)
        else:
            sampler = SequentialSampler(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True
        }

        # Shuffle is mutually exclusive with sampler
        if sampler is not None:
            del self.init_kwargs['shuffle']

        super(BaseDataLoader, self).__init__(sampler=sampler,
                                             **self.init_kwargs)
