import torch
from functools import wraps
from torch.utils.data import DataLoader, RandomSampler, Subset


def incremental_forward(batch_size=2):

    def decorator(func):
        @wraps(func)
        def wrapper(self, x):

            output = None
            for i in range(0, x.shape[0]-2*batch_size, batch_size):
                temp_output = func(self, x[i:i + batch_size])
                if output == None:
                    output = temp_output
                else:
                    output = torch.cat((output, temp_output), dim=0)
                torch.cuda.empty_cache()
            if output == None:
                return func(self, x)
            else:
                final_batch_size = x.shape[0]-output.shape[0]
                output = torch.cat((output, func(self, x[-final_batch_size:])), dim=0)
                return output
        return wrapper
    return decorator


def random_subset(data_set, subset_size, fast=True):
    ind = torch.randperm(len(data_set))[:subset_size] if not fast else torch.randint(0, len(data_set)-1, (subset_size,))
    return Subset(data_set, ind)
