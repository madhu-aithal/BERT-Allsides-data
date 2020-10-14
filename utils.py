import json_lines
import pickle
import torch

def read_samples(file):
    data = pickle.load(open(file, 'rb'))
    return data

def read_and_sample(file, dataset_amount="full", n_samples = None):
    data = read_samples(file)
    # left = list(filter(lambda x:x[1]==0, data))
    # center = list(filter(lambda x:x[1]==1, data))
    # right = list(filter(lambda x:x[1]==2, data))
    # if dataset_amount != "full":
    #     size = min(len(left), len(center), len(right))
    #     return left[:size]+right[:size]+center[:size]
    # else:
    #     return left+right+center
    
    # if n_samples == None:
    #     size = min(len(left), len(center), len(right))
    # else:
    #     size = n_samples
    # return left[:size]+right[:size]+center[:size]

def read_pairwise(file, first = 0, second = 2, dataset_amount="full"):
    data = read_samples(file)
    sample1 = list(filter(lambda x:x[1]==first, data))
    sample2 = list(filter(lambda x:x[1]==second, data))
    sample1 = list(map(lambda x:[x[0], 0], sample1))
    sample2 = list(map(lambda x:[x[0], 1], sample2))
    if dataset_amount != "full":
        size = min(len(sample1), len(sample2))
        return sample1[:size]+sample2[:size]
    else:
        return sample1+sample2
    # if n_samples == None:
    #     size = min(len(left), len(center), len(right))
    # else:
    #     size = n_samples
    # return left[:size]+right[:size]+center[:size]
    

def get_filename(time: int, util_name:str =""):   
    filename = str(time.strftime('%b-%d-%Y_%H-%M-%S'))
    if util_name != "":
        filename = util_name+"_"+filename
    return filename

# If there's a GPU available...
def get_device(device_no: int):
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda:"+str(device_no))

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(device_no))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    return device