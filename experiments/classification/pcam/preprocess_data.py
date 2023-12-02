import torch
import h5py

def preprocess(filepath, attr, save, labels=False):
    hf = h5py.File(filepath, 'r')
    data = torch.tensor(hf.get(attr).value).permute(0, 3, 1, 2)
    if labels:
        data = data.long()
    else:
        data = data.float()
        data.divide_(127.5)
        data.sub_(1.0)
    torch.save(data, save)
    del data

print('Starting on training data...')
preprocess('./camelyonpatch_level_2_split_train_y.h5', 'y', './PCAM_train_labels.pt', labels=True)
preprocess('./camelyonpatch_level_2_split_train_x.h5', 'x', './PCAM_train_data.pt')

print('Starting on validation data...')
preprocess('./camelyonpatch_level_2_split_valid_y.h5', 'y', './PCAM_valid_labels.pt', labels=True)
preprocess('./camelyonpatch_level_2_split_valid_x.h5', 'x', './PCAM_valid_data.pt')

print('Starting on test data...')
preprocess('./camelyonpatch_level_2_split_test_y.h5', 'y', './PCAM_test_labels.pt', labels=True)
preprocess('./camelyonpatch_level_2_split_test_x.h5', 'x', './PCAM_test_data.pt')