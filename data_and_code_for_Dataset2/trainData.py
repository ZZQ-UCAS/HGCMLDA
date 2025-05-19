from __future__ import division


class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index):
        return (self.data_set['ID'], self.data_set['IL'],
                self.data_set['ld'][index]['train'], self.data_set['ld'][index]['test'],
                self.data_set['ld_p'], self.data_set['ld_true'],self.data_set['independent'][0]['test'],
                self.data_set['ld'][index]['train_adj'])

    def __len__(self):
        return self.nums



