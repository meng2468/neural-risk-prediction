from dataloader import EICUDataSet
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    data = EICUDataSet('eicu_prepared_x.csv','eicu_prepared_y.csv')
    loader = DataLoader(data, batch_size=30, shuffle=True)

    print(data[0])

