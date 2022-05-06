from utils import *
from PIL import Image
import pandas as pd
import tqdm



def test_net(net, testloader, testdata_path, csv_path,device):


    # read csv file
    df = pd.read_csv(csv_path)

    print('testing...')
    for i, (img, label, path) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        img = img.to(device)
        label = label.to(device)
        print(path)
        _, y_pred = net(img).max(1)

        for i in range(0,len(path)):
            filename = str(path[i].split(sep='\\')[-1])
            row_numbers = df[df['file'] == filename].index
            df.loc[row_numbers, 'species'] = y_pred.cpu()[i]

    #save csv file
    df.to_csv(csv_path, index=False)
    print('csv file update done')


