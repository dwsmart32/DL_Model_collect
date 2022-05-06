from utils import *
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms



def test_net(net, testdata_path, csv_path):
    trans = transforms.Compose([transforms.ToTensor()])

    # read csv file
    df = pd.read_csv(csv_path)

    for img_name in os.listdir(testdata_path):
        testset_Image = Image.open(testdata_path+img_name)
        _, y_pred = net(trans(testset_Image)).max(1)
        row_numbers = df[df['file'] == img_name].index
        df.loc[row_numbers, 'species'] = y_pred

    #save csv file
    df.to_csv(csv_path, index=False)
    print('csv file update done')


