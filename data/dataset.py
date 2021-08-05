
import torch
import PIL.Image, PIL.ImageStat
from torchvision import transforms as kit

class dataset(torch.utils.data.Dataset):

    def __init__(self, table):

        self.table = table
        pass

    def __len__(self):

        length = len(self.table)
        return(length)

    def __getitem__(self, index):

        item  = self.table.iloc[index, :]
        picture = process(item).image()
        case = picture
        return(case)

    def get(self, index):

        return self.__getitem__(index)

    pass

class process:

    def __init__(self, item):

        self.item = item
        pass

    def image(self):

        picture  = PIL.Image.open(self.item['link']).convert("RGB")
        size     = (64, 64)
        if(self.item['mode']=='train'):

            blueprint = [
                kit.Resize(size),
                kit.ToTensor()
            ]
            convert = kit.Compose(blueprint)
            picture = convert(picture).type(torch.float)
            return(picture)
            
        if(self.item['mode']=='test'):

            blueprint = [
                kit.Resize(size),
                kit.ToTensor()
            ]
            convert = kit.Compose(blueprint)
            picture = convert(picture).type(torch.float)
            return(picture)

        pass

    pass

