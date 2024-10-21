import os   
import torch
from torchvision import transforms

from PIL import Image

def label(filename):
    if   'finish'  in filename:  return 0
    elif 'iron'  in filename:  return 1
    elif 'plate'  in filename:  return 2
    elif 'temperature'  in filename:  return 3   
    elif 'red'  in filename:  return 4   
    elif 'slag'  in filename:  return 5  
    elif 'surface'  in filename:  return 6

    

class Dataset_X_SDD(torch.utils.data.Dataset):
    def __init__(self, args):
        self.root_dir = '../../../../Data/X_SDD'
        self.args = args
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                            transforms.Resize((224, 224)),  # 调整图像尺寸为 (224, 224)
                                            transforms.ToTensor(),  # 将图像转换为张量
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
        
        folder_of_classes = [os.path.join(self.root_dir, os.listdir(self.root_dir)[i]) for i in range(args.ways)]
        list_of_classes = {i:os.listdir(folder_of_classes[i]) for i in range(args.ways)}
        self.x = torch.cat([  torch.stack([self.transform(Image.open(os.path.join(folder_of_classes[c], list_of_classes[c][i]))) for i in range(len(list_of_classes[c]))]) for c in range(args.ways) ])
        self.y = torch.cat([torch.stack([torch.tensor(c) for i in range(len(list_of_classes[c]))]) for c in range(args.ways) ])
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]