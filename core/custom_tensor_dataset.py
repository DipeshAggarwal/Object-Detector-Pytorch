from torch.utils.data import Dataset

class CustomTensorDataset(Dataset):
    
    def __init__(self, tensors, transforms=None):
        self.tensors = tensors
        self.transforms = transforms
        
    def __getitems__(self, index):
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        bbox = self.tensors[2][index]
        
        # Make the channel dimension the first
        image = image.permute(2, 0, 1)
        
        if self.transforms:
            image = self.transforms(image)
            
        return (image, label, bbox)
    
    def __len__(self):
        return self.tensors[0].size(0)
