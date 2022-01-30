from torch.nn import Sequential
from torch.nn import Identity
from torch.nn import Dropout
from torch.nn import Sigmoid
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU

class ObjectDetector(Module):
    
    def __init__(self, base_model, num_classes):
        super(ObjectDetector, self).__init__()
        
        self.base_model = base_model
        self.num_classes = num_classes
        
        self.regressor = Sequential(
            Linear(base_model.fc.in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid()
        )
        
        self.classifier = Sequential(
            Linear(base_model.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.num_classes)
        )
        
        self.base_model.fc = Identity()
        
    def forward(self, x):
        # Pass the input through base model
        features = self.base_model(x)
        
        # Get the predictions from both branches
        bboxes = self.regressor(features)
        class_logits = self.classifier(features)
        
        return (bboxes, class_logits)
