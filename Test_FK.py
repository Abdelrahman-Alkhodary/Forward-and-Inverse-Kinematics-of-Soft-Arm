import torch
import torch.nn as nn

# the class that holds the forward kinematics architecture
class FK_Net(nn.Module):
    def __init__(self):
        super(FK_Net, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(4, 16),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(16,16),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(16,3))
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

FK_model = FK_Net()
path_FK = 'FCN_FK_DrBerkeCode.pt'
FK_model = torch.load(path_FK)
FK_model.eval()

print('Please insert the tensions one by one')
t1 = float(input('insert t1: '))
t2 = float(input('insert t2: '))
t3 = float(input('insert t3: '))
t4 = float(input('insert t4: '))
input_to_model = torch.tensor([[t1, t2, t3, t4]], dtype=torch.float32).cuda()
output = FK_model(input_to_model).detach()
print(output)