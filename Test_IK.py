import torch
import torch.nn as nn

# the class that holds the inverse kinematics architecture
class IK_Net(nn.Module):
    def __init__(self):
        super(IK_Net, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(3, 16),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(16,16),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(16,4))
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

IK_model = IK_Net()
path_IK = 'FCN_IK_DrBerkeCode.pt'
IK_model = torch.load(path_IK)
IK_model.eval()

print('Please insert the coordinates one by one')
x = float(input('insert X: '))
y = float(input('insert Y: '))
z = float(input('insert Z: '))
input_to_model = torch.tensor([[x, y, z]], dtype=torch.float32).cuda()
output = IK_model(input_to_model).detach()
print(output)