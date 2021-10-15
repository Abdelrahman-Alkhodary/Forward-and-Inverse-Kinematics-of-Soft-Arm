# Forward-and-Inverse-Kinematics-of-Soft-Arm
Forward and Inverse Kinematics Models of soft arm that was modeled in the matlab, you can find the data that was used to train the model in the following link
https://bahcesehir-my.sharepoint.com/:f:/g/personal/abdelrahman_alkhoda_bahcesehir_edu_tr/Em0T23QeGTJGjw3gHQslxe8BjFL9TZxXsrK_tCJYtPICvA?e=5eJZif

The file FCN_FK_IK.ipynb is a jupyter notebook that is used to train the forward and inverse kinematics models.

The files FCN_FK_DrBerkeCode.pt and FCN_IK_DrBerkeCode.pt are the neural networks weights of the pytorch models on the forward and inverse models

You can use the test files to test the models, but they have to be run from inside a python environment that has pytorch installed within.

The forward model is a fully connected neural network with four inputs, one hidden layer of 16 neuron and three output. The input to the model is the four cables' actuations, and the output is the position of the tip of the arm.

The inverse model is also a fully connected neural network with three inputs, one hidden layer of 16 neuron and four output. The input to the model is the position of the tip of the arm, and the output is the four cables' actuations.

