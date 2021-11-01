# Forward-and-Inverse-Kinematics-of-Soft-Arm
Forward and Inverse Kinematics Models of soft arm that was modeled in the matlab, you can find the data that was used to train the model in the following link
https://bahcesehir-my.sharepoint.com/:f:/g/personal/abdelrahman_alkhoda_bahcesehir_edu_tr/Em0T23QeGTJGjw3gHQslxe8BjFL9TZxXsrK_tCJYtPICvA?e=5eJZif

The file FCN_FK_IK.ipynb is a jupyter notebook that is used to train the forward and inverse kinematics models.

The files FCN_FK_DrBerkeCode.pt and FCN_IK_DrBerkeCode.pt are the neural networks weights of the pytorch models on the forward and inverse models

You can use the test files to test the models, but they have to be run from inside a python environment that has pytorch installed within.

The forward model is a fully connected neural network with four inputs, one hidden layer of 16 neuron and three output. The input to the model is the four cables' actuations, and the output is the position of the tip of the arm. The mean absolute error of the output is (3.0876, 3.4107, 3.2158) in cm with standard deviation (2.7049, 2.3013, 2.2238).

The inverse model is also a fully connected neural network with three inputs, one hidden layer of 16 neuron and four output. The input to the model is the position of the tip of the arm, and the output is the four cables' actuations. The mean absolute error of the output is (1.6662, 1.6887, 1.6410, 1.6588) with standard deviation (1.2758, 1.2878, 1.2948, 1.2942)

To test the iverse and forward models together, we tested it firstly by inserting the tip position to the inverse model to infer the tensions, then the output of the inverse model is then used as input to forward model and compare the output of the FK model with truth value. The mean absolute error of the output is (4.1383, 3.3612, 3.3428) in cm with standard deviation (2.5764, 2.2733, 2.1268). The error of the two models together is slightly higer than the forward model alone as the error in the inverse model is affecting the results. 


Conv3D_IK_300_0-1_trained.ipynb train the model of the shape of the arm with the file model300_wBN_DrBerkeCode_80KS_0-1.pt is the weights of the model
