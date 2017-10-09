
[X,Y]=create_data(100,numClasses=3,shape="vertical");

NumInputs = 3;
neuronsHidden = 5; 
outputs = 3;

W1 = rand(neuronsHidden,NumInputs)-rand(neuronsHidden,NumInputs);
W2 = rand(outputs,neuronsHidden+1)-rand(outputs,neuronsHidden+1);

y=target(W1,W2,X,Y);
disp(y);