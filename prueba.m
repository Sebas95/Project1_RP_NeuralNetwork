 
 pkg load optim;
 
 [X,Y]=create_data(100000,numClasses=3,shape="horizontal");

 #plot_data(X,Y);
 
 NumInputs = 3;
 neuronsHidden = 10; 
 outputs = 3;
 
 W1 = rand(neuronsHidden,NumInputs)-rand(neuronsHidden,NumInputs);
 W2 = rand(outputs,neuronsHidden+1)-rand(outputs,neuronsHidden+1);
 display(W1);
 display(W2);
 lamda = 0.2;
  for i=[1:rows(X)] 
    
   [gW1,gW2]=gradtarget(W1,W2, X(i,:) ,Y(i,:));
  
    W1 = W1 - lamda.*gW1;
    W2 = W2 - lamda.*gW2;
  endfor

 #X=X(1,:);
 #Y=Y(1);
 #--------
 
 