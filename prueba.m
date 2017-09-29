 
 pkg load optim;
 
 [X,Y]=create_data(1000,numClasses=3,shape="radial");

   
function g=sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
endfunction;
 
  
 W1 = ones(1000,3);
 W2 = ones(1000,3);
 
 X = [ ones(rows(X),1)  X(:,1)  X(:,2)];
 Arg = W1 .* X;
  
 Arg2 = W2   .* sigmoid(Arg);

 y = sigmoid(Arg2);
  
