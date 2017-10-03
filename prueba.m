 
 pkg load optim;
 
 [X,Y]=create_data(1000,numClasses=3,shape="radial");

   
function g=sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
endfunction;
 
  
 W1 = ones(5,3);
 W2 = ones(3,6);
 
 #X = [ 1 2 3];
 X = [ ones(rows(X),1)  X(:,1)  X(:,2)];
 
 arg1 = sigmoid( W1*X'  );
 
 y = sigmoid( W2 * [ ones(1,columns(arg1)) ; arg1] );
  
 #Arg2 = W2   .* sigmoid(Arg);

 #y = sigmoid(Arg2);
  
#______________________________Gradient of J____________________________________
# Analytical solution.
#
# For each theta row (assumed in a row of the theta matrix) it will
# compute also a row with the gradient: the first column is the partial
# derivative w.r.t theta_0 and the second w.r.t theta_1
function res=gradJ(theta,X,Y)
  res=(X'*(X*theta'-Y*ones(1,rows(theta))))';
endfunction;

