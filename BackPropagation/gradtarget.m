function [gW1,gW2]=gradtarget(W1,W2,X,Y)

  # usage gradtarget(W1,W2,X,Y)
  # 
  # This function evaluates the gradient of the target function on W1 and W2.
  # 
  # W1: weights matrix between input and hidden layer
  # W2: weights matrix between the hidden and the output layer
  # X:  training set holding on the rows the input data, plus a final column 
  #     equal to 1
  # Y:  labels of the training set
 xi = [ 1  X];
 yk = predict(W1,W2,xi)';
 dk = yk - Y;    
 zj = sigmoid( W1*xi');
 zj_bias =  [ ones(1,columns(zj)) ; zj];
 dj = (zj_bias.*(1.-zj_bias) )' .*  (dk*W2) ;
 dj(:,1) = []; 
 gW1 =  [dj' dj' dj'] .* [ xi ; xi ; xi ; xi ; xi];
 gW2 =  [dk' dk' dk' dk' dk' dk' ] .* [zj_bias' ; zj_bias' ; zj_bias'];

endfunction;

 function g=sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
endfunction;