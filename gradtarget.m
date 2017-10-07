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
 yk = predict(W1,W2,X)';
 dk = yk - Y;    
 xi = [ 1  X];
 zj = sigmoid( W1*xi');
 zj_bias =  [ ones(1,columns(zj)) ; zj];
 dj = (zj_bias.*(1.-zj_bias) )' .*  (dk*W2) ; 
 
 gW1 =  [ zj  zj  zj ] .* [dk ; dk ; dk ; dk ; dk; dk ; dk ; dk ; dk ; dk ];
 gW2 =  [xi ; xi ; xi ; xi ; xi ; xi ; xi ; xi ; xi  ; xi ; xi  ]'.*[dj; dj; dj];

endfunction;

 function g=sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
endfunction;