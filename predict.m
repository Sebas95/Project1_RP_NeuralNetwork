


function y=predict(W1,W2,X)
    
  # usage predict(W1,W2,X)
  # 
  # This function propagates the input X on the neural network to
  # predict the output vector y, given the weight matrices W1 and W2 for 
  # a two-layered artificial neural network.
  # 
  # W1: weights matrix between input and hidden layer
  # W2: weights matrix between the hidden and the output layer
  # X:  Input vector, extended at its end with a 1
    
  
  X1 = [ ones(rows(X),1)  X(:,1)  X(:,2)];
  Arg = W1 .* X1;
  
  Arg2 = W2   .* 1.0 ./ (1.0 + exp(-Arg));
  
  y =  1.0 ./ (1.0 + exp(-Arg2));
endfunction;

  