


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
  
  #display([1 ;X']);
  #display(W1);
  arg1 = sigmoid( [  W1*  [X']]  );
 # display(arg1);

  y = sigmoid( W2 * [ones(1,columns(arg1)); arg1]);
endfunction;

 function g=sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
endfunction;
