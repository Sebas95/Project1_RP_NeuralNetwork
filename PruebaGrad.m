
 [X,Y]=create_data(1,numClasses=3,shape="radial");
function g=sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
endfunction;
   
 W1 = ones(5,3);
 W2 = ones(3,6);
r1 = rand(5,3);

 yk = predict(W1,W2,X)';
 dk = yk - Y;    
 xi = [ 1  X];
 zj_aux = sigmoid( W1*xi');
 zj =  [ ones(1,columns(zj_aux)) ; zj_aux];
 dj = (zj.*(1.-zj) )' .*  (dk*W2) ; 
 gW1 =  [ zj_aux  zj_aux  zj_aux ] .* [dk ; dk ; dk ; dk ; dk  ];
 gW2 =  [xi ; xi ; xi ; xi ; xi ; xi  ]'.*[dj; dj; dj];