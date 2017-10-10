
 [X,Y]=create_data(1,numClasses=3,shape="radial");
function g=sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
endfunction;
   
 W1 = rand(5,3)-rand(5,3);
 W2 = rand(3,6)-rand(3,6);
 

 yk = predict(W1,W2,X)';
 dk = yk - Y;    
 xi = [ 1  X];
 zj = sigmoid( W1*xi');
 zj_bias =  [ ones(1,columns(zj)) ; zj];
 dj = (zj_bias.*(1.-zj_bias) )' .*  (dk*W2) ;
 dj(:,1) = []; 
 gW1 =  [dj' dj' dj'] .* [ xi ; xi ; xi ; xi ; xi];
 gW2 =  [dk' dk' dk' dk' dk' dk' ] .* [zj_bias' ; zj_bias' ; zj_bias'];