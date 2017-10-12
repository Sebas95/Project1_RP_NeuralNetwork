## Copyright (C) 2017 Dell
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} conf_matrix (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Dell <Dell@DESKTOP-67S8TFV>
## Created: 2017-10-10

function ret = conf_matrix (W1, W2, dist)
    #conjunto de prueba
    [X,Yreal]=create_data(1000,numClasses=3,shape=dist);
    #guardar conjunto de prueba
    save -6 "xtest_set.mat" X;
    save -6 "ytest_set.mat" Yreal;
    
    #datos predecidos
    Ypred= predict(W1, W2, [ones(rows(X), 1), X]);
    Ypred= Ypred';
    ret= zeros(3, 3);
    for i=[1:rows(Ypred)]
      if (Ypred(i,1)>Ypred(i,2) && Ypred(i,1)>Ypred(i,3)) #predijo clase 1
          if (Yreal(i,1)==1) #Clase real clase 1
               ret(1, 1)++;
          elseif (Yreal(i,2)==1) #Clase real clase 2
               ret(2, 1)++;
          elseif (Yreal(i,3)==1) #Clase real clase 2
               ret(3, 1)++;
          endif
      elseif (Ypred(i,2)>Ypred(i,1) && Ypred(i,2)>Ypred(i,3)) #predijo clase 2
          if (Yreal(i,1)==1) #Clase real clase 1
               ret(1, 2)++;
          elseif (Yreal(i,2)==1) #Clase real clase 2
               ret(2, 2)++;
          elseif (Yreal(i,3)==1) #Clase real clase 2
               ret(3, 2)++;
          endif
      elseif (Ypred(i,3)>Ypred(i,2) && Ypred(i,3)>Ypred(i,1)) #predijo clase 3
          if (Yreal(i,1)==1) #Clase real clase 1
               ret(1, 3)++;
          elseif (Yreal(i,2)==1) #Clase real clase 2
               ret(2, 3)++;
          elseif (Yreal(i,3)==1) #Clase real clase 2
               ret(3, 3)++;
          endif
      endif
    endfor
    sens= [sensitivity(ret, 1), sensitivity(ret, 2), sensitivity(ret, 3)];
    pres= [precision(ret, 1), precision(ret, 2), precision(ret, 3)];
    ret =[ret, sens', pres'];
endfunction

function res= sensitivity(matrix, class_n)
   res = matrix(class_n, class_n)/(sum(matrix(class_n,:)));
endfunction

function res= precision(matrix, class_n)
   res = matrix(class_n, class_n)/(sum(matrix(:,class_n)));
endfunction
