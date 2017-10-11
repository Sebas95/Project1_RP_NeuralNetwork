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
## @deftypefn {} {@var{retval} =} visual_data (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Dell <Dell@DESKTOP-67S8TFV>
## Created: 2017-10-10

function visual_data (W1, W2, X, Y)
      #crear matriz con todas las posibles posiciones de la imagen de 512x512
      [tt0,tt1] = meshgrid(1:512,1:512);
      Xn=[tt0(:) tt1(:)];
      #mapear posiciones a intervalo -1 a 1
      Xn= (Xn./256-1);
      #predecir Xn 
      Xaux= [ones(rows(Xn), 1), Xn];
      Ypred= predict(W1, W2, Xaux);
      
      #mapear X a coordenadas para producir el plot en la misma gráfica
      Xgraph= X*256+256;
      
      imagen= reshape(Ypred', 512, 512, 3);
      imwrite(imagen, "output.jpg");
      hold on;
      imshow(imagen);
      plot_data(Xgraph, Y);
      hold off;
endfunction
