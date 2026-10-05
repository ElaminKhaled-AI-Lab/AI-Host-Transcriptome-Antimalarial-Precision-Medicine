function [rgb, names] = mal_palette(n)
%MAL_PALETTE  Colour-blind-friendly qualitative palette (extended Okabe-Ito).
base = [ 0.000 0.447 0.698 ;   % blue
         0.902 0.624 0.000 ;   % orange
         0.000 0.620 0.451 ;   % green
         0.800 0.475 0.655 ;   % pink
         0.337 0.706 0.914 ;   % sky
         0.835 0.369 0.000 ;   % vermillion
         0.941 0.894 0.259 ;   % yellow
         0.494 0.184 0.556 ;   % purple
         0.466 0.674 0.188 ;   % olive
         0.635 0.078 0.184 ;   % dark red
         0.300 0.300 0.300 ];  % grey
names = {'blue','orange','green','pink','sky','vermillion','yellow','purple','olive','darkred','grey'};
if nargin < 1 || isempty(n), n = size(base,1); end
if n <= size(base,1)
    rgb = base(1:n,:);
else
    rgb = base(mod(0:n-1, size(base,1))+1, :);
end
end
