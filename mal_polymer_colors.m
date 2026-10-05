function [C, order] = mal_polymer_colors()
%MAL_POLYMER_COLORS  Fixed colour per polymer so every figure agrees.
order = {'PLGA_50_50','PLGA_75_25','PLA_PDLLA','PCL','Chitosan', ...
         'Eudragit_RS100','Eudragit_RL100','Eudragit_S100','Alginate', ...
         'Ethylcellulose','HPMC'};
rgb = mal_palette(numel(order));
C = containers.Map(order, num2cell(rgb,2)');
end
