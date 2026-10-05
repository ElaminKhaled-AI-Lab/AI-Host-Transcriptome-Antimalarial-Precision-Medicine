function F09_nano_characterisation()
%F09_NANO_CHARACTERISATION  Nanoparticle characterisation across every polymer.
%   Subfigures: F09a_size, F09b_pdi, F09c_zeta, F09d_ee, F09e_dl, F09f_liver_index
C = mal_cfg(); P = mal_paths();
[cmap, order] = mal_polymer_colors();

fp = fullfile(P.tables,'nano_dds_dissolution_parameters.csv');
if ~isfile(fp), warning('F09 skipped: %s not found', fp); return; end
T = readtable(fp,'Delimiter',',','ReadVariableNames',true,'VariableNamingRule','preserve');
pol = string(T.polymer);
present = order(ismember(order, unique(pol)));

spec = {'size_nm','particle size (nm)','Particle size','F09a_size'; ...
        'pdi','polydispersity index','Polydispersity index','F09b_pdi'; ...
        'zeta_mv','zeta potential (mV)','Zeta potential','F09c_zeta'; ...
        'ee_pct','entrapment efficiency (%)','Entrapment efficiency','F09d_ee'; ...
        'dl_pct','drug loading (%)','Drug loading','F09e_dl'; ...
        'liver_targeting_index','liver-targeting index','Liver-targeting index','F09f_liver_index'};

for s = 1:size(spec,1)
    if ~any(strcmp(T.Properties.VariableNames, spec{s,1})), continue; end
    mu = nan(numel(present),1); sd = mu;
    for i = 1:numel(present)
        v = T.(spec{s,1})(pol == present{i});
        mu(i) = mean(v,'omitnan'); sd(i) = std(v,0,'omitnan');
    end
    fig = mal_figure(34,20); ax = axes(fig); hold(ax,'on');
    b = bar(ax, mu, 0.65,'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
    for i = 1:numel(present), b.CData(i,:) = cmap(present{i}); end
    errorbar(ax, 1:numel(present), mu, sd, 'k','LineStyle','none','LineWidth',1.8,'CapSize',10);
    set(ax,'XTick',1:numel(present),'XTickLabel',strrep(present,'_','\_'));
    ax.XTickLabelRotation = 35;

    switch spec{s,1}
        case 'size_nm'
            yl = ylim(ax); patch(ax,[0.4 numel(present)+0.6 numel(present)+0.6 0.4], ...
                [100 100 200 200],[0.4 0.4 0.4],'FaceAlpha',0.14,'EdgeColor','none');
            uistack(findobj(ax,'Type','patch'),'bottom'); ylim(ax,yl);
            mal_label(ax,'shaded band = 100-200 nm Kupffer window','northwest');
        case 'zeta_mv'
            yline(ax, 20,'r--','LineWidth',1.6); yline(ax,-20,'r--','LineWidth',1.6);
            mal_label(ax,'|\zeta| = 20 mV colloidal-stability threshold','southwest');
    end
    mal_axes(ax, spec{s,3},'', spec{s,2});
    mal_label(ax, sprintf('n = %d formulations', height(T)),'northeast');
    mal_export(fig, spec{s,4});
end
fprintf('F09 done.\n');
end
