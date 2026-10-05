function F02_data_and_cohort()
%F02_DATA_AND_COHORT  Omics resources and the real paediatric cohort.
%   Subfigures: F02a_dataset_completeness, F02b_cohort_age,
%               F02c_omic_phenotype_plane, F02d_resistance_distribution
C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);

% ---------- (a) dataset completeness -------------------------------------
fc = fullfile(P.reports,'data_completeness.tsv');
if isfile(fc)
    T = readtable(fc,'FileType','text','Delimiter','\t','VariableNamingRule','preserve');
    ds = string(T.(1)); st = string(T.(2));
    [ust,~,gi] = unique(st);
    fig = mal_figure(30,20); ax = axes(fig); hold(ax,'on');
    cmap = containers.Map({'PASS','PARTIAL','MANAGED','MISSING'}, ...
                          {pal(3,:), pal(2,:), pal(5,:), pal(6,:)});
    for i = numel(ds):-1:1
        c = pal(11,:); if isKey(cmap,char(st(i))), c = cmap(char(st(i))); end
        barh(ax, numel(ds)-i+1, 1, 'FaceColor',c,'EdgeColor','k','LineWidth',1.2);
    end
    set(ax,'YTick',1:numel(ds),'YTickLabel',flipud(cellstr(strrep(ds,'_','\_'))));
    xlim(ax,[0 1.35]); set(ax,'XTick',[]);
    for i = 1:numel(ds)
        text(ax, 1.04, numel(ds)-i+1, char(st(i)), 'FontName',C.fontName, ...
             'FontSize',C.fsAnnot,'VerticalAlignment','middle');
    end
    mal_axes(ax,'Data resource acquisition status','','');
    hs = gobjects(numel(ust),1);
    for k = 1:numel(ust)
        c = pal(11,:); if isKey(cmap,char(ust(k))), c = cmap(char(ust(k))); end
        hs(k) = patch(ax,NaN,NaN,c);
    end
    legend(ax,hs,cellstr(ust),'Location','southoutside','Orientation','horizontal');
    mal_axes(ax);
    mal_export(fig,'F02a_dataset_completeness');
else
    warning('F02a skipped: %s not found', fc);
end

% ---------- cohort --------------------------------------------------------
fco = fullfile(P.tables,'paediatric_cohort_used.csv');
if ~isfile(fco)
    warning('F02b-d skipped: %s not found', fco); return
end
K = readtable(fco,'Delimiter',',','ReadVariableNames',true,'VariableNamingRule','preserve');
n  = height(K);
src = 'unknown'; if any(strcmp(K.Properties.VariableNames,'source')), src = char(string(K.source(1))); end

% (b) age distribution
fig = mal_figure(26,19); ax = axes(fig); hold(ax,'on');
histogram(ax, K.age_years, 'BinWidth',1, 'FaceColor',pal(1,:), ...
          'EdgeColor','k','LineWidth',1.2,'FaceAlpha',0.85);
mal_axes(ax,'Paediatric cohort: age','age (years)','number of children');
mal_label(ax, sprintf('n = %d children\nsource: %s', n, strrep(src,'_','\_')),'northeast');
mal_export(fig,'F02b_cohort_age');

% (c) omic phenotype plane
fig = mal_figure(28,19); ax = axes(fig); hold(ax,'on');
scatter(ax, K.host_metabolic_z, K.host_tox_z, C.markerSize, K.parasite_resistance, ...
        'filled','MarkerEdgeColor','k','LineWidth',0.8);
colormap(ax, parula); cb = colorbar(ax);
cb.Label.String = mal_titlecase('parasite resistance burden');
cb.Label.FontSize = C.fsAxis; cb.Label.FontName = C.fontName;
xline(ax,0,'k--','LineWidth',1.2); yline(ax,0,'k--','LineWidth',1.2);
mal_axes(ax,'Host molecular phenotype plane', ...
         'host metabolic proxy (z)','host toxicity proxy (z)');
mal_label(ax, sprintf('n = %d', n),'northwest');
mal_export(fig,'F02c_omic_phenotype_plane');

% (d) resistance distribution
fig = mal_figure(26,19); ax = axes(fig); hold(ax,'on');
histogram(ax, K.parasite_resistance, 'BinWidth',0.1, 'FaceColor',pal(6,:), ...
          'EdgeColor','k','LineWidth',1.2,'FaceAlpha',0.85);
mal_axes(ax,'Parasite resistance burden','resistance burden (0-1)','number of children');
mal_export(fig,'F02d_resistance_distribution');
fprintf('F02 done.\n');
end
