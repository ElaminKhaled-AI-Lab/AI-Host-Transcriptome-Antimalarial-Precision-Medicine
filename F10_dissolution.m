function F10_dissolution()
%F10_DISSOLUTION  In-vitro release profiles, dissolution parameters and kinetics.
%   Subfigures: F10a..F10h
C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);
[cmap, order] = mal_polymer_colors();

% ---------------- release profiles ----------------------------------------
fpr = fullfile(P.tables,'nano_dds_release_profiles.csv');
if isfile(fpr)
    R = readtable(fpr,'Delimiter',',','ReadVariableNames',true,'VariableNamingRule','preserve');
    drugMain = 'lumefantrine';

    % (a) all polymers, one named drug
    sel = strcmp(string(R.drug),drugMain) & strcmp(string(R.series),'centre');
    S = R(sel,:); pol = string(S.polymer);
    present = order(ismember(order, unique(pol)));
    fig = mal_figure(34,20); ax = axes(fig); hold(ax,'on');
    h = gobjects(numel(present),1);
    for i = 1:numel(present)
        q = S(pol == present{i},:); [~,ix] = sort(q.time_h);
        h(i) = plot(ax, q.time_h(ix), q.Q_pct(ix),'-o','Color',cmap(present{i}), ...
            'MarkerFaceColor',cmap(present{i}),'MarkerSize',7,'LineWidth',C.lineWidth);
    end
    vh = isgraphics(h);
    legend(ax,h(vh),strrep(present(vh),'_','\_'),'Location','eastoutside');
    mal_axes(ax, sprintf('Release profiles - drug: %s', upper(drugMain)), ...
             'time (h)','cumulative release (%)');
    mal_label(ax,'centre point: ratio 5:1, stabiliser 1%, medium energy','southeast');
    mal_export(fig,'F10a_release_profiles_polymers');

    % (b) all drugs on one polymer
    sel = strcmp(string(R.polymer),'PLGA_50_50') & strcmp(string(R.series),'centre');
    S = R(sel,:); dr = string(S.drug); ud = unique(dr);
    fig = mal_figure(34,20); ax = axes(fig); hold(ax,'on');
    pd = mal_palette(numel(ud)); h = gobjects(numel(ud),1);
    for i = 1:numel(ud)
        q = S(dr == ud(i),:); [~,ix] = sort(q.time_h);
        h(i) = plot(ax, q.time_h(ix), q.Q_pct(ix),'-s','Color',pd(i,:), ...
            'MarkerFaceColor',pd(i,:),'MarkerSize',7,'LineWidth',C.lineWidth);
    end
    vh = isgraphics(h); ud_ = cellstr(ud);
    legend(ax,h(vh),ud_(vh),'Location','eastoutside');
    mal_axes(ax,'Release profiles - polymer: PLGA 50:50','time (h)','cumulative release (%)');
    mal_export(fig,'F10b_release_profiles_drugs');

    % (c) factor effect on release
    fig = mal_figure(32,20); ax = axes(fig); hold(ax,'on');
    ser = {'A_low','ratio 2:1'; 'centre','ratio 5:1 (centre)'; 'A_high','ratio 10:1'; ...
           'C_low','low energy'; 'C_high','high energy'};
    h = gobjects(size(ser,1),1); nk = 0;
    for i = 1:size(ser,1)
        sel = strcmp(string(R.drug),drugMain) & strcmp(string(R.polymer),'PLGA_50_50') & ...
              strcmp(string(R.series),ser{i,1});
        if ~any(sel), continue; end
        nk = nk + 1; q = R(sel,:); [~,ix] = sort(q.time_h);
        h(nk) = plot(ax, q.time_h(ix), q.Q_pct(ix),'-o','Color',pal(nk,:), ...
            'MarkerFaceColor',pal(nk,:),'MarkerSize',8,'LineWidth',C.lineWidth);
        lbls{nk} = ser{i,2}; %#ok<AGROW>
    end
    vh = isgraphics(h(1:nk));
    legend(ax,h(vh),lbls(vh),'Location','eastoutside');
    mal_axes(ax, sprintf('Factor effect on release - %s / PLGA 50:50', upper(drugMain)), ...
             'time (h)','cumulative release (%)');
    mal_export(fig,'F10c_release_factor_effect');
end

% ---------------- dissolution parameters ----------------------------------
fp = fullfile(P.tables,'nano_dds_dissolution_parameters.csv');
if ~isfile(fp), warning('F10d-h skipped: %s not found', fp); fprintf('F10 done.\n'); return; end
T = readtable(fp,'Delimiter',',','ReadVariableNames',true,'VariableNamingRule','preserve');
pol = string(T.polymer);
present = order(ismember(order, unique(pol)));

spec = {'DE_pct','dissolution efficiency DE (%)','Dissolution efficiency','F10d_DE'; ...
        'MDT_h','mean dissolution time MDT (h)','Mean dissolution time','F10e_MDT'; ...
        't50_h','t_{50} (h)','Time to 50% release','F10f_t50'; ...
        'n_peppas','Korsmeyer-Peppas exponent n','Release exponent','F10g_peppas_n'};
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
    errorbar(ax,1:numel(present),mu,sd,'k','LineStyle','none','LineWidth',1.8,'CapSize',10);
    set(ax,'XTick',1:numel(present),'XTickLabel',strrep(present,'_','\_'));
    ax.XTickLabelRotation = 35;
    if strcmp(spec{s,1},'n_peppas')
        yline(ax,0.43,'r--','LineWidth',1.8); yline(ax,0.85,'k--','LineWidth',1.8);
        mal_label(ax,'n \leq 0.43 Fickian; 0.43 < n < 0.85 anomalous','northwest');
    end
    mal_axes(ax, spec{s,3},'', spec{s,2});
    mal_export(fig, spec{s,4});
end

% (h) kinetic-model fit heat map
r2v = {'r2_zero_order','zero-order'; 'r2_first_order','first-order'; ...
       'r2_higuchi','Higuchi'; 'r2_peppas','Korsmeyer-Peppas'; 'r2_hixson','Hixson-Crowell'};
have = ismember(r2v(:,1), T.Properties.VariableNames);
r2v = r2v(have,:);
if ~isempty(r2v)
    M = nan(numel(present), size(r2v,1));
    for i = 1:numel(present)
        m = pol == present{i};
        for jj = 1:size(r2v,1)
            M(i,jj) = mean(T.(r2v{jj,1})(m),'omitnan');
        end
    end
    fig = mal_figure(32,22); ax = axes(fig);
    imagesc(ax, M); colormap(ax, parula); mal_clim(ax,[0 1]);
    cb = colorbar(ax); cb.Label.String = mal_titlecase('mean R^2');
    cb.Label.FontName = C.fontName; cb.Label.FontSize = C.fsAxis;
    set(ax,'XTick',1:size(r2v,1),'XTickLabel',r2v(:,2), ...
           'YTick',1:numel(present),'YTickLabel',strrep(present,'_','\_'));
    ax.XTickLabelRotation = 30;
    for i = 1:numel(present)
        for jj = 1:size(r2v,1)
            if M(i,jj) < 0.55, tc = 'w'; else, tc = 'k'; end
            text(ax,jj,i,sprintf('%.2f',M(i,jj)),'HorizontalAlignment','center', ...
                'FontName',C.fontName,'FontSize',C.fsAnnot,'Color',tc);
        end
    end
    mal_axes(ax,'Kinetic-model fit by polymer','','');
    mal_export(fig,'F10h_kinetic_model_heatmap');
end
fprintf('F10 done.\n');
end
