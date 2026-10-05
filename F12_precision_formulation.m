function F12_precision_formulation()
%F12_PRECISION_FORMULATION  Patient-specific formulation and the integration benchmark.
%   Subfigures: F12a_polymer_per_child, F12b_phenotype_plane,
%               F12c_personalisation_gain, F12d_interaction_metric,
%               F12e_strategy_accuracy, F12f_therapeutic_window
C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);
[cmap, order] = mal_polymer_colors();

fpp = fullfile(P.tables,'personalized_formulation_per_patient.csv');
if isfile(fpp)
    K = readtable(fpp,'Delimiter',',','ReadVariableNames',true,'VariableNamingRule','preserve');
    pol = string(K.polymer);
    present = order(ismember(order, unique(pol)));
    cnt = cellfun(@(p) sum(pol == p), present);

    % (a) polymer chosen per child
    [cnt,ix] = sort(cnt,'descend'); nm = present(ix);
    fig = mal_figure(32,20); ax = axes(fig); hold(ax,'on');
    b = bar(ax, cnt, 0.6,'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
    for i = 1:numel(cnt), b.CData(i,:) = cmap(nm{i}); end
    set(ax,'XTick',1:numel(cnt),'XTickLabel',strrep(nm,'_','\_'));
    ax.XTickLabelRotation = 25;
    for i = 1:numel(cnt)
        text(ax,i,cnt(i),sprintf('%d',cnt(i)),'FontName',C.fontName,'FontSize',C.fsAnnot, ...
             'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
    mal_axes(ax,'Personalised polymer choice','','number of children');
    mal_label(ax,sprintf('n = %d children',height(K)),'northeast');
    mal_export(fig,'F12a_polymer_per_child');

    % (b) phenotype plane coloured by chosen polymer
    fig = mal_figure(34,20); ax = axes(fig); hold(ax,'on');
    h = gobjects(numel(nm),1);
    for i = 1:numel(nm)
        m = pol == nm{i};
        h(i) = scatter(ax, K.host_metabolic_z(m), K.host_tox_z(m), C.markerSize, ...
            'filled','MarkerFaceColor',cmap(nm{i}),'MarkerEdgeColor','k','LineWidth',0.8);
    end
    xline(ax,0,'k--','LineWidth',1.2); yline(ax,0,'k--','LineWidth',1.2);
    vh = isgraphics(h);
    legend(ax,h(vh),strrep(nm(vh),'_','\_'),'Location','eastoutside');
    mal_axes(ax,'Chosen formulation across the omic phenotype plane', ...
             'host metabolic proxy (z)','host toxicity proxy (z)');
    mal_export(fig,'F12b_phenotype_plane');

    % (c) personalisation gain
    if any(strcmp(K.Properties.VariableNames,'objective_gain'))
        fig = mal_figure(30,20); ax = axes(fig); hold(ax,'on');
        histogram(ax, K.objective_gain,'BinWidth',0.01,'FaceColor',pal(1,:), ...
                  'EdgeColor','k','LineWidth',1.2,'FaceAlpha',0.85);
        mg = mean(K.objective_gain,'omitnan');
        xline(ax, mg,'r--','LineWidth',2.2);
        mal_axes(ax,'Benefit of personalisation', ...
                 'therapeutic-utility gain vs one-size-fits-all','number of children');
        mal_label(ax, sprintf('mean +%.4f, max +%.4f', mg, max(K.objective_gain)),'northeast');
        mal_export(fig,'F12c_personalisation_gain');
    end

    % (f) therapeutic window at each child's optimum
    if all(ismember({'day7_ng_per_mL','utility','toxicity_risk'},K.Properties.VariableNames))
        fig = mal_figure(32,20); ax = axes(fig); hold(ax,'on');
        scatter(ax, K.day7_ng_per_mL, K.utility, C.markerSize, K.toxicity_risk, ...
                'filled','MarkerEdgeColor','k','LineWidth',0.8);
        colormap(ax, flipud(hot)); cb = colorbar(ax);
        cb.Label.String = mal_titlecase('toxicity risk');
        cb.Label.FontName = C.fontName; cb.Label.FontSize = C.fsAxis;
        set(ax,'XScale','log');
        mal_axes(ax,'Therapeutic window at each child''s optimum', ...
                 'day-7 exposure (ng/mL)','therapeutic utility');
        mal_export(fig,'F12f_therapeutic_window');
    end
end

% ---------------- integration-strategy benchmark --------------------------
fb = fullfile(P.results,'integration_benchmark_profile_net.json');
if ~isfile(fb), warning('F12d/e skipped: %s not found', fb); fprintf('F12 done.\n'); return; end
B = mal_readjson(fb);
M = B.metrics; sn = fieldnames(M);
nice = strrep(sn,'_','\_');
ir = nan(numel(sn),1); irsd = ir; ex = ir; ut = ir; gn = ir;
for i = 1:numel(sn)
    m = M.(sn{i});
    if isfield(m,'interaction_residual_rank_corr'),    ir(i)   = m.interaction_residual_rank_corr; end
    if isfield(m,'interaction_residual_rank_corr_sd'), irsd(i) = m.interaction_residual_rank_corr_sd; end
    if isfield(m,'exposure_r2'),                       ex(i)   = m.exposure_r2; end
    if isfield(m,'utility_r2'),                        ut(i)   = m.utility_r2; end
    if isfield(m,'personalization_gain_utility'),      gn(i)   = m.personalization_gain_utility; end
end
win = ''; if isfield(B,'winner'), win = B.winner; end
sep = true; if isfield(B,'winner_is_separated_from_noise'), sep = logical(B.winner_is_separated_from_noise); end

% (d) the deciding interaction metric
fig = mal_figure(34,20); ax = axes(fig); hold(ax,'on');
bb = barh(ax, ir, 0.6,'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
for i = 1:numel(ir)
    if strcmp(sn{i},win) && sep, bb.CData(i,:) = pal(3,:); else, bb.CData(i,:) = pal(1,:); end
end
errorbar(ax, ir, 1:numel(ir), irsd,'horizontal','k','LineStyle','none', ...
         'LineWidth',1.8,'CapSize',12);
xline(ax,0,'k-','LineWidth',1.4);
set(ax,'YTick',1:numel(ir),'YTickLabel',nice);
mal_axes(ax,'Patient x formulation interaction captured', ...
         'interaction-residual rank correlation','');
for i = 1:numel(ir)
    text(ax, max(ir(i),0)+max(irsd)+0.03, i, sprintf('%.3f',ir(i)), ...
         'FontName',C.fontName,'FontSize',C.fsAnnot,'VerticalAlignment','middle');
end
xlim(ax,[min(0,min(ir))-0.08, max(ir)+0.22]);
if sep
    mal_label(ax, sprintf('winner: %s (separated from seed noise)', strrep(win,'_','\_')),'southeast');
else
    mal_label(ax,'joint strategies tie; LATE integration fails','southeast');
end
mal_export(fig,'F12d_interaction_metric');

% (e) accuracy and gain by strategy
fig = mal_figure(34,20); ax = axes(fig); hold(ax,'on');
Y = [ex ut];
b = bar(ax, Y,'grouped','EdgeColor','k','LineWidth',1.4);
b(1).FaceColor = pal(1,:); b(2).FaceColor = pal(2,:);
set(ax,'XTick',1:numel(sn),'XTickLabel',nice); ax.XTickLabelRotation = 20;
ylim(ax,[0 1.12]);
legend(ax,{'exposure R^2','utility R^2'},'Location','northoutside','Orientation','horizontal');
mal_axes(ax,'Predictive accuracy by integration strategy','','R^2');
mal_export(fig,'F12e_strategy_accuracy');
fprintf('F12 done.\n');
end
