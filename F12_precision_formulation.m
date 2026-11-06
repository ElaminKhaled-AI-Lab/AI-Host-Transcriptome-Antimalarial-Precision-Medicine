function F12_precision_formulation()
%F12_PRECISION_FORMULATION  Personalisation under the corrected objective, and
%   the integration-strategy benchmark.
%
%   Subfigures: F12a_polymer_collapse, F12b_gain_vs_null,
%               F12c_strategy_accuracy, F12d_interaction_metric
%
%   Reframed for the revision. Under the mechanistically appropriate lumefantrine
%   objective (the hepatic-targeting term removed for a blood-stage drug), the
%   per-child assignment collapses onto a single polymer and the utility gain is
%   negligible and not above a molecular-permutation null. Panels (a) and (b)
%   therefore contrast the as-submitted (liver-weighted) objective with the
%   revised one, instead of asserting a polymer crossover that does not survive
%   correction. Panels (c) and (d) report the integration benchmark under
%   repeated grouped cross-validation over the 56 children.
%
%   Inputs:
%     results/personalization_stats.json         (both objectives + null)
%     results/benchmark_cv_revised.json           (cross-validated benchmark)

C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);
[cmap, order] = mal_polymer_colors();

% ======================= personalisation (a, b) ===========================
fp = fullfile(P.results,'personalization_stats.json');
if ~isfile(fp)
    warning('F12a/b skipped: %s not found', fp);
else
    S = mal_readjson(fp);
    conds = {'as_submitted_hepatic_on','as-submitted'; ...
             'revised_hepatic_off','revised'};
    haveC = cellfun(@(f) isfield(S,f), conds(:,1));
    conds = conds(haveC,:); nc = size(conds,1);

    % ---- (a) polymer assignment collapses to a single polymer -------------
    polset = {};
    for c = 1:nc
        pc = S.(conds{c,1}).polymer_counts;
        polset = [polset; fieldnames(pc)]; %#ok<AGROW>
    end
    polset = unique(polset,'stable');
    polk = order(ismember(order, polset));               % canonical order
    Cnt = zeros(numel(polk), nc);
    for c = 1:nc
        pc = S.(conds{c,1}).polymer_counts;
        for i = 1:numel(polk)
            if isfield(pc, polk{i}), Cnt(i,c) = pc.(polk{i}); end
        end
    end
    fig = mal_figure(32,20); ax = axes(fig); hold(ax,'on');
    hb = bar(ax, Cnt, 'grouped','EdgeColor','k','LineWidth',1.4);
    cond_cols = [pal(11,:); pal(6,:)];                   % grey = as-submitted, accent = revised
    for c = 1:nc, hb(c).FaceColor = cond_cols(min(c,size(cond_cols,1)),:); end
    set(ax,'XTick',1:numel(polk),'XTickLabel',strrep(polk,'_','\_'));
    ax.XTickLabelRotation = 20;
    legend(ax, conds(:,2),'Location','northoutside','Orientation','horizontal');
    mal_axes(ax,'Polymer assignment collapses to one polymer','','number of children');
    ymax = max(Cnt(:));
    for c = 1:nc
        for i = 1:numel(polk)
            if Cnt(i,c) > 0
                text(ax, hb(c).XEndPoints(i), Cnt(i,c), sprintf('%d',Cnt(i,c)), ...
                     'FontName',C.fontName,'FontSize',C.fsAnnot, ...
                     'HorizontalAlignment','center','VerticalAlignment','bottom');
            end
        end
    end
    ylim(ax,[0 ymax*1.18]);
    mal_label(ax,{'Removing the liver term (inappropriate for a blood-stage', ...
                  'drug) sends every child to polycaprolactone.'},'northeast');
    mal_export(fig,'F12a_polymer_collapse');

    % ---- (b) utility gain against a molecular-permutation null ------------
    obs = nan(nc,1); lo = obs; hi = obs; nmd = obs; nlo = obs; nhi = obs; pval = obs;
    for c = 1:nc
        b = S.(conds{c,1});
        obs(c) = b.mean_gain;
        ci = b.mean_gain_95ci_bootstrap_children; lo(c) = ci(1); hi(c) = ci(2);
        mp = b.molecular_permutation_null;
        nmd(c) = mp.null_median; rg = mp.null_95pct_range; nlo(c) = rg(1); nhi(c) = rg(2);
        pval(c) = mp.p_one_sided;
    end
    fig = mal_figure(30,20); ax = axes(fig); hold(ax,'on');
    x = 1:nc;
    for c = x
        % permutation-null band (grey) and its median (dashed)
        patch(ax, [c-0.3 c+0.3 c+0.3 c-0.3], [nlo(c) nlo(c) nhi(c) nhi(c)], ...
              [0.6 0.6 0.6], 'FaceAlpha',0.25,'EdgeColor','none');
        plot(ax, [c-0.3 c+0.3], [nmd(c) nmd(c)], 'k--','LineWidth',2.0);
    end
    errorbar(ax, x, obs, obs-lo, hi-obs, 'LineStyle','none','Color','k', ...
             'LineWidth',2.0,'CapSize',14);
    scatter(ax, x, obs, 240, 'filled','MarkerFaceColor',pal(6,:), ...
            'MarkerEdgeColor','k','LineWidth',1.2);
    for c = x
        text(ax, c, hi(c), sprintf('  %.4f\n  p = %.3f', obs(c), pval(c)), ...
             'FontName',C.fontName,'FontSize',C.fsAnnot, ...
             'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
    set(ax,'XTick',x,'XTickLabel',conds(:,2)); xlim(ax,[0.4 nc+0.6]);
    ylim(ax,[0 max(hi)*1.35]);
    mal_axes(ax,'Utility gain does not exceed a permutation null','', ...
             'mean utility gain vs one-size-fits-all');
    mal_label(ax,{'Grey band = molecular-permutation null (95% range);', ...
                  'dashed line = null median. Neither objective''s gain', ...
                  'is distinguishable from permuted covariates.'},'northeast');
    mal_export(fig,'F12b_gain_vs_null');
end

% ======================= integration benchmark (c, d) =====================
fb = fullfile(P.results,'benchmark_cv_revised.json');
if ~isfile(fb)
    warning('F12c/d skipped: %s not found', fb);
    fprintf('F12 done.\n'); return
end
B = mal_readjson(fb);
st = B.strategies; sn = fieldnames(st);
nice = strrep(strrep(sn,'PROFILE_Net','PROFILE-Net'),'_','\_');
ir = nan(numel(sn),1); irlo = ir; irhi = ir; ex = ir; ut = ir;
for i = 1:numel(sn)
    m = st.(sn{i});
    if isfield(m,'interaction_rank_mean'), ir(i) = m.interaction_rank_mean; end
    if isfield(m,'interaction_rank_95ci'), ci = m.interaction_rank_95ci; irlo(i)=ci(1); irhi(i)=ci(2); end
    if isfield(m,'exposure_r2'), ex(i) = m.exposure_r2; end
    if isfield(m,'utility_r2'),  ut(i) = m.utility_r2; end
end

% ---- (c) predictive accuracy by strategy ---------------------------------
fig = mal_figure(34,20); ax = axes(fig); hold(ax,'on');
Y = [ex ut];
hb = bar(ax, Y,'grouped','EdgeColor','k','LineWidth',1.4);
hb(1).FaceColor = pal(1,:); hb(2).FaceColor = pal(2,:);
set(ax,'XTick',1:numel(sn),'XTickLabel',nice); ax.XTickLabelRotation = 18;
ylim(ax,[0 1.12]);
legend(ax,{'exposure R^2','utility R^2'},'Location','northoutside','Orientation','horizontal');
mal_axes(ax,'Predictive accuracy by integration strategy','','R^2');
mal_export(fig,'F12c_strategy_accuracy');

% ---- (d) interaction captured: the deciding metric -----------------------
fig = mal_figure(34,20); ax = axes(fig); hold(ax,'on');
bb = barh(ax, ir, 0.6,'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
for i = 1:numel(ir)
    if ir(i) < 0, bb.CData(i,:) = pal(8,:); else, bb.CData(i,:) = pal(1,:); end
end
errorbar(ax, ir, 1:numel(ir), ir-irlo, irhi-ir, 'horizontal', ...
         'LineStyle','none','Color','k','LineWidth',1.8,'CapSize',12);
xline(ax,0,'k-','LineWidth',1.4);
set(ax,'YTick',1:numel(sn),'YTickLabel',nice);
for i = 1:numel(ir)
    xa = irhi(i) + 0.04; if isnan(xa), xa = ir(i)+0.04; end
    text(ax, xa, i, sprintf('%.3f', ir(i)), 'FontName',C.fontName, ...
         'FontSize',C.fsAnnot,'VerticalAlignment','middle');
end
xlim(ax,[min(0,min(irlo))-0.10, max(irhi)+0.28]);
mal_axes(ax,'Patient x formulation interaction captured', ...
         'interaction-residual rank correlation','');
mal_label(ax,{'Late, additive integration (negative, red) cannot represent', ...
              'a patient-by-formulation interaction; the joint strategies', ...
              'recover it and tie after Holm correction.'},'southeast');
mal_export(fig,'F12d_interaction_metric');
fprintf('F12 done.\n');
end
