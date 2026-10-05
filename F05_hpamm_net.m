function F05_hpamm_net()
%F05_HPAMM_NET  HPAMM-Net multitask performance and ablations.
%   Subfigures: F05a_multitask_auroc, F05b_multitask_mae, F05c_ablations
C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);

L = dir(fullfile(P.results,'hpamm_multitask_seed*.json'));
if isempty(L), warning('F05a/b skipped: no hpamm_multitask_seed*.json'); else
    tasksA = {'resistance','cure','recrudescence','toxicity'};
    tasksM = {'pk_auc','clearance'};
    nA = numel(tasksA); nM = numel(tasksM); ns = numel(L);
    A = nan(ns,nA,2); Mv = nan(ns,nM,2);
    for s = 1:ns
        d = mal_readjson(fullfile(L(s).folder,L(s).name));
        for t = 1:nA
            if isfield(d,'baseline_xgb') && isfield(d.baseline_xgb,tasksA{t})
                A(s,t,1) = d.baseline_xgb.(tasksA{t}).auroc; end
            if isfield(d,'hpamm_full') && isfield(d.hpamm_full,tasksA{t})
                A(s,t,2) = d.hpamm_full.(tasksA{t}).auroc; end
        end
        for t = 1:nM
            if isfield(d,'baseline_xgb') && isfield(d.baseline_xgb,tasksM{t})
                Mv(s,t,1) = d.baseline_xgb.(tasksM{t}).mae; end
            if isfield(d,'hpamm_full') && isfield(d.hpamm_full,tasksM{t})
                Mv(s,t,2) = d.hpamm_full.(tasksM{t}).mae; end
        end
    end
    % (a) classification AUROC
    fig = mal_figure(32,20); ax = axes(fig); hold(ax,'on');
    mu = squeeze(mean(A,1,'omitnan')); sd = squeeze(std(A,0,1,'omitnan'));
    b = bar(ax, mu, 'grouped','EdgeColor','k','LineWidth',1.4);
    b(1).FaceColor = pal(11,:); b(2).FaceColor = pal(3,:);
    for jj = 1:2
        errorbar(ax, b(jj).XEndPoints, mu(:,jj), sd(:,jj),'k','LineStyle','none', ...
                 'LineWidth',1.8,'CapSize',12);
    end
    yline(ax,0.5,'k--','LineWidth',1.6);
    set(ax,'XTick',1:nA,'XTickLabel',tasksA); ylim(ax,[0.4 0.85]);
    legend(ax,{'baseline (gradient boosting)','HPAMM-Net'},'Location','northoutside', ...
           'Orientation','horizontal');
    mal_axes(ax,'Multitask discrimination','','AUROC');
    mal_label(ax,sprintf('mean \\pm SD over %d seeds; dashed line = chance',ns),'southwest');
    mal_export(fig,'F05a_multitask_auroc');

    % (b) regression MAE
    fig = mal_figure(28,20); ax = axes(fig); hold(ax,'on');
    mu = squeeze(mean(Mv,1,'omitnan')); sd = squeeze(std(Mv,0,1,'omitnan'));
    if isvector(mu), mu = reshape(mu,nM,2); sd = reshape(sd,nM,2); end
    b = bar(ax, mu, 'grouped','EdgeColor','k','LineWidth',1.4);
    b(1).FaceColor = pal(11,:); b(2).FaceColor = pal(3,:);
    for jj = 1:2
        errorbar(ax, b(jj).XEndPoints, mu(:,jj), sd(:,jj),'k','LineStyle','none', ...
                 'LineWidth',1.8,'CapSize',12);
    end
    set(ax,'XTick',1:nM,'XTickLabel',{'PK AUC','parasite clearance'});
    legend(ax,{'baseline','HPAMM-Net'},'Location','northoutside','Orientation','horizontal');
    mal_axes(ax,'Multitask regression error','','mean absolute error (lower is better)');
    mal_export(fig,'F05b_multitask_mae');
end

% (c) ablations
fp = fullfile(P.results,'hpamm_ablations.json');
if isfile(fp)
    d = mal_readjson(fp); fn = fieldnames(d);
    val = nan(numel(fn),1);
    for i = 1:numel(fn)
        if isfield(d.(fn{i}),'resistance_auroc'), val(i) = d.(fn{i}).resistance_auroc; end
    end
    [val,ix] = sort(val,'descend'); lbl = strrep(fn(ix),'_','\_');
    fig = mal_figure(30,20); ax = axes(fig); hold(ax,'on');
    bb = barh(ax, val, 0.6, 'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
    for i = 1:numel(val)
        if strcmp(fn{ix(i)},'full'), bb.CData(i,:) = pal(3,:); else, bb.CData(i,:) = pal(1,:); end
    end
    xline(ax,0.5,'k--','LineWidth',1.6);
    set(ax,'YTick',1:numel(val),'YTickLabel',lbl); xlim(ax,[0.4 0.85]);
    mal_axes(ax,'Ablation: parasite resistance AUROC','AUROC','');
    mal_label(ax,'dashed line = chance','southeast');
    mal_export(fig,'F05c_ablations');
end
fprintf('F05 done.\n');
end
