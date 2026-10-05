function F07_pk_ai()
%F07_PK_AI  Artificial-intelligence pharmacokinetics: exposure and cure.
%   Subfigures: F07a_exposure_r2, F07b_exposure_drivers, F07c_cure_auroc
C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);

fp = fullfile(P.results,'pk_ai_results.json');
if ~isfile(fp), warning('F07 skipped: %s not found', fp); return; end
d = mal_readjson(fp);

% (a) exposure R2 by model
mods = {'exposure_allometric','allometric (body size only)'; ...
        'exposure_ai_gbm','AI: gradient boosting'; ...
        'exposure_ai_neural','AI: neural network'};
r2 = nan(size(mods,1),1);
for i = 1:size(mods,1)
    if isfield(d,mods{i,1}), r2(i) = d.(mods{i,1}).r2; end
end
fig = mal_figure(30,20); ax = axes(fig); hold(ax,'on');
b = bar(ax, r2, 0.55, 'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
b.CData = [pal(11,:); pal(1,:); pal(3,:)];
set(ax,'XTick',1:numel(r2),'XTickLabel',mods(:,2));
ax.XTickLabelRotation = 20;
ylim(ax,[0 max(r2)*1.35]);
for i = 1:numel(r2)
    text(ax,i,r2(i),sprintf('%.3f',r2(i)),'FontName',C.fontName,'FontSize',C.fsAnnot, ...
         'HorizontalAlignment','center','VerticalAlignment','bottom');
end
mal_axes(ax,'Day-7 exposure prediction','','R^2 (held-out)');
mal_export(fig,'F07a_exposure_r2');

% (b) drivers of exposure
if isfield(d,'exposure_feature_importance')
    S = d.exposure_feature_importance; fn = fieldnames(S);
    v = cellfun(@(k) S.(k), fn);
    [v,ix] = sort(v,'ascend'); lbl = strrep(fn(ix),'_','\_');
    fig = mal_figure(30,20); ax = axes(fig); hold(ax,'on');
    bb = barh(ax, v, 0.6, 'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
    for i = 1:numel(v)
        if contains(fn{ix(i)},'host'), bb.CData(i,:) = pal(3,:); else, bb.CData(i,:) = pal(1,:); end
    end
    set(ax,'YTick',1:numel(v),'YTickLabel',lbl);
    mal_axes(ax,'Drivers of exposure','relative importance','');
    mal_label(ax,'green = host molecular state','southeast');
    mal_export(fig,'F07b_exposure_drivers');
end

% (c) cure discrimination
lab = {}; val = [];
if isfield(d,'cure_auroc_ai'),       lab{end+1}='AI (all features)';        val(end+1)=d.cure_auroc_ai; end
if isfield(d,'cure_auroc_day7only'), lab{end+1}='day-7 concentration only'; val(end+1)=d.cure_auroc_day7only; end
if ~isempty(val)
    fig = mal_figure(28,20); ax = axes(fig); hold(ax,'on');
    b = bar(ax, val(:), 0.5,'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
    b.CData = [pal(1,:); pal(2,:)];
    yline(ax,0.5,'k--','LineWidth',1.6);
    set(ax,'XTick',1:numel(val),'XTickLabel',lab); ylim(ax,[0 1.15]);
    ax.XTickLabelRotation = 12;
    for i = 1:numel(val)
        text(ax,i,val(i),sprintf('%.3f',val(i)),'FontName',C.fontName,'FontSize',C.fsAnnot, ...
             'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
    mal_axes(ax,'Cure discrimination','','AUROC');
    mal_label(ax,'dashed line = chance','southwest');
    mal_export(fig,'F07c_cure_auroc');
end
fprintf('F07 done.\n');
end
