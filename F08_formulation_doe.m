function F08_formulation_doe()
%F08_FORMULATION_DOE  Design of experiments: 3^3 factorial main effects and RSM.
%   Subfigures: F08a..d main effects, F08e RSM coefficients
C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);

fp = fullfile(P.tables,'nano_dds_dissolution_parameters.csv');
if ~isfile(fp), warning('F08 skipped: %s not found', fp); return; end
T = readtable(fp,'Delimiter',',','ReadVariableNames',true,'VariableNamingRule','preserve');

factors = {'A_ratio','polymer:drug ratio (w/w)'; ...
           'B_stabilizer_pct','stabiliser (% w/v)'; ...
           'C_energy','homogenisation energy'};
resp = {'size_nm','particle size (nm)','F08a_main_effects_size'; ...
        'ee_pct','entrapment efficiency (%)','F08b_main_effects_ee'; ...
        'DE_pct','dissolution efficiency (%)','F08c_main_effects_DE'; ...
        'liver_targeting_index','liver-targeting index','F08d_main_effects_liver'};

for r = 1:size(resp,1)
    if ~any(strcmp(T.Properties.VariableNames,resp{r,1})), continue; end
    fig = mal_figure(30,20); ax = axes(fig); hold(ax,'on');
    h = gobjects(size(factors,1),1);
    for k = 1:size(factors,1)
        fv = T.(factors{k,1});
        [g,lv] = findgroups(fv);
        mu = splitapply(@(x) mean(x,'omitnan'), T.(resp{r,1}), g);
        sd = splitapply(@(x) std(x,0,'omitnan'),  T.(resp{r,1}), g);
        xn = (lv - min(lv)) / max(range(lv),eps) * 2 - 1;   % code to -1..+1
        h(k) = errorbar(ax, xn, mu, sd, '-o','Color',pal(k,:), ...
            'MarkerFaceColor',pal(k,:),'MarkerSize',14,'LineWidth',C.lineWidth,'CapSize',12);
    end
    set(ax,'XTick',[-1 0 1],'XTickLabel',{'low','centre','high'});
    xlim(ax,[-1.35 1.35]);
    legend(ax,h,factors(:,2),'Location','northoutside','Orientation','horizontal');
    mal_axes(ax, sprintf('Main effects: %s', resp{r,2}),'coded factor level', resp{r,2});
    mal_label(ax,'mean \pm SD over all polymers, drugs and replicates','southwest');
    mal_export(fig, resp{r,3});
end

% (e) RSM coefficients from the R analysis
fr = fullfile(P.tables,'rsm_anova_size_nm.csv');
if isfile(fr)
    R = readtable(fr,'Delimiter',',','ReadVariableNames',true);
    vn = R.Properties.VariableNames;
    est = R.(vn{1}); trm = string(R.(vn{end}));
    pcol = find(contains(vn,'Pr'),1);
    pv = nan(size(est)); if ~isempty(pcol), pv = R.(vn{pcol}); end
    keep = ~strcmpi(trm,'(Intercept)');
    est = est(keep); trm = trm(keep); pv = pv(keep);
    fig = mal_figure(30,20); ax = axes(fig); hold(ax,'on');
    bb = barh(ax, est, 0.6,'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
    for i = 1:numel(est)
        if ~isnan(pv(i)) && pv(i) < 0.05, bb.CData(i,:) = pal(1,:);
        else,                             bb.CData(i,:) = pal(11,:); end
    end
    set(ax,'YTick',1:numel(est),'YTickLabel',strrep(cellstr(trm),'_','\_'));
    xline(ax,0,'k-','LineWidth',1.4);
    mal_axes(ax,'Response surface: particle size','coefficient estimate','model term');
    mal_label(ax,'blue = p < 0.05 (R response-surface fit)','southeast');
    mal_export(fig,'F08e_rsm_coefficients_size');
end
fprintf('F08 done.\n');
end
