function F11_optimisation_validation()
%F11_OPTIMISATION_VALIDATION  Formulation optimisation, stability and validation.
%   Subfigures: F11a_polymer_desirability, F11b_design_space,
%               F11c_stability, F11d_literature_validation
C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);
[cmap, order] = mal_polymer_colors();

% (a) polymer ranking by desirability
fp = fullfile(P.results,'nano_dds_optimization.json');
if isfile(fp)
    d = mal_readjson(fp);
    S = d.polymer_ranking_by_desirability; fn = fieldnames(S);
    v = cellfun(@(k) S.(k), fn);
    [v,ix] = sort(v,'ascend'); nm = fn(ix);
    fig = mal_figure(32,20); ax = axes(fig); hold(ax,'on');
    bb = barh(ax, v, 0.65,'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
    for i = 1:numel(v)
        if isKey(cmap, nm{i}), bb.CData(i,:) = cmap(nm{i}); else, bb.CData(i,:) = pal(11,:); end
    end
    set(ax,'YTick',1:numel(v),'YTickLabel',strrep(nm,'_','\_'));
    dr = 'lumefantrine'; if isfield(d,'drug'), dr = d.drug; end
    mal_axes(ax, sprintf('Polymer ranking - drug: %s', upper(dr)),'mean desirability','');
    for i = 1:numel(v)
        text(ax, v(i)+0.008, i, sprintf('%.3f',v(i)),'FontName',C.fontName, ...
             'FontSize',C.fsAnnot,'VerticalAlignment','middle');
    end
    xlim(ax,[0 max(v)*1.18]);
    mal_export(fig,'F11a_polymer_desirability');
end

% (b) design space: size vs EE coloured by liver targeting
fpd = fullfile(P.tables,'nano_dds_dissolution_parameters.csv');
if isfile(fpd)
    T = readtable(fpd,'Delimiter',',','ReadVariableNames',true,'VariableNamingRule','preserve');
    fig = mal_figure(32,20); ax = axes(fig); hold(ax,'on');
    yl = [min(T.ee_pct) max(T.ee_pct)];
    patch(ax,[100 200 200 100],[yl(1) yl(1) yl(2) yl(2)],[0.4 0.4 0.4], ...
          'FaceAlpha',0.14,'EdgeColor','none');
    scatter(ax, T.size_nm, T.ee_pct, 26, T.liver_targeting_index,'filled','MarkerFaceAlpha',0.5);
    colormap(ax,parula); cb = colorbar(ax);
    cb.Label.String = mal_titlecase('liver-targeting index');
    cb.Label.FontName = C.fontName; cb.Label.FontSize = C.fsAxis;
    ylim(ax,yl);
    mal_axes(ax,'Formulation design space','particle size (nm)','entrapment efficiency (%)');
    mal_label(ax,sprintf('n = %d formulations; band = Kupffer window',height(T)),'southeast');
    mal_export(fig,'F11b_design_space');
end

% (c) accelerated stability
fps = fullfile(P.results,'nano_dds_stability.json');
if isfile(fps)
    d = mal_readjson(fps);
    if isfield(d,'stability') && isfield(d.stability,'size_trajectory_40C_nm')
        y = d.stability.size_trajectory_40C_nm(:); x = (0:numel(y)-1)';
        fig = mal_figure(30,20); ax = axes(fig); hold(ax,'on');
        plot(ax,x,y,'-o','Color',pal(6,:),'MarkerFaceColor',pal(6,:), ...
             'MarkerSize',14,'LineWidth',C.lineWidth);
        yline(ax, y(1)*1.10,'k--','LineWidth',1.8);
        pol = ''; if isfield(d,'polymer'), pol = strrep(d.polymer,'_','\_'); end
        mal_axes(ax, sprintf('ICH accelerated stability (40 \\circC / 75%% RH)%s', ...
                 ternary(isempty(pol),'',[' - ' pol])),'months','particle size (nm)');
        sl = NaN;
        if isfield(d.stability,'predicted_shelf_life_25C_months')
            sl = d.stability.predicted_shelf_life_25C_months; end
        mal_label(ax, sprintf('dashed line = +10%% size\npredicted shelf life at 25 \\circC: %.1f months', sl),'northwest');
        mal_export(fig,'F11c_stability');
    end
end

% (d) literature validation
fpv = fullfile(P.tables,'nano_dds_validation.csv');
if isfile(fpv)
    V = readtable(fpv,'Delimiter',',','ReadVariableNames',true,'VariableNamingRule','preserve');
    % the column has been called design_space_overlaps and, in an earlier
    % version, within_range -- match on a pattern so either file works
    ok = mal_getcol(V, {'overlap','within','equivalent','pass'});
    if isempty(ok)
        warning('F11:noOverlapColumn', ...
            ['F11d skipped: %s has no overlap column.\n' ...
             '         Columns present: %s\n' ...
             '         Fix: python scripts/13_nano_dds/validate_and_report.py'], ...
            fpv, strjoin(V.Properties.VariableNames, ', '));
        ok = [];
    end
    if ~isempty(ok)
    if islogical(ok), ok = double(ok);
    elseif ~isnumeric(ok), ok = double(ismember(lower(string(ok)),["true","yes","1"]));
    end
    lbl = strcat(string(mal_getcol(V,{'anchor'}))," | ",string(mal_getcol(V,{'metric'})));
    fig = mal_figure(36,22); ax = axes(fig); hold(ax,'on');
    bb = barh(ax, ok, 0.6,'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
    for i = 1:numel(ok)
        if ok(i) > 0, bb.CData(i,:) = pal(3,:); else, bb.CData(i,:) = pal(6,:); end
    end
    set(ax,'YTick',1:numel(ok),'YTickLabel',strrep(cellstr(lbl),'_','\_'), ...
           'XTick',[0 1],'XTickLabel',{'outside','overlaps'});
    xlim(ax,[0 1.25]);
    mal_axes(ax, sprintf('Validation against published values (%d/%d overlap)', ...
             sum(ok>0), numel(ok)),'','');
    mal_export(fig,'F11d_literature_validation');
    end
end
fprintf('F11 done.\n');
end

function out = ternary(c,a,b)
if c, out = a; else, out = b; end
end
