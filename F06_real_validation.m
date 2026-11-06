function F06_real_validation()
%F06_REAL_VALIDATION  Real-cohort validation, reported against proper controls.
%   Subfigures: F06a_cluster_balance, F06b_reference_null,
%               F06c_severity_ari, F06d_what_clusters_track
%
%   Silhouette on its own is not evidence of structure. A clustering that
%   isolates one sample from 55 scores highly by construction, and a silhouette
%   measured inside a learned latent is circular. These panels therefore report
%   the diagnostics that ARE informative: cluster balance (normalised entropy of
%   the cluster sizes), bootstrap stability against a cluster-free reference,
%   agreement with clinical severity, and what the clusters actually track.
C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);

blocks  = {'host_only','host RNA only'; 'dual_full','host + parasite'; ...
           'dual_parasite_informed_subset','parasite-informed subset'};
cohorts = {'real_mali','Mali'; 'external_gambia','Gambia'};
% Final clustering uses the deduplicated Gambia set (45 unique samples) as the
% primary external cohort, matching the manuscript; the QC panel (a) still reads
% the per-cohort QC files, which are unaffected by deduplication.
clustcoh = {'real_mali','Mali'; 'external_gambia_dedup','Gambia'};
gambia_clustfile = 'final_clustering_external_gambia_dedup.json';

% ---------- (a) cluster balance: the degeneracy diagnostic -----------------
B = nan(size(blocks,1), size(cohorts,1));
for c = 1:size(cohorts,1)
    fp = fullfile(P.results, sprintf('dualrna_qc_%s.json', cohorts{c,1}));
    if ~isfile(fp), continue; end
    d = mal_readjson(fp);
    for i = 1:size(blocks,1)
        if isfield(d.robustness, blocks{i,1}) && isfield(d.robustness.(blocks{i,1}),'BLISS')
            B(i,c) = d.robustness.(blocks{i,1}).BLISS.cluster_balance;
        end
    end
end
if any(~isnan(B(:)))
    fig = mal_figure(32,20); ax = axes(fig); hold(ax,'on');
    b = bar(ax, B, 'grouped','EdgeColor','k','LineWidth',1.4);
    b(1).FaceColor = pal(1,:); b(2).FaceColor = pal(2,:);
    yline(ax, 0.5, 'r--','LineWidth',2.2);
    set(ax,'XTick',1:size(blocks,1),'XTickLabel',blocks(:,2));
    ax.XTickLabelRotation = 12; ylim(ax,[0 1.15]);
    legend(ax, cohorts(:,2),'Location','northoutside','Orientation','horizontal');
    mal_axes(ax,'Cluster balance after library-size correction','','balance (normalised entropy)');
    for i = 1:size(B,1)
        for jj = 1:size(B,2)
            if ~isnan(B(i,jj))
                txt = sprintf('%.3f', B(i,jj));
                if B(i,jj) < 0.5, txt = sprintf('%s *', txt); end
                text(ax, b(jj).XEndPoints(i), B(i,jj)+0.02, txt, 'FontName',C.fontName, ...
                     'FontSize',C.fsAnnot,'HorizontalAlignment','center', ...
                     'VerticalAlignment','bottom');
            end
        end
    end
    mal_label(ax,{'* degenerate (balance < 0.5). After correction the Malian', ...
                  'solutions and the Gambian host-only solution are balanced;', ...
                  'adding the now-abundant parasite block collapses the Gambian', ...
                  'split onto a few high-parasite samples. Dashed line =', ...
                  'degeneracy threshold; reproducibility is tested in panel (b).'},'southwest');
    mal_export(fig,'F06a_cluster_balance');
end

% ---------- (b) stability against a cluster-free reference ----------------
obs = []; nullmed = []; nullp95 = []; pv = []; lbl = {};
for c = 1:size(clustcoh,1)
    fp = fullfile(P.results, sprintf('final_clustering_%s.json', clustcoh{c,1}));
    if ~isfile(fp), continue; end
    d = mal_readjson(fp); n = d.reference_null;
    obs(end+1)     = d.stability;                    %#ok<AGROW>
    nullmed(end+1) = n.stability_null_median;        %#ok<AGROW>
    nullp95(end+1) = n.stability_null_p95;           %#ok<AGROW>
    pv(end+1)      = n.stability_p;                  %#ok<AGROW>
    lbl{end+1}     = clustcoh{c,2};                  %#ok<AGROW>
end
if ~isempty(obs)
    fig = mal_figure(32,20); ax = axes(fig); hold(ax,'on');
    x = 1:numel(obs);
    for i = x
        patch(ax, [i-0.32 i+0.32 i+0.32 i-0.32], ...
                  [0 0 nullp95(i) nullp95(i)], [0.6 0.6 0.6], ...
                  'FaceAlpha',0.25,'EdgeColor','none');
        plot(ax, [i-0.32 i+0.32], [nullmed(i) nullmed(i)], 'k--','LineWidth',2.0);
    end
    sc = scatter(ax, x, obs, 260, 'filled','MarkerFaceColor',pal(6,:), ...
                 'MarkerEdgeColor','k','LineWidth',1.2);
    for i = x
        text(ax, i, obs(i)+0.05, sprintf('%.3f\np = %.3f', obs(i), pv(i)), ...
             'FontName',C.fontName,'FontSize',C.fsAnnot, ...
             'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
    set(ax,'XTick',x,'XTickLabel',lbl); xlim(ax,[0.4 numel(obs)+0.6]); ylim(ax,[0 1.25]);
    mal_axes(ax,'Bootstrap stability versus a cluster-free reference','','stability');
    mal_label(ax,{'Grey band = reference null up to the 95th percentile;','dashed line = null median. Neither cohort exceeds it.'},'northwest');
    mal_export(fig,'F06b_reference_null');
end

% ---------- (c) agreement with clinical severity --------------------------
fp = fullfile(P.results, gambia_clustfile);
if isfile(fp)
    d = mal_readjson(fp);
    if isfield(d,'severity') && ~isempty(d.severity) && isstruct(d.severity)
        s = d.severity;
        fig = mal_figure(28,20); ax = axes(fig); hold(ax,'on');
        bb = bar(ax, s.ari, 0.35,'FaceColor',pal(6,:),'EdgeColor','k','LineWidth',1.4);
        yline(ax, 0, 'k-','LineWidth',1.6);
        set(ax,'XTick',1,'XTickLabel',{sprintf('Gambia\n(n = %d labelled)', s.n_labelled)});
        ylim(ax,[-0.35 1.05]); xlim(ax,[0.5 1.5]);
        text(ax, 1, s.ari - 0.05, sprintf('ARI = %.4f\nFisher p = %.3f', s.ari, s.fisher_p), ...
             'FontName',C.fontName,'FontSize',C.fsAnnot, ...
             'HorizontalAlignment','center','VerticalAlignment','top');
        mal_axes(ax,'Agreement with cerebral vs uncomplicated malaria','','ARI');
        mal_label(ax,{'ARI of zero means the clusters carry no','information about clinical severity'},'northwest');
        mal_export(fig,'F06c_severity_ari');
    end
end

% ---------- (d) what the clusters actually track --------------------------
if isfile(fp)
    d = mal_readjson(fp);
    cv = d.covariates;
    if ~isempty(cv)
        n = numel(cv);
        eff = zeros(n,1); sig = false(n,1); nm = cell(n,1);
        for i = 1:n
            r = cv(i); if iscell(cv), r = cv{i}; end
            eff(i) = r.effect; nm{i} = strrep(r.covariate,'_','\_');
            sig(i) = logical(r.significant);
        end
        [eff, ix] = sort(eff,'ascend'); nm = nm(ix); sig = sig(ix);
        fig = mal_figure(32,22); ax = axes(fig); hold(ax,'on');
        bb = barh(ax, eff, 0.65,'FaceColor','flat','EdgeColor','k','LineWidth',1.2);
        for i = 1:numel(eff)
            if sig(i), bb.CData(i,:) = pal(6,:); else, bb.CData(i,:) = pal(11,:); end
        end
        set(ax,'YTick',1:numel(eff),'YTickLabel',nm); xlim(ax,[0 1.12]);
        mal_axes(ax,'What the Gambia clusters track','effect size','');
        mal_label(ax,{'Coloured = significant after FDR correction.', ...
                      'Host PC1 is the clustering axis itself; among the', ...
                      'remaining covariates the significant ones are parasite', ...
                      'transcriptional components, not read-fraction burden', ...
                      'and not clinical severity.'},'southeast');
        mal_export(fig,'F06d_what_clusters_track');
    end
end
fprintf('F06 done.\n');
end
