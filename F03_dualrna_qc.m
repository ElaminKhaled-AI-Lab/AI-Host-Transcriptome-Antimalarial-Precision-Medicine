function F03_dualrna_qc()
%F03_DUALRNA_QC  Dual host-parasite RNA processing and quality control.
%   Subfigures: F03a_parasite_read_fraction, F03b_transcripts_detected,
%               F03c_samples_above_threshold
C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);

cohorts = {'real_mali','Mali (discovery)'; 'external_gambia','Gambia (external)'};
Q = struct([]); keep = {};
for i = 1:size(cohorts,1)
    fp = fullfile(P.results, sprintf('dualrna_qc_%s.json', cohorts{i,1}));
    if ~isfile(fp), warning('missing %s', fp); continue; end
    d = mal_readjson(fp);
    q = d.qc; pf = q.parasite_read_fraction;
    s.name   = cohorts{i,2};
    s.n      = q.n_samples;
    s.med    = 100*pf.median;  s.max = 100*pf.max;  s.min = 100*pf.min;
    s.hostTx = q.host_transcripts_detected_median;
    s.paraTx = q.parasite_transcripts_detected_median;
    s.above  = [getByPrefix(pf,'n_above_0_05'), getByPrefix(pf,'n_above_0_1'), ...
                getByPrefix(pf,'n_above_0_5')];
    if isempty(Q), Q = s; else, Q(end+1) = s; end %#ok<AGROW>
    keep{end+1} = s.name; %#ok<AGROW>
end
if isempty(Q), warning('F03 skipped: no QC json found'); return; end
nc = numel(Q);

% (a) parasite read fraction (median with min-max range)
fig = mal_figure(26,19); ax = axes(fig); hold(ax,'on');
med = [Q.med]; lo = [Q.min]; hi = [Q.max];
b = bar(ax, 1:nc, med, 0.55, 'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
for i = 1:nc, b.CData(i,:) = pal(i,:); end
errorbar(ax, 1:nc, med, med-lo, hi-med, 'k','LineStyle','none','LineWidth',2.0,'CapSize',14);
set(ax,'XTick',1:nc,'XTickLabel',{Q.name});
mal_axes(ax,'Parasite-read fraction','','parasite reads (% of total)');
for i = 1:nc
    text(ax,i,hi(i),sprintf('  max %.3f%%',hi(i)),'FontName',C.fontName, ...
         'FontSize',C.fsAnnot,'VerticalAlignment','bottom','HorizontalAlignment','center');
end
mal_label(ax,'bars = median, whiskers = min-max','northwest');
mal_export(fig,'F03a_parasite_read_fraction');

% (b) transcripts detected
fig = mal_figure(28,19); ax = axes(fig); hold(ax,'on');
M = [[Q.hostTx]', [Q.paraTx]'];
b = bar(ax, M, 'grouped','EdgeColor','k','LineWidth',1.4);
b(1).FaceColor = pal(1,:); b(2).FaceColor = pal(6,:);
set(ax,'XTick',1:nc,'XTickLabel',{Q.name},'YScale','log');
legend(ax,{'host transcripts','parasite transcripts'},'Location','northoutside', ...
       'Orientation','horizontal');
mal_axes(ax,'Transcripts detected (median per sample)','','transcripts (log scale)');
mal_export(fig,'F03b_transcripts_detected');

% (c) samples above parasite-signal thresholds
fig = mal_figure(28,19); ax = axes(fig); hold(ax,'on');
A = reshape([Q.above],3,nc)';
b = bar(ax, A, 'grouped','EdgeColor','k','LineWidth',1.4);
cols = [pal(3,:); pal(2,:); pal(6,:)];
for i = 1:3, b(i).FaceColor = cols(i,:); end
set(ax,'XTick',1:nc,'XTickLabel',{Q.name});
legend(ax,{'> 0.05%','> 0.1%','> 0.5%'},'Location','northoutside','Orientation','horizontal');
mal_axes(ax,'Samples exceeding parasite-signal thresholds','','number of samples');
for i = 1:nc
    text(ax,i,max(A(i,:)),sprintf('n = %d',Q(i).n),'FontName',C.fontName, ...
        'FontSize',C.fsAnnot,'VerticalAlignment','bottom','HorizontalAlignment','center');
end
mal_export(fig,'F03c_samples_above_threshold');
fprintf('F03 done.\n');
end

function v = getByPrefix(s, pre)
v = NaN; fn = fieldnames(s);
for i = 1:numel(fn)
    if startsWith(fn{i}, pre), v = s.(fn{i}); return; end
end
end
