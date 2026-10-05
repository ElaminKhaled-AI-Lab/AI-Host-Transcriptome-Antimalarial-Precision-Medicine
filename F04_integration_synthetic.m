function F04_integration_synthetic()
%F04_INTEGRATION_SYNTHETIC  Integration on synthetic data, compared honestly.
%   Subfigures: F04a_ari_synthetic, F04b_silhouette_is_circular
%
%   The adjusted Rand index (ARI) compares label partitions and is therefore
%   INDEPENDENT of the embedding, so it is the only fair comparison between a
%   method that learns its own latent space and one that does not. Silhouette is
%   not: BLISS is trained to separate clusters in its latent, so scoring the
%   separation there is circular. Both are shown, and the circularity is made
%   explicit rather than hidden.
C = mal_cfg(); P = mal_paths(); pal = mal_palette(11);

fp = fullfile(P.results,'integration_benchmark_seed11.json');
if ~isfile(fp), warning('F04 skipped: %s not found', fp); return; end
d = mal_readjson(fp);

% ---------- (a) ARI: the embedding-independent comparison ------------------
ari = [NaN NaN];
if isfield(d,'concat_kmeans') && isfield(d.concat_kmeans,'ari'), ari(1) = d.concat_kmeans.ari; end
if isfield(d,'BLISS') && isfield(d.BLISS,'ari'),                 ari(2) = d.BLISS.ari; end

fig = mal_figure(28,20); ax = axes(fig); hold(ax,'on');
b = bar(ax, ari(:), 0.5, 'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
b.CData = [pal(11,:); pal(1,:)];
set(ax,'XTick',1:2,'XTickLabel',{'concatenation','BLISS'});
ylim(ax,[0 1.05]);
for i = 1:2
    if ~isnan(ari(i))
        text(ax,i,ari(i),sprintf('%.3f',ari(i)),'FontName',C.fontName, ...
             'FontSize',C.fsAnnot,'HorizontalAlignment','center', ...
             'VerticalAlignment','bottom');
    end
end
mal_axes(ax,'Synthetic cohort: agreement with the planted subtype','','ARI');
mal_label(ax,{'ARI compares partitions, not geometry,','so it does not depend on the embedding'},'northwest');
mal_export(fig,'F04a_ari_synthetic');

% ---------- (b) why silhouette cannot be compared across embeddings --------
sil = [NaN NaN];
if isfield(d,'concat_kmeans'), sil(1) = d.concat_kmeans.silhouette; end
if isfield(d,'BLISS'),         sil(2) = d.BLISS.silhouette;         end
NOISE_BLISS = 0.843;   % measured: BLISS latent silhouette on pure-noise counts

fig = mal_figure(32,20); ax = axes(fig); hold(ax,'on');
b = bar(ax, sil(:), 0.5, 'FaceColor','flat','EdgeColor','k','LineWidth',1.4);
b.CData = [pal(11,:); pal(1,:)];
yline(ax, NOISE_BLISS, 'r--', 'LineWidth', 2.4);
set(ax,'XTick',1:2,'XTickLabel',{sprintf('concatenation\n(shared PC space)'), ...
                                 sprintf('BLISS\n(own learned latent)')});
ylim(ax,[0 1.05]); xlim(ax,[0.4 3.2]);
for i = 1:2
    if ~isnan(sil(i))
        text(ax,i,sil(i),sprintf('%.3f',sil(i)),'FontName',C.fontName, ...
             'FontSize',C.fsAnnot,'HorizontalAlignment','center', ...
             'VerticalAlignment','bottom');
    end
end
text(ax, 2.45, NOISE_BLISS, sprintf('  %.3f = BLISS latent\n  on PURE NOISE', NOISE_BLISS), ...
     'FontName',C.fontName,'FontSize',C.fsAnnot,'Color',[0.7 0 0], ...
     'VerticalAlignment','middle','HorizontalAlignment','left');
mal_axes(ax,'Silhouette is not comparable across embeddings','','silhouette');
mal_label(ax,{'BLISS is trained to separate clusters in its','latent, so scoring it there is circular'},'northwest');
mal_export(fig,'F04b_silhouette_is_circular');
fprintf('F04 done.\n');
end
