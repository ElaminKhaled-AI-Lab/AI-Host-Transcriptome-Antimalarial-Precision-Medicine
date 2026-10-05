function C = mal_cfg()
%MAL_CFG  Central style configuration for every Malaria_Project figure.
%   Edit here once and every figure script follows.
C.fontName    = 'Arial';
C.fsAxis      = 25;    % all in-axes text (ticks, labels) -- minimum requested
C.fsTitle     = 30;    % panel headings
C.fsLegend    = 25;    % legends kept at the same minimum; placed outside axes
C.fsAnnot     = 22;    % small in-panel annotations only (n = ..., thresholds)
C.dpi         = 1200;  % PNG export resolution
C.lineWidth   = 2.0;
C.axisWidth   = 1.4;
C.markerSize  = 90;
C.figW        = 26;    % default figure width  (cm)
C.figH        = 19;    % default figure height (cm)
C.gridAlpha   = 0.15;
C.showGrid    = false;    % false = no grid lines (journal-clean). true = grid on.
C.background  = 'white';  % 'white' for print, or 'none' for a TRANSPARENT export
C.applyMinus  = true;     % U+2212 minus pass. Set false if a figure fails to
                          % export - it is the only step that alters the figure.
C.showBands   = true;     % shaded reference bands (Kupffer window, null band).
                          % These carry meaning - set false only if you must.
end
