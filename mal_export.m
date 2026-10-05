function mal_export(fig, name)
%MAL_EXPORT  Write one subfigure as .fig, .png (1200 dpi) and .pdf (vector).
%
%   Order matters here. The editable .fig is saved FIRST, before any cosmetic
%   pass touches the figure, so that even if something later goes wrong the
%   figure is never lost and can be reopened and re-exported by hand.
%
%   If a figure still fails, set  C.applyMinus = false  in mal_cfg.m and rerun.
%   That disables the U+2212 minus-sign pass, which is the only step that
%   modifies the figure before export. If the failure disappears, that pass is
%   the cause and the figure is otherwise fine.

C = mal_cfg(); P = mal_paths();
if ~isfolder(P.out), mkdir(P.out); end
base = fullfile(P.out, name);

% ---- 0. the handle must be alive before we do anything ---------------------
if isempty(fig) || ~all(isgraphics(fig, 'figure'))
    warning('mal_export:deadOnEntry', ...
        ['%s: the figure handle was ALREADY invalid on entry to mal_export. ' ...
         'It was destroyed by the plotting code above, not by the export.'], name);
    return
end

% ---- 1. save the editable figure first, while it is certainly intact -------
figSaved = false;
try
    savefig(fig, [base '.fig']);
    figSaved = true;
catch ME
    warning('mal_export:nofig', '%s: could not write .fig (%s)', name, ME.message);
end

% ---- 2. cosmetic passes, each unable to take the figure down with it -------
if C.applyMinus
    try
        mal_minus(fig);
    catch ME
        warning('mal_minus:threw', '%s: minus-sign pass failed (%s)', name, ME.message);
    end
    if ~all(isgraphics(fig, 'figure'))
        warning('mal_export:killedByMinus', ...
            ['%s: the figure became invalid DURING the minus-sign pass. ' ...
             'Set C.applyMinus = false in mal_cfg.m to skip it. ' ...
             'The .fig was saved beforehand and is intact.'], name);
        return
    end
end

try                                   % hide the interactive axes toolbar
    axs = findall(fig, 'Type', 'axes');
    for a = 1:numel(axs)
        try, axs(a).Toolbar.Visible = 'off'; catch, end
    end
catch
end
try, drawnow; catch, end

if ~all(isgraphics(fig, 'figure'))
    warning('mal_export:killedByDrawnow', ...
        '%s: figure became invalid during drawnow/toolbar step.', name);
    return
end

% ---- 3. raster and vector -------------------------------------------------
usedFallback = false; wrote = false;
try
    exportgraphics(fig, [base '.png'], 'Resolution', C.dpi, 'BackgroundColor', C.background);
    exportgraphics(fig, [base '.pdf'], 'ContentType', 'vector', 'BackgroundColor', C.background);
    wrote = true;
catch ME
    usedFallback = true;
    warning('mal_export:fallback', '%s: exportgraphics failed (%s); using print().', ...
            name, ME.message);
    try
        print(fig, [base '.png'], '-dpng', sprintf('-r%d', C.dpi));
        print(fig, [base '.pdf'], '-dpdf', '-painters');
        wrote = true;
    catch ME2
        warning('mal_export:failed', '%s: could not write raster/vector (%s).', ...
                name, ME2.message);
    end
end

if wrote && ~usedFallback
    fprintf('    %s  ->  .png (%d dpi) .pdf (vector) .fig\n', name, C.dpi);
elseif wrote
    fprintf('    %s  ->  .png .pdf .fig  [print() fallback]\n', name);
elseif figSaved
    fprintf('    %s  ->  .fig ONLY (raster/vector failed; reopen and export by hand)\n', name);
else
    fprintf('    %s  ->  NOT WRITTEN\n', name);
end

try, close(fig); catch, end
end
