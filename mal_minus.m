function mal_minus(fig)
%MAL_MINUS  Convert minus signs to the typographic minus (U+2212) for MDPI.
%
%   MDPI copy-editing asks that a NEGATIVE SIGN be the true minus character
%   "−" (U+2212), not the ASCII hyphen-minus "-" (U+002D). Applied to every
%   axes, tick label, title, axis label, text annotation, legend entry and
%   colour-bar in the figure.
%
%   IMPORTANT: hyphens inside words are left alone. "Korsmeyer-Peppas",
%   "liver-targeting", "one-size-fits-all" and "host_only" keep their ASCII
%   hyphen; only a hyphen that begins a number is converted. Numeric tick
%   labels are converted wholesale because they can contain nothing else.
%
%   Called automatically by mal_export, so every exported figure complies.

MINUS = char(8722);   % U+2212

if nargin < 1 || isempty(fig), fig = gcf; end
if isempty(fig) || ~isvalid(fig), return; end

% This is a purely cosmetic pass. It must NEVER prevent a figure from being
% exported, so every step is guarded and the whole body is wrapped.
try

% ---- axes: ticks, titles, labels ----------------------------------------
axs = findall(fig,'Type','axes');
for i = 1:numel(axs)
    ax = axs(i);
    if isempty(ax) || ~isvalid(ax), continue; end
    tg = '';
    try, tg = lower(char(ax.Tag)); catch, end
    if contains(tg,'legend') || contains(tg,'colorbar'), continue; end
    for d = {'X','Y','Z'}
        tickProp  = [d{1} 'Tick'];
        labelProp = [d{1} 'TickLabel'];
        try
            L = get(ax, labelProp);
            if isempty(L), continue; end
            set(ax, tickProp, get(ax, tickProp));      % freeze ticks first
            set(ax, labelProp, numericMinus(L, MINUS));
        catch
            continue
        end
    end
    try, setStr(ax.Title,  MINUS); catch, end
    try, setStr(ax.XLabel, MINUS); catch, end
    try, setStr(ax.YLabel, MINUS); catch, end
    try
        if isprop(ax,'ZLabel'), setStr(ax.ZLabel, MINUS); end
    catch
    end
end

% ---- free-standing text annotations -------------------------------------
txt = findall(fig,'Type','text');
for i = 1:numel(txt), setStr(txt(i), MINUS); end

% ---- legends -------------------------------------------------------------
% Deliberately NOT modified. Legend entries here are polymer, drug and strategy
% names, which contain no negative numbers, and reassigning a legend's String
% can invalidate a legend built from explicit handles.

% ---- colour bars ---------------------------------------------------------
cbs = findall(fig,'Type','ColorBar');
for i = 1:numel(cbs)
    try
        if ~isempty(cbs(i).TickLabels)
            cbs(i).TickLabels = numericMinus(cbs(i).TickLabels, MINUS);
        end
    catch
    end
    try
        setStr(cbs(i).Label, MINUS);
    catch
    end
end

catch ME
    warning('mal_minus:skipped', 'minus-sign pass skipped (%s)', ME.message);
end
end

% =========================================================================
function setStr(h, MINUS)
%SETSTR  Replace minus signs in a graphics object's String property.
if isempty(h) || ~isvalid(h) || ~isprop(h,'String'), return; end
try
    h.String = textMinus(h.String, MINUS);
catch
end
end

function out = numericMinus(L, MINUS)
%NUMERICMINUS  Wholesale hyphen -> minus (safe: tick labels are numeric only).
wasChar = ischar(L);
if wasChar, L = cellstr(L); end
if isstring(L), L = cellstr(L); end
out = cellfun(@(s) strrep(s, '-', MINUS), L, 'UniformOutput', false);
if wasChar && numel(out) == 1, out = out{1}; end
end

function out = textMinus(S, MINUS)
%TEXTMINUS  Convert only a hyphen that starts a NUMBER; keep word hyphens.
%   Matches a hyphen at the start of the string, or after whitespace, "(",
%   "[", "=", ",", ";" or ":", when immediately followed by a digit or a
%   decimal point. Everything else is untouched.
pat = '(^|[\s\(\[=,;:])-(?=[\d\.])';
rep = ['$1' MINUS];
if isempty(S), out = S; return; end
if ischar(S)
    out = regexprep(S, pat, rep);
elseif isstring(S)
    out = arrayfun(@(s) string(regexprep(char(s), pat, rep)), S);
elseif iscell(S)
    out = cellfun(@(s) localApply(s, pat, rep), S, 'UniformOutput', false);
else
    out = S;
end
end

function s = localApply(s, pat, rep)
if ischar(s)
    s = regexprep(s, pat, rep);
elseif isstring(s)
    s = string(regexprep(char(s), pat, rep));
end
end
