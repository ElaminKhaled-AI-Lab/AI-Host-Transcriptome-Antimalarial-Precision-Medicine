function out = mal_titlecase(s)
%MAL_TITLECASE  Title Case for axis labels, safe for units and symbols.
%
%   Capitalises the first letter of each significant word. Deliberately leaves
%   alone:
%     * anything inside parentheses  -- units and qualifiers stay as written,
%       so "(nm)", "(mV)", "(ng/mL)", "(lower is better)", "(held-out)" and
%       "(normalised entropy)" are untouched;
%     * tokens already containing a capital -- acronyms and unit symbols such
%       as ARI, AUROC, PDI, mV, ng/mL, R^2;
%     * LaTeX markup -- anything containing \ { } ^ or _ , e.g. t_{50};
%     * symbol-style tokens, a lowercase letter followed by a digit (t50, k0);
%     * minor words when they are not the first word: a, an, and, as, at, but,
%       by, for, from, in, into, nor, of, on, onto, or, over, per, the, to, up,
%       versus, via, vs, with.
%
%   Hyphenated compounds are capitalised throughout, so "one-size-fits-all"
%   becomes "One-Size-Fits-All" and "day-7" becomes "Day-7".
%
%   Applied automatically by mal_axes to the x and y labels.

if isempty(s), out = s; return; end
if ~ischar(s) && ~isstring(s), out = s; return; end
wasString = isstring(s);
s = char(s);

toks = strsplit(s, ' ');
depth = 0; first = true;
for i = 1:numel(toks)
    w = toks{i};
    if isempty(w), continue; end
    inParen = depth > 0 || startsWith(w, '(');
    if ~inParen
        toks{i} = localCap(w, first);
        first = false;
    end
    depth = depth + count(w, '(') - count(w, ')');
    if depth < 0, depth = 0; end
end
out = strjoin(toks, ' ');
if wasString, out = string(out); end
end

% -------------------------------------------------------------------------
function w = localCap(w, isFirst)
MINOR = {'a','an','and','as','at','but','by','for','from','in','into','nor', ...
         'of','on','onto','or','over','per','the','to','up','versus','via', ...
         'vs','with'};
if any(ismember(w, '\{}^_')),            return; end   % LaTeX markup
if ~isempty(regexp(w, '^[a-z]\d', 'once')), return; end % t50, k0
if any(isstrprop(w, 'upper')),           return; end   % ARI, mV, ng/mL
if ~isFirst && ismember(lower(w), MINOR), return; end

parts = strsplit(w, '-');
for j = 1:numel(parts)
    p = parts{j};
    if isempty(p), continue; end
    if any(isstrprop(p, 'upper')),                continue; end
    if ~isempty(regexp(p, '^[a-z]\d', 'once')),   continue; end
    if j > 1 && ismember(lower(p), MINOR),        continue; end
    if isstrprop(p(1), 'lower')
        p(1) = upper(p(1));
        parts{j} = p;
    end
end
w = strjoin(parts, '-');
end
