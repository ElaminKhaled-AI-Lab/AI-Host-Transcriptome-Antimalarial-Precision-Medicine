function v = mal_getcol(T, patterns, fallback)
%MAL_GETCOL  Fetch a table column by name pattern, tolerant of schema changes.
%   Tries each pattern as a case-insensitive substring of the variable names and
%   returns the first match, so a CSV written by a slightly different version of
%   the generating script still works. Returns FALLBACK (default []) if none match.
if nargin < 3, fallback = []; end
v = fallback;
vn = T.Properties.VariableNames;
if ischar(patterns) || isstring(patterns), patterns = cellstr(patterns); end
for p = 1:numel(patterns)
    hit = find(contains(lower(vn), lower(patterns{p})), 1);
    if ~isempty(hit)
        v = T.(vn{hit});
        return
    end
end
end
