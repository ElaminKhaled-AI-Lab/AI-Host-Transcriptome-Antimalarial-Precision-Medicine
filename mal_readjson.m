function s = mal_readjson(p)
%MAL_READJSON  jsondecode that survives bare NaN / Infinity literals.
%   Python writes NaN unquoted, which jsondecode rejects; convert to null.
if ~isfile(p)
    error('mal_readjson: file not found: %s', p);
end
txt = fileread(p);
txt = regexprep(txt, '(?<=[:\[,\s])NaN(?=\s*[,\}\]])',        'null');
txt = regexprep(txt, '(?<=[:\[,\s])-?Infinity(?=\s*[,\}\]])', 'null');
s = jsondecode(txt);
end
