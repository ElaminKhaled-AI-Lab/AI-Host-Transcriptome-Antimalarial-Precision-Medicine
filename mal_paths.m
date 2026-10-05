function P = mal_paths()
%MAL_PATHS  Locate the project data folders (results/, tables/, reports/).
%   Place these .m files either in the project root (beside results/ and
%   tables/) or in a subfolder of it. Otherwise set ROOT manually below.

ROOT_OVERRIDE = '';   % <-- set an absolute path here if auto-detection fails

here = fileparts(mfilename('fullpath'));
if ~isempty(ROOT_OVERRIDE)
    cand = {ROOT_OVERRIDE};
else
    cand = {here, fullfile(here,'..'), fullfile(here,'..','..'), pwd, fullfile(pwd,'..')};
end

root = '';
for i = 1:numel(cand)
    if isfolder(fullfile(cand{i},'results')) && isfolder(fullfile(cand{i},'tables'))
        root = cand{i}; break
    end
end
if isempty(root)
    error(['mal_paths: could not find a folder containing both results/ and tables/.\n' ...
           'Put these .m files in the project root, or edit ROOT_OVERRIDE in mal_paths.m.']);
end

P.root    = root;
P.results = fullfile(root,'results');
P.tables  = fullfile(root,'tables');
P.reports = fullfile(root,'reports');
P.out     = fullfile(root,'figures_matlab');
if ~isfolder(P.out), mkdir(P.out); end
end
