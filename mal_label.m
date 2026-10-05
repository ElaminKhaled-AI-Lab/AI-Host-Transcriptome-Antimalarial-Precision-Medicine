function mal_label(ax, str, pos)
%MAL_LABEL  Small in-panel annotation (n = ..., thresholds).
C = mal_cfg();
if nargin<3 || isempty(pos), pos = 'northwest'; end
switch lower(pos)
    case 'northwest', x=0.03; y=0.96; ha='left';  va='top';
    case 'northeast', x=0.97; y=0.96; ha='right'; va='top';
    case 'southwest', x=0.03; y=0.04; ha='left';  va='bottom';
    otherwise,        x=0.97; y=0.04; ha='right'; va='bottom';
end
text(ax, x, y, str, 'Units','normalized', 'FontName',C.fontName, ...
     'FontSize',C.fsAnnot, 'HorizontalAlignment',ha, 'VerticalAlignment',va, ...
     'Margin',3);
end
