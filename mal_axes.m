function mal_axes(ax, ttl, xl, yl)
%MAL_AXES  Apply the house style (Arial, 25 pt inside, 30 pt heading) to an axes.
C = mal_cfg();
set(ax,'FontName',C.fontName,'FontSize',C.fsAxis,'LineWidth',C.axisWidth,'Box','off', ...
       'TickDir','out','Layer','top');
if nargin>1 && ~isempty(ttl)
    title(ax, ttl, 'FontName',C.fontName,'FontSize',C.fsTitle,'FontWeight','bold');
end
if nargin>2 && ~isempty(xl)
    xlabel(ax, mal_titlecase(xl), 'FontName',C.fontName,'FontSize',C.fsAxis);
end
if nargin>3 && ~isempty(yl)
    ylabel(ax, mal_titlecase(yl), 'FontName',C.fontName,'FontSize',C.fsAxis);
end
if C.showGrid
    grid(ax,'on'); ax.GridAlpha = C.gridAlpha;
else
    grid(ax,'off');
end
set(ax,'Color','none');          % axes interior clear; the figure sets the colour
lg = findobj(ancestor(ax,'figure'),'Type','Legend');
for i = 1:numel(lg)
    set(lg(i),'FontName',C.fontName,'FontSize',C.fsLegend,'Box','off');
end
cb = findobj(ancestor(ax,'figure'),'Type','ColorBar');
for i = 1:numel(cb)
    set(cb(i),'FontName',C.fontName,'FontSize',C.fsAxis);
end
end
