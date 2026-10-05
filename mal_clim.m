function mal_clim(ax, lims)
%MAL_CLIM  Set colour limits on any MATLAB release (clim R2022a+, else caxis).
try
    clim(ax, lims);
catch
    caxis(ax, lims); %#ok<CAXIS>
end
end
