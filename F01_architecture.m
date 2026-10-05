function F01_architecture()
%F01_ARCHITECTURE  Study design: the end-to-end pipeline of the project.
%   Subfigure: F01a_pipeline_architecture
C = mal_cfg();

fig = mal_figure(34, 22);
ax  = axes(fig); hold(ax,'on'); axis(ax,'off');
xlim(ax,[0 100]); ylim(ax,[0 100]);

pal = mal_palette(11);
boxes = { ...
 %  x    y    w    h   colour        label
  { 3,  78, 26, 15, pal(1,:),  sprintf('DATA RESOURCES\nhost RNA / protein\nparasite genomics\ndual host-parasite RNA')}, ...
  {37,  78, 26, 15, pal(1,:),  sprintf('PAEDIATRIC COHORTS\nMali n = 56 (unlabelled)\nGambia n = 52 (CM / UM)')}, ...
  {71,  78, 26, 15, pal(1,:),  sprintf('PROCESSING\ndual salmon quantification\nhost / parasite splitting')}, ...
  { 3,  56, 26, 15, pal(3,:),  sprintf('INTEGRATION\nBLISS latent integration\nHPAMM-Net multitask')}, ...
  {37,  56, 26, 15, pal(3,:),  sprintf('VALIDATION\nsynthetic -> Mali -> Gambia\ntransportability')}, ...
  {71,  56, 26, 15, pal(2,:),  sprintf('DRUG ARM\nAI pharmacokinetics\npharmaco-omics ADMET')}, ...
  { 3,  34, 26, 15, pal(2,:),  sprintf('FORMULATION\n3^3 full factorial\nBox-Behnken RSM')}, ...
  {37,  34, 26, 15, pal(6,:),  sprintf('NANO-DDS TWIN\nsize / PDI / zeta / EE\nliver targeting')}, ...
  {71,  34, 26, 15, pal(6,:),  sprintf('CHARACTERISATION\n5 release-kinetics models\nDE / MDT / t50 / f2')}, ...
  {37,  11, 26, 15, pal(8,:),  sprintf('PRECISION FORMULATION\npatient omics x formulation\nPROFILE-Net benchmark')} };

for i = 1:numel(boxes)
    b = boxes{i};
    rectangle(ax,'Position',[b{1} b{2} b{3} b{4}],'Curvature',0.12, ...
        'FaceColor',[b{5} 0.18],'EdgeColor',b{5},'LineWidth',2.5);
    text(ax, b{1}+b{3}/2, b{2}+b{4}/2, b{6}, 'HorizontalAlignment','center', ...
        'VerticalAlignment','middle','FontName',C.fontName,'FontSize',18, ...
        'FontWeight','bold','Color',[0.12 0.12 0.12]);
end

arrows = [ 29 85.5 36.4 85.5;   % data -> cohorts
           63 85.5 70.4 85.5;   % cohorts -> processing
           84 78.0 84.0 71.6;   % processing -> down
           29 63.5 36.4 63.5;   % integration -> validation
           63 63.5 70.4 63.5;   % validation -> drug arm
           84 56.0 84.0 49.6;   % drug arm -> down
           29 41.5 36.4 41.5;   % formulation -> nano twin
           63 41.5 70.4 41.5;   % nano twin -> characterisation
           50 34.0 50.0 26.4];  % nano twin -> precision formulation
% horizontal routing rails (drawn as plain lines, no arrow head)
plot(ax,[84 16],[71.0 71.0],'k-','LineWidth',2.0);
plot(ax,[16 16],[71.0 71.6],'k-','LineWidth',2.0);
localArrow(ax,16,71.6,16,71.2);
plot(ax,[84 16],[49.0 49.0],'k-','LineWidth',2.0);
localArrow(ax,16,49.0,16,49.4);
for i = 1:size(arrows,1)
    localArrow(ax, arrows(i,1), arrows(i,2), arrows(i,3), arrows(i,4));
end

title(ax,'Study architecture: from public omics to a personalised nanoparticle formulation', ...
      'FontName',C.fontName,'FontSize',C.fsTitle,'FontWeight','bold');
mal_export(fig,'F01a_pipeline_architecture');
fprintf('F01 done.\n');
end

function localArrow(ax,x1,y1,x2,y2)
plot(ax,[x1 x2],[y1 y2],'k-','LineWidth',2.0);
d = [x2-x1, y2-y1]; L = hypot(d(1),d(2));
if L < 1e-6, return; end
u = d/L; p = [-u(2) u(1)]; s = 1.6;
tip = [x2 y2]; b1 = tip - s*u + 0.5*s*p; b2 = tip - s*u - 0.5*s*p;
patch(ax,'XData',[tip(1) b1(1) b2(1)],'YData',[tip(2) b1(2) b2(2)], ...
      'FaceColor','k','EdgeColor','k');
end
