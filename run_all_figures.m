function run_all_figures()
%RUN_ALL_FIGURES  Generate every subfigure of the Malaria_Project manuscript.
%   Each subfigure is written as .png (1200 dpi), .pdf (vector) and .fig into
%   <project root>/figures_matlab/.
%
%   Run the individual F##_*.m scripts instead if you only need one figure.

P = mal_paths();
fprintf('Malaria_Project figure suite\n');
fprintf('  data root : %s\n', P.root);
fprintf('  output    : %s\n\n', P.out);

steps = {@F01_architecture,           'F01 study architecture'; ...
         @F02_data_and_cohort,        'F02 data resources and cohort'; ...
         @F03_dualrna_qc,             'F03 dual-RNA quality control'; ...
         @F04_integration_synthetic,  'F04 integration on synthetic data'; ...
         @F05_hpamm_net,              'F05 HPAMM-Net multitask and ablations'; ...
         @F06_real_validation,        'F06 real validation and transportability'; ...
         @F07_pk_ai,                  'F07 AI pharmacokinetics'; ...
         @F08_formulation_doe,        'F08 formulation design of experiments'; ...
         @F09_nano_characterisation,  'F09 nanoparticle characterisation'; ...
         @F10_dissolution,            'F10 dissolution and release kinetics'; ...
         @F11_optimisation_validation,'F11 optimisation, stability, validation'; ...
         @F12_precision_formulation,  'F12 precision formulation'};

failed = {};
for i = 1:size(steps,1)
    fprintf('[%2d/%2d] %s\n', i, size(steps,1), steps{i,2});
    try
        steps{i,1}();
    catch ME
        failed{end+1} = steps{i,2}; %#ok<AGROW>
        fprintf('   !! FAILED: %s\n', ME.message);
    end
end

fprintf('\n');
if isempty(failed)
    fprintf('All figures written to %s\n', P.out);
else
    fprintf('Completed with failures in:\n');
    for i = 1:numel(failed), fprintf('   - %s\n', failed{i}); end
end
end
