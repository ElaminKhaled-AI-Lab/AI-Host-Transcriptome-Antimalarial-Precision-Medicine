function check_figure_inputs()
%CHECK_FIGURE_INPUTS  Report which figures can be drawn, and how to fix the rest.
%
%   Every figure reads specific files from results/ and tables/. This checks all
%   of them and, for anything missing, prints the EXACT command that produces it.
%   Run it before run_all_figures to avoid discovering gaps halfway through.
%
%   Usage:  check_figure_inputs

P = mal_paths();
fprintf('\nChecking figure inputs under %s\n\n', P.root);

% {figure, description, {file, generating command; ...}}
S = {
 'F01', 'study architecture', {};
 'F02', 'data resources and cohort', {
    'reports/data_completeness.tsv',  '(from the earlier data-audit step; F02a is skipped without it, F02b-d still draw)'
    'tables/paediatric_cohort_used.csv', 'python scripts/14_omics_formulation/build_patient_formulation_dataset.py --drug lumefantrine'};
 'F03', 'dual-RNA quality control', {
    'results/dualrna_qc_real_mali.json',       'python scripts/11_validation/dualrna_qc_robustness.py --cohort real_mali'
    'results/dualrna_qc_external_gambia.json', 'python scripts/11_validation/dualrna_qc_robustness.py --cohort external_gambia'};
 'F04', 'integration on synthetic data', {
    'results/integration_benchmark_seed11.json', 'python scripts/07_integration_benchmarks/run_synthetic_benchmark.py'};
 'F05', 'HPAMM-Net multitask and ablations', {
    'results/hpamm_ablations.json',        'python scripts/08_hpamm_net/run_ablations_multitask.py'
    'results/hpamm_multitask_seed11.json', 'python scripts/08_hpamm_net/train_hpamm_multitask.py'};
 'F06', 'real validation against controls', {
    'results/dualrna_qc_real_mali.json',            'python scripts/11_validation/dualrna_qc_robustness.py --cohort real_mali'
    'results/dualrna_qc_external_gambia.json',      'python scripts/11_validation/dualrna_qc_robustness.py --cohort external_gambia'
    'results/final_clustering_real_mali.json',      'python scripts/11_validation/finalise_clustering.py --cohort real_mali --n-perm 1000'
    'results/final_clustering_external_gambia.json','python scripts/11_validation/finalise_clustering.py --cohort external_gambia --n-perm 1000'};
 'F07', 'AI pharmacokinetics', {
    'results/pk_ai_results.json', 'python scripts/09_treatment_models/simulate_pk_cohort.py  THEN  python scripts/09_treatment_models/pk_ai_pipeline.py'};
 'F08', 'formulation design of experiments', {
    'tables/nano_dds_dissolution_parameters.csv', 'python scripts/13_nano_dds/analyze_dissolution.py --replicates 3 --noise-rsd 0.05'
    'tables/rsm_anova_size_nm.csv',               'Rscript scripts/13_nano_dds/rsm_dissolution_analysis.R PLGA_50_50 lumefantrine'};
 'F09', 'nanoparticle characterisation', {
    'tables/nano_dds_dissolution_parameters.csv', 'python scripts/13_nano_dds/analyze_dissolution.py --replicates 3 --noise-rsd 0.05'};
 'F10', 'dissolution and release kinetics', {
    'tables/nano_dds_dissolution_parameters.csv', 'python scripts/13_nano_dds/analyze_dissolution.py --replicates 3 --noise-rsd 0.05'
    'tables/nano_dds_release_profiles.csv',       'python scripts/13_nano_dds/export_release_profiles.py'};
 'F11', 'optimisation, stability, validation', {
    'results/nano_dds_optimization.json', 'python scripts/13_nano_dds/optimize_nano_dds.py --drug lumefantrine'
    'results/nano_dds_stability.json',    'python scripts/13_nano_dds/validate_and_report.py'
    'tables/nano_dds_validation.csv',     'python scripts/13_nano_dds/validate_and_report.py'
    'tables/nano_dds_dissolution_parameters.csv', 'python scripts/13_nano_dds/analyze_dissolution.py --replicates 3 --noise-rsd 0.05'};
 'F12', 'precision formulation', {
    'tables/personalized_formulation_per_patient.csv', 'python scripts/14_omics_formulation/personalize_all_patients.py'
    'results/integration_benchmark_profile_net.json',  'python scripts/14_omics_formulation/benchmark_integration.py --device cpu'};
};

fixes = {};
nOK = 0;
for i = 1:size(S,1)
    fid = S{i,1}; desc = S{i,2}; req = S{i,3};
    if isempty(req)
        fprintf('  [ OK ] %s  %s  (no data needed)\n', fid, desc);
        nOK = nOK + 1; continue
    end
    missing = {};
    for r = 1:size(req,1)
        f = fullfile(P.root, req{r,1});
        if ~isfile(f)
            missing{end+1} = req(r,:); %#ok<AGROW>
        end
    end
    if isempty(missing)
        fprintf('  [ OK ] %s  %s\n', fid, desc);
        nOK = nOK + 1;
    else
        fprintf('  [MISS] %s  %s\n', fid, desc);
        for m = 1:numel(missing)
            fprintf('           missing: %s\n', missing{m}{1});
            fprintf('           fix    : %s\n', missing{m}{2});
            fixes{end+1} = missing{m}{2}; %#ok<AGROW>
        end
    end
end

fprintf('\n  %d of %d figures have everything they need.\n', nOK, size(S,1));
if ~isempty(fixes)
    u = unique(fixes, 'stable');
    fprintf('\n  Run these (from the project root, with the malaria environment active),\n');
    fprintf('  then run_all_figures:\n\n');
    for i = 1:numel(u)
        fprintf('    %s\n', u{i});
    end
end
fprintf('\n');
end
