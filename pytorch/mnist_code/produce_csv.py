import pandas as pd
import os

from experiments import load_metric_history, exists_metric_history
from run_experiments import grid, model_params, train_params, optim_params, data_set

results_folder = "./results/"
experiments = [
    ('vadam_mlp_class', {}, 'Vadam'),
    ('vadamuon_skipfirst_no_rms_mlp_class', {'use_rms': False}, 'VadaMuon')
]



data_set = "mnist"

data = []

exp_id = 0

for i, (hidden_sizes, mc, bs, prec) in enumerate(grid):
    model_params['hidden_sizes'] = hidden_sizes
    model_params['prior_prec'] = prec
    train_params['train_mc_samples'] = mc
    train_params['batch_size'] = bs
    if bs == 1:
        train_params['num_epochs'] = 2
    elif bs == 10:
        train_params['num_epochs'] = 20
    elif bs == 100:
        train_params['num_epochs'] = 200

    base_optim_params = {
        k: v for k, v in optim_params.items() 
        if k in ['learning_rate', 'betas', 'prec_init']
    }
    base_optim_params['prec_init'] = prec

    for exp_name, exp_params, label in experiments:
        exp_optim_params = {**base_optim_params, **exp_params}
        metrics = load_metric_history(
            exp_name, data_set, model_params, train_params, exp_optim_params,
            results_folder, silent_fail=True
        )

        if not metrics:
            print(f"No experiment found for {exp_name}", 
                  f"{hidden_sizes=}, {mc=}, {bs=}, {prec=}, skipping...")
            continue

        

        for metric_name, metric_value in metrics.items():
            for idx, value in enumerate(metric_value):
                num_evals = len(metric_value)
                epoch = (idx + 1) * train_params['num_epochs'] / num_evals
                data_row = {
                    'exp_id': exp_id,
                    'params_id': i,
                    'model': label,
                    'hidden_sizes': str(hidden_sizes),
                    'mc_samples': mc,
                    'batch_size': bs,
                    'prior_prec': prec,
                    'record_idx': idx,
                    'epoch': epoch,
                    'num_evals': num_evals,
                    'metric_name': metric_name,
                    'metric_value': value
                }
                data.append(data_row)
        
        exp_id += 1

df = pd.DataFrame(data)

# Pivot
pivot_df = df.pivot_table(
    index=['exp_id', 'params_id', 'model', 'hidden_sizes', 'mc_samples', 'batch_size', 'prior_prec', 'record_idx', 'epoch', 'num_evals'],
    columns='metric_name',
    values='metric_value'
).reset_index()

# Save to CSV
print("Saving summary CSV to results folder...")
output_csv_path = os.path.join(results_folder, "summary.csv")
pivot_df.to_csv(output_csv_path, index=False)