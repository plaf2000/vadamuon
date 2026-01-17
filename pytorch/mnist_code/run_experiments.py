from experiments import ExperimentVadaMuonMLPClass, ExperimentVadamMLPClass, ExperimentBBBMLPClass, exists_metric_history
from multiprocessing import Pool

####################
## Set parameters ##
####################

# Folder for storing results
results_folder = "./results/"

# Folder containing data
data_folder = "./../vadam/data"

# Data set
data_set = "mnist"

# Model parameters
model_params = {'hidden_sizes': None,
                'act_func': "relu",
                'prior_prec': None}

# Training parameters
train_params = {'num_epochs': None,
                'batch_size': 100,
                'train_mc_samples': None,
                'eval_mc_samples': 10,
                'seed': 123}

# Optimizer parameters
optim_params = {'learning_rate': 0.001,
                'betas': (0.9,0.999),
                'prec_init': None}
optim_params_vogn = {'learning_rate': 0.001,
                     'beta': 0.999,
                     'prec_init': None}

# Evaluations per epoch
evals_per_epoch = None


#################
## Define grid ##
#################

grid = [(hidden_sizes, mc, bs, prec) 
for hidden_sizes in ([32, 16], [400,400], [64, 32, 16])
for mc in (1, 10)
for bs in (100, 10, 1)
for prec in (1e-2, 1e-1, 1e0, 1e1, 1e2)][::-1]

##################################
## Run experiments sequentially ##
##################################

N_PROCESSES = 8

def train_loop(i, hidden_sizes, mc, bs, prec, grid_size = len(grid)):
    global model_params, train_params, optim_params, optim_params_vogn, evals_per_epoch, data_set, data_folder, results_folder
    model_params['hidden_sizes'] = hidden_sizes
    model_params['prior_prec'] = prec
    optim_params['prec_init'] = prec
    optim_params_vogn['prec_init'] = prec
    train_params['train_mc_samples'] = mc
    train_params['batch_size'] = bs
    if bs==1:
        train_params['num_epochs'] = 2
        evals_per_epoch = 600
    elif bs==10:
        train_params['num_epochs'] = 20
        evals_per_epoch = 60
    elif bs==100:
        train_params['num_epochs'] = 200
        evals_per_epoch = 6
    
    # Run VadaMuon

    # With RMS scaling

    for j, use_rms in enumerate([True, False]):
        # for skip_first in [True, False]:
            optim_params['use_rms'] = use_rms
            # optim_params['skip_first_layer'] = skip_first
            experiment = ExperimentVadaMuonMLPClass(results_folder = results_folder, 
                                                data_folder = data_folder,
                                                data_set = data_set, 
                                                model_params = model_params, 
                                                train_params = train_params, 
                                                optim_params = optim_params,
                                                evals_per_epoch = evals_per_epoch,
                                                normalize_x = False)
            
            if not exists_metric_history(experiment.experiment_name, model_params, train_params, optim_params, results_folder, data_set):
                print(f"Running experiment {i*3+j+1}/{len(grid)*3}: {experiment.experiment_name=}, {hidden_sizes=}, {mc=}, {bs=}, {prec=}, {use_rms=}")
                experiment.run(log_metric_history = True)
            
                experiment.save(save_final_metric = True,
                            save_metric_history = True,
                            save_objective_history = False,
                            save_model = False,
                            save_optimizer = False)
    
    ###################################################################################
    # Since we already have the results for Vadam and BBVI (./results/results.zip),   #
    # skip these two experiments.                                                     #
    ###################################################################################
    
    # Run Vadam

    # Only consider relevant parameters
    optim_params = {
            k: v for k, v in optim_params.items() if k in ['learning_rate', 'betas', 'prec_init']
    }

    experiment = ExperimentVadamMLPClass(results_folder = results_folder, 
                                        data_folder = data_folder,
                                        data_set = data_set, 
                                        model_params = model_params, 
                                        train_params = train_params, 
                                        optim_params = optim_params,
                                        evals_per_epoch = evals_per_epoch,
                                        normalize_x = False)
    if not exists_metric_history(experiment.experiment_name, model_params, train_params, optim_params, results_folder, data_set):
        print(f"Running experiment {i*3+3}/{len(grid)*3}: {experiment.experiment_name=}, {hidden_sizes=}, {mc=}, {bs=}, {prec=}")
        experiment.run(log_metric_history = True)
    
        experiment.save(save_final_metric = True,
                    save_metric_history = True,
                    save_objective_history = False,
                    save_model = False,
                    save_optimizer = False)
    
    # Run BBVI
    # experiment = ExperimentBBBMLPClass(results_folder = results_folder, 
    #                                    data_folder = data_folder,
    #                                    data_set = data_set, 
    #                                    model_params = model_params, 
    #                                    train_params = train_params, 
    #                                    optim_params = optim_params,
    #                                    evals_per_epoch = evals_per_epoch,
    #                                    normalize_x = False)
    
    # experiment.run(log_metric_history = True)
    
    # experiment.save(save_final_metric = True,
    #                 save_metric_history = True,
    #                 save_objective_history = False,
    #                 save_model = False,
    #                 save_optimizer = False)

def process_job(process_grid):
    for i, (hidden_sizes, mc, bs, prec) in enumerate(process_grid):
        train_loop(i, hidden_sizes, mc, bs, prec, grid_size = len(process_grid))
     
if __name__ == "__main__":
    if N_PROCESSES > 1:

        with Pool(processes=N_PROCESSES) as pool:
            pool.starmap(process_job, [[grid[i::N_PROCESSES]] for i in range(N_PROCESSES)])


    else:
        process_job(grid)

        
