# Size of figure
figsize = (6, 4)
# figsize = (5, 2.5)

# Font sizes
title_size = 14
legend_size = 12
label_size = 16
tick_size = 16
linewidth = 2

##################
## Load Imports ##
##################

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.animation as anim
plt.ioff()
#plt.ion()

from experiments import load_metric_history, exists_metric_history
from run_experiments import grid, model_params, train_params, optim_params, data_set
        
class AnimatedGif:
    def __init__(self, figsize=(12, 10)):
        plt.rc('text', usetex=True)
        self.fig = plt.figure(figsize=figsize)
        self.images = []
        self.frame_metadata = []  # Store metadata for each frame
 
    def add(self, plot_list, metadata=None, data=None):
        self.images.append(plot_list)
        self.frame_metadata.append(metadata)
        
 
    def save(self, filename, fps=1):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=fps)
    
    def export_frames(self, base_filename, frame_folder='animations/frames'):
        """Export each frame as a separate PNG file."""
        os.makedirs(frame_folder, exist_ok=True)
        
        for i, metadata in enumerate(self.frame_metadata):
            # Construct filename with metadata
            if metadata:
                frame_filename = f"{base_filename}_prec{float(metadata['prec']):06.2f}.png"
            else:
                frame_filename = f"{base_filename}_frame{i:03d}.png"
            
            frame_path = os.path.join(frame_folder, frame_filename)
            
            # Temporarily show only this frame's artists
            for j, img_list in enumerate(self.images):
                for artist in img_list:
                    artist.set_visible(j == i)
            
            self.fig.savefig(frame_path, dpi=150, bbox_inches='tight', pad_inches=0)
            print(f"Saved frame {i+1}/{len(self.images)}: {frame_path}")
        
        # Restore visibility of all artists
        for img_list in self.images:
            for artist in img_list:
                artist.set_visible(True)


def setup_plot_style(num_epochs, ylim, ylabel):
    """Configure plot appearance."""
    plt.xlim([0, num_epochs])
    plt.ylim(ylim)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel("Epoch", fontdict=dict(fontsize=label_size))
    plt.ylabel(ylabel, fontdict=dict(fontsize=label_size))
    plt.grid(True, which="both", color='0.75')
    plt.tight_layout()


def get_metric_data(experiment_name, model_params, train_params, optim_params, 
                    metric_key, plot_every, results_folder):
    """Load and process metric data for plotting."""
    if not exists_metric_history(experiment_name, data_set, model_params, 
                                  train_params, optim_params, results_folder):
        return None, None, None
    metrics = load_metric_history(experiment_name, data_set, model_params, 
                                   train_params, optim_params, results_folder)
    num_evals = len(metrics[metric_key])
    idx = np.arange(start=0, stop=num_evals, step=plot_every)
    epoch = (idx + 1) * train_params['num_epochs'] / num_evals
    met = np.array(metrics[metric_key])
    
    return epoch, met[idx], num_evals


def save_animation(animated_gif, current_params, train_params, ylim, ylabel, 
                   filename_suffix, labels):
    """Save the animated gif with proper formatting."""
    hidden_sizes_, mc_, bs_ = current_params
    print("Creating gif...")
    
    setup_plot_style(train_params['num_epochs'], ylim, ylabel)
    plt.legend(labels, fontsize=legend_size, loc='upper center')
    
    base_filename = (f'layers_{"_".join(map(str, hidden_sizes_))}'
                     f'_batchsize{bs_:03d}{filename_suffix}')
    filename = f'animations/{base_filename}.gif'
    
    # Save the animated gif
    animated_gif.save(filename)
    
    # Export individual frames
    print("Exporting individual frames...")
    animated_gif.export_frames(base_filename)
    
def create_animation(metric_type='test_loss'):
    """Create animation for either test loss or ELBO.
    
    Args:
        metric_type: Either 'test_loss' or 'elbo'
    """
    # Configuration based on metric type
    if metric_type == 'test_loss':
        metric_key = 'test_pred_logloss'
        ylim = [0, 2]
        ylabel = r'Test $\log_{10}$loss'
        filename_suffix = ''
        title_y_offset = 0.02

        experiments = [
            ('vadam_mlp_class', {}, 'r', 'Vadam'),
            ('vadamuon_skipfirst_no_rms_mlp_class', {'use_rms': False}, 'b', 'VadaMuon')
        ]
        transform = lambda x: x / np.log(10)

    elif metric_type == 'test_accuracy':
        metric_key = 'test_pred_accuracy'
        ylim = [0, 1]
        ylabel = r'Test Accuracy'
        filename_suffix = '_accuracy'
        title_y_offset = 0.02
        experiments = [
            ('vadam_mlp_class', {}, 'r', 'Vadam'),
            ('vadamuon_skipfirst_no_rms_mlp_class', {'use_rms': False}, 'b', 'VadaMuon')
        ]
        transform = lambda x: x


    elif metric_type == 'elbo':  
        metric_key = 'elbo_neg_ave'
        ylim = [0, 4]
        ylabel = r'Train Negative Mean ELBO'
        filename_suffix = '_elbo'
        title_y_offset = 0.04
        experiments = [
            ('vadam_mlp_class', {}, 'r', 'Vadam'),
            ('vadamuon_skipfirst_no_rms_mlp_class', {'use_rms': False}, 'b', 'VadaMuon')
        ]
        transform = lambda x: x
    
    else:
        raise ValueError(f"Unknown {metric_type=}")
    
    results_folder = "./results/"
    animated_gif = AnimatedGif(figsize=figsize)
    current_params = None
    plot_every = 10
    
    for i, (hidden_sizes, mc, bs, prec) in enumerate(grid):
        # Check if we need to save the previous animation
        if current_params and current_params != (hidden_sizes, mc, bs):
            if animated_gif.images:
                labels = [exp[3] for exp in experiments]
                save_animation(animated_gif, current_params, train_params, ylim, 
                              ylabel, filename_suffix, labels)
            animated_gif = AnimatedGif(figsize=figsize)
        
        current_params = (hidden_sizes, mc, bs)
        
        # Update parameters
        model_params['hidden_sizes'] = hidden_sizes
        model_params['prior_prec'] = prec
        train_params['train_mc_samples'] = mc
        train_params['batch_size'] = bs
        
        # Set epochs based on batch size
        if bs == 1:
            train_params['num_epochs'] = 2
        elif bs == 10:
            train_params['num_epochs'] = 20
        elif bs == 100:
            train_params['num_epochs'] = 200
        
        # Base optim params (without experiment-specific params)
        base_optim_params = {
            k: v for k, v in optim_params.items() 
            if k in ['learning_rate', 'betas', 'prec_init']
        }
        base_optim_params['prec_init'] = prec
        
        # Try to plot all experiments
        plots = []
        all_found = True
        
        for exp_name, exp_params, color, label in experiments:
            current_optim_params = {**base_optim_params, **exp_params}
            
            epoch, met, _ = get_metric_data(
                exp_name, model_params, train_params, current_optim_params,
                metric_key, plot_every, results_folder
            )
            
            if epoch is None:
                print(f"No experiment found for {exp_name}", 
                      f"{hidden_sizes=}, {mc=}, {bs=}, {prec=}, skipping...")
                all_found = False
                break
            
            
            plot, = plt.plot(epoch, transform(met), color=color, 
                           linestyle='-', linewidth=linewidth, label=label)
            plots.append(plot)
        
        if not all_found:
            # Clear any plots that were created
            for plot in plots:
                plot.remove()
            continue
        
        # Add title
        plot_titles = [
            plt.text(
                train_params['num_epochs'], 
                ylim[1] + title_y_offset, 
                f"Precision: {prec}", 
                horizontalalignment='right', 
                verticalalignment='bottom', 
                fontdict=dict(fontsize=title_size)
            ),
            plt.text(
                0, 
                ylim[1] + title_y_offset, 
                f"Batch Size: {bs}", 
                horizontalalignment='left', 
                verticalalignment='bottom', 
                fontdict=dict(fontsize=title_size)
            )
        ]
        
        print(f"All experiments found for {hidden_sizes=}, {mc=}, {bs=}, {prec=}")
        
        # Add frame with metadata and data
        frame_metadata = {
            'prec': prec,
            'hidden_sizes': hidden_sizes,
            'mc': mc,
            'bs': bs
        }
        
        frame_data = {
            'metric_type': metric_type,
            'metric_key': metric_key
        }
        
        animated_gif.add(plot_titles + plots, metadata=frame_metadata, data=frame_data)
    
    # Save final animation
    if animated_gif.images and current_params:
        labels = [exp[3] for exp in experiments]
        print(f"{current_params=}")
        save_animation(animated_gif, current_params, train_params, ylim, 
                      ylabel, filename_suffix, labels)


if __name__ == '__main__':
    # Create both animations
    print("Creating test loss animations...")
    create_animation(metric_type='test_loss')
    
    print("\nCreating ELBO animations...")
    create_animation(metric_type='elbo')

    print("\nCreating test accuracy animations...")
    create_animation(metric_type='test_accuracy')