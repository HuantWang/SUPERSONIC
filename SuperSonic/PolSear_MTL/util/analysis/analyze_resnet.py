import os
import re
import argparse
import logging
import pickle
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from matplotlib import cm
from tqdm import tqdm

import torch
from torch.optim import Adam

from PIL import Image

# from misc_functions import preprocess_image, recreate_image, save_image
# from torchbeast.polybeast import Net as ResNetPoly
from torchbeast.models.resnet_monobeast import ResNet as ResNetMono

logging.getLogger("matplotlib.font_manager").disabled = True

########################################################################################################################
# From https://github.com/utkuozbulak/pytorch-cnn-visualizations                                                       #
########################################################################################################################


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        # im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


class CNNLayerVisualization:
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = torch.zeros([1])
        self.created_image = None
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # go through the CNN layers and count up to the selected_layer CONV layer, then register the hook for that layer
        # would be nice to only pass stuff through the network up to that point but seems difficult to do
        all_convs = []
        for i in range(0, len(self.model.feat_convs)):
            # the order of these needs to match that in the forward() function of the ResNet
            all_convs.append(self.model.feat_convs[i][0])
            all_convs.append(self.model.resnet1[i][1])
            all_convs.append(self.model.resnet1[i][3])
            all_convs.append(self.model.resnet2[i][1])
            all_convs.append(self.model.resnet2[i][3])

        # Hook the selected layer
        # self.model[self.selected_layer].register_forward_hook(hook_function)
        all_convs[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        # random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # random_image = np.uint8(np.random.uniform(0, 255, (1, 84, 84)))
        random_image = np.random.uniform(0, 255, (4, 84, 84))
        # Process image and return variable
        # processed_image = preprocess_image(random_image, False)
        processed_image = torch.tensor(random_image, requires_grad=True)
        processed_image = processed_image.unsqueeze(0).unsqueeze(0).detach().requires_grad_(True)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 101):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            """
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            """
            self.model(processed_image, run_to_conv=self.selected_layer)
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            # self.created_image = recreate_image(processed_image)
            self.created_image = processed_image
            # Save image
            if i % 5 == 0:
                im_path = os.path.expandvars(os.path.expanduser("~/logs/optim_test"))
                im_path = im_path + '/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                im = np.squeeze(self.created_image.detach().numpy(), (0, 1))
                # im[im < 0] = 0
                # im[im > 1] = 1
                im = im[0]
                im -= im.min()
                im = np.divide(im, im.max())
                im = np.round(im * 255)
                im = np.uint8(im)
                save_image(im, im_path)

########################################################################################################################
#                                                                                                                      #
########################################################################################################################


single_task_names = ["Carnival", "AirRaid", "DemonAttack", "NameThisGame", "Pong", "SpaceInvaders"]
multi_task_name = "MultiTask"
multi_task_popart_name = "MultiTaskPopart"
comparison_choices = ["single_multi", "single_multipop", "multi_multipop", "single_single", "time_steps"]


def filter_vis(flags):
    paths = flags.load_path.split(",")
    if len(paths) > 1:
        logging.warning("More than one model specified for filter visualisation. "
                        "Only the first model will be visualised.")
        paths = paths[:1]
    model = load_models(paths)

    # Fully connected layer is not needed
    layer_vis = CNNLayerVisualization(model, flags.layer_index, flags.filter_index)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()


def single_filter_comp(model_a, model_b, compute_optimal=True):
    models = [model_a, model_b]
    filter_list = [[] for _ in models]
    for m_idx, m in enumerate(models):
        for i in range(0, len(m.feat_convs)):
            filter_list[m_idx].append(m.feat_convs[i][0].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet1[i][1].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet1[i][3].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet2[i][1].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet2[i][3].weight.detach().numpy())
    filter_list_a = filter_list[0]
    filter_list_b = filter_list[1]

    distance_data = []
    default_dist_data = []
    optimal_dist_data = []
    for f_idx, f in enumerate(tqdm(filter_list_b)):
        # reshape filters
        all_filters = f.shape[0] * f.shape[1]
        filter_size = f.shape[2] * f.shape[3]
        original = np.reshape(filter_list_a[f_idx], (all_filters, filter_size))
        comparison = np.reshape(f, (all_filters, filter_size))

        # compute distances
        distances = cdist(original, comparison, metric="sqeuclidean")
        default_dist_sum = np.diagonal(distances).sum()
        default_dist_mean = np.diagonal(distances).mean()

        # compute optimal assignment
        if compute_optimal:
            row_idx, col_idx = linear_sum_assignment(distances)
            optimal_dist_sum = distances[row_idx, col_idx].sum()
            optimal_dist_mean = distances[row_idx, col_idx].mean()
        else:
            optimal_dist_sum = 0
            optimal_dist_mean = 0

        distance_data.append(distances)
        default_dist_data.append((f.shape, default_dist_sum, default_dist_mean))
        optimal_dist_data.append((f.shape, optimal_dist_sum, optimal_dist_mean))

    return distance_data, default_dist_data, optimal_dist_data


def parallel_filter_calc(combined_input):
    models, model_name, comparison = combined_input
    if type(comparison) is str:
        return single_filter_comp(models[comparison], models[model_name], not flags.comp_no_optimal)
    return single_filter_comp(comparison[model_name], models[model_name], not flags.comp_no_optimal)


def filter_comp(flags):
    model_names = single_task_names + [multi_task_name] + [multi_task_popart_name]

    # 1. get the model base directory
    # 2. get the paths for all subdirectories (only those matching a list of "model names")
    # 3. get lists of all model names: either x models every y checkpoints for all models
    #    or the same but only for the first 50 models (and the last)
    # 4. load models for every time step and do the filter comparison between
    #    a) all single-task and the vanilla multi-task model
    #    b) all single-task and the popart multi-task model
    #    c) the vanilla and popart multi-task models
    # 5. write values to separate CSV files and store them as well
    # 6. display stuff

    base_path = flags.load_path.split(",")
    if len(base_path) > 1:
        logging.warning("More than one path specified for filter progress visualization. "
                        "In this mode only the base directory is required. Using only the first path.")
    base_path = base_path[0]

    # get base directories for all intermediate models
    intermediate_paths = []
    for p in os.listdir(base_path):
        if p in model_names:
            intermediate_paths.append((p, os.path.join(base_path, p, "intermediate")))

    # get all model checkpoints that should be loaded
    logging.info("Determining checkpoints to load.")
    selected_model_paths = []
    for model_name, path in intermediate_paths:
        checkpoints = []
        for checkpoint in os.listdir(path):
            if not checkpoint.endswith(".tar"):
                continue
            checkpoint_n = int(re.search(r'\d+', checkpoint).group())
            checkpoints.append((checkpoint_n, checkpoint))
        checkpoints.sort()

        if flags.match_num_models and "MultiTask" in model_name:
            index = list(np.round(np.linspace(0, 50 - 1, flags.comp_num_models)).astype(int))
            index.append(len(checkpoints) - 1)
        else:
            index = np.round(np.linspace(0, len(checkpoints) - 1, flags.comp_num_models)).astype(int)
        selected_models = [os.path.join(base_path, model_name, "intermediate", checkpoints[i][1]) for i in index]
        selected_model_paths.append((model_name, selected_models))

    # prepare data structures
    single_multi_data = {
        n: {
            s: {
                t: [] for t in ["sum", "mean"]
            } for s in ["default", "optimal"]
        } for n in single_task_names
    }
    single_single_data = {
        n: {
            s: {
                t: [] for t in ["sum", "mean"]
            } for s in ["default", "optimal"]
        } for n in single_task_names if n != flags.comp_single_single_model
    }
    single_multipop_data = {
        n: {
            s: {
                t: [] for t in ["sum", "mean"]
            } for s in ["default", "optimal"]
        } for n in single_task_names
    }
    multi_multipop_data = {
        s: {
            t: [] for t in ["sum", "mean"]
        } for s in ["default", "optimal"]
    }
    time_steps_data = {
        n: {
            s: {
                t: [] for t in ["sum", "mean"]
            } for s in ["default", "optimal"]
        } for n in model_names
    }
    all_data = {
        "single_multi": single_multi_data,
        "single_single": single_single_data,
        "single_multipop": single_multipop_data,
        "multi_multipop": multi_multipop_data,
        "time_steps": time_steps_data
    }

    # go through each time step
    logging.info("Starting the computation.")
    time_steps_last_models = None
    for t in range(flags.comp_num_models + (1 if flags.match_num_models else 0)):
        # load checkpoints for each model
        logging.info("Loading checkpoints for all models ({}/{}).".format(t + 1, flags.comp_num_models))
        models = {}
        for model_name, model_paths in selected_model_paths:
            models[model_name] = load_models(
                [model_paths[t - (1 if t == flags.comp_num_models and "multi" not in model_name.lower() else 0)]])

        # compare single and vanilla multi-task models
        if "single_multi" in flags.comp_between:
            logging.info("Comparing single-task and vanilla multi-task models ({}/{})."
                         .format(t + 1, flags.comp_num_models))
            with mp.Pool(min(len(single_task_names), os.cpu_count())) as pool:
                full_data = pool.map(parallel_filter_calc, [(models, mn, multi_task_name) for mn in single_task_names])
            for model_name, data in zip(single_task_names, full_data):
                dist, dd_data, od_data = data
                single_multi_data[model_name]["default"]["sum"].append(np.stack([d[1] for d in dd_data]))
                single_multi_data[model_name]["default"]["mean"].append(np.stack([d[2] for d in dd_data]))
                single_multi_data[model_name]["optimal"]["sum"].append(np.stack([d[1] for d in od_data]))
                single_multi_data[model_name]["optimal"]["mean"].append(np.stack([d[2] for d in od_data]))

        # compare one single-task model with all the others
        if "single_single" in flags.comp_between:
            logging.info("Comparing {} single-task with all other single-task models ({}/{})."
                         .format(flags.comp_single_single_model, t + 1, flags.comp_num_models))
            with mp.Pool(min(len(single_task_names) - 1, os.cpu_count())) as pool:
                full_data = pool.map(parallel_filter_calc, [(models, mn, flags.comp_single_single_model)
                                                            for mn in single_task_names
                                                            if mn != flags.comp_single_single_model])
            for model_name, data in zip([n for n in single_task_names if n != flags.comp_single_single_model],
                                        full_data):
                dist, dd_data, od_data = data
                single_single_data[model_name]["default"]["sum"].append(np.stack([d[1] for d in dd_data]))
                single_single_data[model_name]["default"]["mean"].append(np.stack([d[2] for d in dd_data]))
                single_single_data[model_name]["optimal"]["sum"].append(np.stack([d[1] for d in od_data]))
                single_single_data[model_name]["optimal"]["mean"].append(np.stack([d[2] for d in od_data]))

        # compare single and multi-task PopArt models
        if "single_multipop" in flags.comp_between:
            logging.info("Comparing single-task and multi-task PopArt models ({}/{})."
                         .format(t + 1, flags.comp_num_models))
            with mp.Pool(min(len(single_task_names), os.cpu_count())) as pool:
                full_data = pool.map(parallel_filter_calc, [(models, mn, multi_task_popart_name)
                                                            for mn in single_task_names])
            for model_name, data in zip(single_task_names, full_data):
                dist, dd_data, od_data = data
                single_multipop_data[model_name]["default"]["sum"].append(np.stack([d[1] for d in dd_data]))
                single_multipop_data[model_name]["default"]["mean"].append(np.stack([d[2] for d in dd_data]))
                single_multipop_data[model_name]["optimal"]["sum"].append(np.stack([d[1] for d in od_data]))
                single_multipop_data[model_name]["optimal"]["mean"].append(np.stack([d[2] for d in od_data]))

        # compare multi-task and multi-task PopArt models
        if "multi_multipop" in flags.comp_between:
            logging.info("Comparing vanilla multi-task and multi-task PopArt models ({}/{})."
                         .format(t + 1, flags.comp_num_models))
            dist, dd_data, od_data = single_filter_comp(models[multi_task_popart_name], models[multi_task_name],
                                                        not flags.comp_no_optimal)
            multi_multipop_data["default"]["sum"].append(np.stack([d[1] for d in dd_data]))
            multi_multipop_data["default"]["mean"].append(np.stack([d[2] for d in dd_data]))
            multi_multipop_data["optimal"]["sum"].append(np.stack([d[1] for d in od_data]))
            multi_multipop_data["optimal"]["mean"].append(np.stack([d[2] for d in od_data]))

        if "time_steps" in flags.comp_between:
            logging.info("Comparing between time steps ({}/{}).".format(t + 1, flags.comp_num_models))
            if time_steps_last_models:
                with mp.Pool(min(len(model_names), os.cpu_count())) as pool:
                    full_data = pool.map(parallel_filter_calc, [(models, mn, time_steps_last_models)
                                                                for mn in model_names])
                for model_name, data in zip(model_names, full_data):
                    dist, dd_data, od_data = data
                    time_steps_data[model_name]["default"]["sum"].append(np.stack([d[1] for d in dd_data]))
                    time_steps_data[model_name]["default"]["mean"].append(np.stack([d[2] for d in dd_data]))
                    time_steps_data[model_name]["optimal"]["sum"].append(np.stack([d[1] for d in od_data]))
                    time_steps_data[model_name]["optimal"]["mean"].append(np.stack([d[2] for d in od_data]))
            time_steps_last_models = {mn: models[mn] for mn in models}

    def update_dict(d):
        if type(d) == list:
            if type(d[0]) == list:
                result = [np.stack([d[time][filter] for time in range(len(d))]) for filter in range(len(d[0]))]
            else:
                result = np.stack(d)
            return result
        else:
            for k in d:
                d[k] = update_dict(d[k])
            return d

    # write data to files
    for file_name, dictionary in all_data.items():
        if file_name not in flags.comp_between:
            continue

        dictionary = update_dict(dictionary)
        full_path = os.path.join(
            os.path.expanduser(flags.save_dir),
            "filter_comp", "{}_{}".format(flags.comp_num_models, "match" if flags.match_num_models else "no_match"),
            "{}{}.pkl".format(file_name, "" if file_name != "single_single" else "_{}"
                              .format(flags.comp_single_single_model))
        )
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        with open(full_path, "wb") as f:
            pickle.dump(dictionary, f)
        logging.info("Wrote data to file '{}'.".format(full_path))


def filter_comp_plot(flags):
    plot_titles = {
        "single_multi": "Single-task and vanilla multi-task",
        "single_multipop": "Single-task and multi-task with PopArt",
        "multi_multipop": "Vanilla multi-task and multi-task with PopArt",
        "single_single": "{} with the other single-task models",
        "time_steps": "Difference between models over training iterations"
    }

    all_data = {}
    for f in os.listdir(flags.load_path):
        if f.endswith(".pkl") and (f[:-4] in comparison_choices or "single_single" in f):
            with open(os.path.join(flags.load_path, f), "rb") as p:
                all_data[f[:-4]] = pickle.load(p)

    # some useful stuff
    num_layers = 15
    labels = ["layer_{:02d}".format(l_idx) for l_idx in range(num_layers)]
    color_idx = np.linspace(0.2, 1.2, num_layers)
    color_map = cm.get_cmap("Blues")

    for comp_choice, current_data in all_data.items():
        if comp_choice == "multi_multipop":
            plt.figure()
            ax = plt.gca()
            data = current_data[flags.plot_match_type][flags.plot_metric_type]
            if flags.plot_heatmaps:
                plt.imshow(data.T)
                plt.xticks(np.arange(data.shape[0]), np.arange(data.shape[0]))
                plt.yticks(np.arange(num_layers), np.arange(num_layers))
            else:
                for color, d, label in zip(color_idx, data.T, labels):
                    plt.plot(d, color=color_map(color), label=label)
                ax.set_xticks(np.arange(data.shape[0]))
                ax.grid(linestyle="dashed", linewidth="0.5", color="gray")
                plt.ylim(bottom=0)
            plt.xlabel("Model checkpoints throughout training")
            plt.ylabel("SSD ({}) between corresponding filters ({} match)"
                       .format(flags.plot_metric_type, flags.plot_match_type))
            plt.title(plot_titles[comp_choice])
            if flags.save_figures:
                plt.savefig(os.path.join(flags.load_path, "multi_multipop_{}_{}.png"
                                         .format(flags.plot_match_type, flags.plot_metric_type)))

            data_default = current_data["default"]["mean"]
            data_optimal = current_data["optimal"]["mean"]
            ratio = np.divide(data_default, data_optimal)
            """"
            print("{}: mean = {}, std = {} ({})".format(plot_titles[comp_choice],
                                                        ratio.mean(), ratio.std(), list(ratio.mean(axis=0))))
            """
        else:
            if comp_choice == "time_steps":
                single_task_cols = 3
            else:
                single_task_cols = 2
                if "single_single" in comp_choice:
                    comp_choice = "single_single"
            single_task_rows = int(np.ceil((len(current_data) / single_task_cols)))
            fig, ax = plt.subplots(nrows=single_task_rows, ncols=single_task_cols,
                                   figsize=(single_task_cols * 3, single_task_rows * 2.5,),
                                   sharex=True, sharey=True)
            max_value = 0
            for i in range(len(ax)):
                for j in range(len(ax[i])):
                    if i * single_task_cols + j == len(current_data):
                        ax[i][j].axis("off")
                        continue

                    data = current_data[list(current_data.keys())[i * single_task_cols + j]][
                        flags.plot_match_type][flags.plot_metric_type]
                    if flags.plot_heatmaps:
                        ax[i][j].imshow(data.T)
                        ax[i][j].set_xticks(np.arange(data.shape[0]))
                        ax[i][j].set_yticks(np.arange(num_layers))
                        ax[i][j].set_xticklabels(np.arange(data.shape[0]))
                        ax[i][j].set_yticklabels(np.arange(num_layers))
                    else:
                        for color, d, label in zip(color_idx, data.T, labels):
                            ax[i][j].plot(d, color=color_map(color), label=label)
                        ax[i][j].set_xticks(np.arange(data.shape[0]))
                        ax[i][j].grid(linestyle="dashed", linewidth="0.5", color="gray")
                        max_value = max(max_value, data.max())
                    ax[i][j].set_title(list(current_data.keys())[i * single_task_cols + j])

                    data_default = current_data[list(current_data.keys())[i * single_task_cols + j]]["default"]["mean"]
                    data_optimal = current_data[list(current_data.keys())[i * single_task_cols + j]]["optimal"]["mean"]
                    ratio = np.divide(data_default, data_optimal)
                    """
                    print("{} (with {}): mean = {}, std = {} ({})"
                          .format(plot_titles[comp_choice] if comp_choice != "single_single" else plot_titles[comp_choice]
                                  .format(list(mn for mn in single_task_names if mn not in current_data)[0]),
                                  list(current_data.keys())[i * single_task_cols + j],
                                  ratio.mean(), ratio.std(), list(ratio.mean(axis=0))))
                    """

            if not flags.plot_heatmaps:
                plt.ylim((0, max_value * 1.1))
            fig.text(0.5, 0.04, "Model checkpoints throughout training", ha="center")
            fig.text(0.02, 0.5, "SSD ({}) between corresponding filters ({} match)"
                     .format(flags.plot_metric_type, flags.plot_match_type), va="center", rotation="vertical")
            fig.suptitle(plot_titles[comp_choice] if comp_choice != "single_single" else plot_titles[comp_choice]
                         .format(list(mn for mn in single_task_names if mn not in current_data)[0]))
            if flags.save_figures:
                plt.savefig(os.path.join(flags.load_path, "{}_{}_{}{}.png"
                                         .format(comp_choice, flags.plot_match_type, flags.plot_metric_type,
                                                 "" if comp_choice != "single_single" else
                                                 list(mn for mn in single_task_names if mn not in current_data)[0])))

        if not flags.hide_plots:
            plt.show()


def _filter_comp(flags):
    paths = flags.load_path.split(",")
    if len(paths) == 1:
        raise ValueError("Need to supply paths to two models for filter comparison.")
    models = load_models(paths)

    filter_list = [[] for _ in models]
    for m_idx, m in enumerate(models):
        for i in range(0, len(m.feat_convs)):
            filter_list[m_idx].append(m.feat_convs[i][0].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet1[i][1].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet1[i][3].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet2[i][1].weight.detach().numpy())
            filter_list[m_idx].append(m.resnet2[i][3].weight.detach().numpy())

    diffs = [[] for _ in models]
    for filters in filter_list[1:]:
        for f_idx, f in enumerate(filters):
            diffs = np.zeros(f.shape[:2])
            for oc_idx, out_channel in enumerate(f):
                for ic_idx, in_channel in enumerate(out_channel):
                    s = ((in_channel - filter_list[0][f_idx][oc_idx][ic_idx]) ** 2).sum()
                    diffs[oc_idx, ic_idx] = s
            plt.imshow(diffs, cmap="hot", interpolation="nearest")
            plt.show()

            all_filters = f.shape[0] * f.shape[1]
            filter_size = f.shape[2] * f.shape[3]
            original = np.reshape(filter_list[0][f_idx], (all_filters, filter_size))
            comparison = np.reshape(f, (all_filters, filter_size))
            distances = cdist(original, comparison, metric="sqeuclidean")

            row_idx, col_idx = linear_sum_assignment(distances)
            distance_sum = distances[row_idx, col_idx].sum()
            print("Total distance sum:", distance_sum)
            print("Without matching:", distances[0, :].sum())
            """
            ax = sns.heatmap(distances)
            for r_idx, c_idx in zip(row_idx, col_idx):
                ax.add_patch(Rectangle((r_idx, c_idx), 1, 1, fill=False, edgecolor="blue", lw=1))
            """
            plt.show()


def load_models(paths):
    models = []
    for p in paths:
        # load parameters
        checkpoint = torch.load(p, map_location="cpu")

        # determine input for building the model
        if "baseline.mu" not in checkpoint["model_state_dict"]:
            checkpoint["model_state_dict"]["baseline.mu"] = torch.zeros(1)
            checkpoint["model_state_dict"]["baseline.sigma"] = torch.ones(1)
            num_tasks = 1
        else:
            num_tasks = checkpoint["model_state_dict"]["baseline.mu"].shape[0]
        num_actions = checkpoint["model_state_dict"]["policy.weight"].shape[0]

        # construct model and transfer loaded parameters
        model = ResNetMono(observation_shape=None,
                           num_actions=num_actions,
                           num_tasks=num_tasks,
                           use_lstm=False,
                           use_popart=True,
                           reward_clipping="abs_one")
        model.eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        models.append(model)

    return models if len(models) > 1 else models[0]


if __name__ == '__main__':
    logging.basicConfig(format="[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s", level=0)

    parser = argparse.ArgumentParser(description="Visualizations for the ResNet")
    parser.add_argument("--load_path", default="./logs/torchbeast",
                        help="Path to the model (or other data) that should be used for the visualizations.")
    parser.add_argument("--save_dir", default="~/logs/resnet_vis",
                        help=".")
    parser.add_argument("--mode", type=str, default="filter_vis",
                        choices=["filter_vis", "filter_comp", "_filter_comp", "filter_comp_plot"],
                        help="What visualizations to create.")
    parser.add_argument("--layer_index", type=int, default=0,
                        help="Layer for which to visualize a filter.")
    parser.add_argument("--filter_index", type=int, default=0,
                        help="Filter to visualize (only in mode 'filter_vis').")
    parser.add_argument("--pairwise_comp", action="store_true",
                        help="Visualise difference between all pairwise filters, "
                             "not just corresponding ones (only in mode '_filter_comp').")

    # filter_comp parameters
    parser.add_argument("--match_num_models", action="store_true",
                        help="When comparing between single-task and multi-task models, compare between models "
                             "trained for the same number of steps instead of percentage of training time. "
                             "(NOTE: currently not properly implemented)")
    parser.add_argument("--comp_num_models", type=int, default=10,
                        help="Number of model checkpoints to load and to compare.")
    parser.add_argument("--comp_no_optimal", action="store_true",
                        help="Do not compute the mean SSDs for the optimal matching "
                             "between filters (which can take a long time)")
    parser.add_argument("--comp_between", type=str, nargs="+", choices=comparison_choices,
                        default=["single_multi", "single_multipop", "multi_multipop"],
                        help="List what types of comparisons should be made (choices mostly self-explanatory).")
    parser.add_argument("--comp_single_single_model", type=str, default="Carnival", choices=single_task_names,
                        help="When comparing single-task models, which model to take "
                             "as the reference to compare the other models with.")

    # filter_comp_plot parameters
    parser.add_argument("--plot_match_type", type=str, default="default", choices=["default", "optimal"],
                        help="Which type of matching between filters to plot (default or optimal).")
    parser.add_argument("--plot_metric_type", type=str, default="mean", choices=["mean", "sum"],
                        help="Which metric to plot (sum or mean of SSDs in each layer).")
    parser.add_argument("--plot_heatmaps", action="store_true",
                        help="Plot heatmaps instead of normal plots.")
    parser.add_argument("--save_figures", action="store_true",
                        help="Save the generated figures to the directory which the "
                             "data was loaded from (specified with --load_path).")
    parser.add_argument("--hide_plots", action="store_true",
                        help="Do not display the plots. Mostly useful if one just "
                             "wants to generate the figures to save them.")

    # correct model params
    parser.add_argument("--frame_height", type=int, default=84,
                        help="Height to which frames are rescaled.")
    parser.add_argument("--frame_width", type=int, default=84,
                        help="Width to which frames are rescaled.")
    parser.add_argument("--num_actions", type=int, default=6,
                        help="The number of actions of the loaded model(s).")

    flags = parser.parse_args()

    if flags.mode == "filter_vis":
        filter_vis(flags)
    elif flags.mode == "_filter_comp":
        _filter_comp(flags)
    elif flags.mode == "filter_comp":
        filter_comp(flags)
    elif flags.mode == "filter_comp_plot":
        filter_comp_plot(flags)
