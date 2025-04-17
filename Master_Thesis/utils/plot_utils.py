from typing import List
import os

import torch
import torchvision.utils as vutils

import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import plotly.express as px
import plotly.graph_objects as go
import mpl_toolkits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.io import write_image
import plotly.io as pio   
pio.kaleido.scope.mathjax = None
import pandas as pd

from evaluation.evaluation_utils import load_npz_files


def get_railsem19_plotting_elements():
    """
    Generates the class mappings, color mappings, colormap, normalization, and legend patches for plotting.

    Returns:
        class_mapping (dict): Mapping of class indices to class names.
        color_mapping (dict): Mapping of class indices to RGB color values.
        cmap (colors.ListedColormap): Colormap for the classes.
        norm (colors.BoundaryNorm): Normalization for the colormap.
        patches (list): List of patches for creating a legend.
    """
    class_mapping = {
        0: ["road"],
        1: ["sidewalk"],
        2: ['construction', 'fence'],
        3: ['rail_raised', 'rail_embedded'],
        4: ['pole', 'traffic_light', 'traffic_sign'],
        5: ['sky'],
        6: ['human'],
        7: ['tram_track', 'rail_track'],
        8: ['car', 'truck'],
        9: ['on_rails'],
        10: ['vegetation'],
        11: ['trackbed'],
        12: ['background', 'terrain']
    }
    palette = {
        0: [128, 64, 128], 1: [244, 35, 232], 2: [190, 153, 153], 3: [250, 170, 30], 4: [153, 153, 153],
        5: [70, 130, 180], 6: [220, 20, 60], 7: [119, 11, 32], 8: [0, 0, 142], 9: [0, 80, 100],
        10: [107, 142, 35], 11: [53, 74, 74], 12: [152, 251, 152]
    }
    # Define a consistent color mapping using RGB values normalized to [0, 1]
    color_mapping = {
        0: np.array([128, 64, 128]) / 255,        # Road                             # Greyish Pink
        1: np.array([244, 35, 232]) / 255,        # SideWalk                         # Magenta
        2: np.array([190, 153, 153]) / 255,       # Construction/Fence               # Greyish Light Red
        3: np.array([250, 170, 30]) / 255,         # Rail_raised/Rail_embedded       # Orange
        4: np.array([153, 153, 153]) / 255,       # Pole/Traffic_light/Traffic_sign  # Light Gray
        5: np.array([70, 130, 180]) / 255,        # Sky                              # Sky Blue
        6: np.array([220, 20, 60]) / 255,         # Human                            # Pinkish Red
        7: np.array([119, 11, 32]) / 255,         # Tram_track/Rail_track            # Marron
        8: np.array([0, 0, 142]) / 255,           # Car/Truck                        # Dark Blue
        9: np.array([0, 80, 100]) / 255,          # On_rails                         # Blackish Light Blue
        10: np.array([107, 142, 35]) / 255,       # Vegetation                       # Parrot Green
        11: np.array([53, 74, 74]) / 255,         # Trackbed                         # Dark Gray
        12: np.array([152, 251, 152]) / 255       # Background/terrain               # Radium Green
    }
    cmap = colors.ListedColormap([color_mapping[i] for i in range(len(class_mapping.keys()))])
    bounds = [i for i in range(0, len(class_mapping.keys()))]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    patches = [mpatches.Patch(color=cmap.colors[i], label=str(class_mapping[i])) for i in range(len(class_mapping.keys()))]
    return class_mapping, color_mapping, cmap, norm, patches, palette


def get_cityscapes_plotting_elements():
    """
    Generates the class mappings, color mappings, colormap, normalization, and legend patches for plotting Cityscapes dataset.

    Returns:
        class_mapping (dict): Mapping of class indices to class names.
        color_mapping (dict): Mapping of class indices to RGB color values.
        cmap (colors.ListedColormap): Colormap for the classes.
        norm (colors.BoundaryNorm): Normalization for the colormap.
        patches (list): List of patches for creating a legend.
    """
    class_mapping = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic light",
        7: "traffic sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle",
    }
    palette = {
        0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70], 3: [102, 102, 156], 4: [190, 153, 153],
        5: [153, 153, 153], 6: [250, 170, 30], 7: [220, 220, 0], 8: [107, 142, 35], 9: [152, 251, 152],
        10: [70, 130, 180], 11: [220, 20, 60], 12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70],
        15: [0, 60, 100], 16: [0, 80, 100], 17: [0, 0, 230], 18: [119, 11, 32]
    }

    # Define a consistent color mapping using RGB values normalized to [0, 1]
    color_mapping = {
        0: np.array([128, 64, 128]) / 255,        # / 255 # Road     # Greyish Pink
        1: np.array([244, 35, 232]) / 255,        # Sidewalk         # Magenta
        2: np.array([70, 70, 70]) / 255,          # Building         # Gray
        3: np.array([102, 102, 156]) / 255,       # Wall             # Lavendor Blue
        4: np.array([190, 153, 153]) / 255,       # Fence            # Greyish Light Red
        5: np.array([153, 153, 153]) / 255,       # Pole             # Light Gray
        6: np.array([250, 170, 30]) / 255,        # Traffic light    # Orange
        7: np.array([220, 220, 0]) / 255,         # Traffic sign     # Lime Yellow
        8: np.array([107, 142, 35]) / 255,        # Vegetation       # Parrot Green
        9: np.array([152, 251, 152]) / 255,       # Terrain          # Radium Green
        10: np.array([70, 130, 180]) / 255,       # Sky              # Sky Blue
        11: np.array([220, 20, 60]) / 255,        # Person           # Pinkish Red
        12: np.array([255, 0, 0]) / 255,          # Rider            # Red
        13: np.array([0, 0, 142]) / 255,          # Car              # Dark Blue
        14: np.array([0, 0, 70]) / 255,           # Truck            # Navy Blue
        15: np.array([0, 60, 100]) / 255,         # Bus              # Greyish Light Blue
        16: np.array([0, 80, 100]) / 255,         # Train            # Blackish Light Blue
        17: np.array([0, 0, 230]) / 255,          # Motorcycle       # Royal Bright Blue
        18: np.array([119, 11, 32]) / 255,        # Bicycle          # Maroon
    }
    cmap = colors.ListedColormap([color_mapping[i] for i in range(len(class_mapping))])
    # Create a BoundaryNorm for discrete classes
    bounds = [i for i in range(len(class_mapping) + 1)]  # # Class boundaries: [0, 1, 2, ..., len(class_names)]
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    patches = [mpatches.Patch(color=cmap.colors[i], label=class_mapping[i]) for i in range(len(class_mapping))]
    return class_mapping, color_mapping, cmap, norm, patches, palette


def get_ood_plotting_elements():
    """
    Generates the class mappings, color mappings, colormap, normalization, and legend patches for plotting ood datasets
    except `lostandfound` and `streethazards` dataset.
    Allowed Datasets are:
    
    - lostandfound        -     Unique labels {0, 1, 255}
    - fishyscapes         -     Unique labels {0, 1, 255}
    - anamoly             -     Unique labels {0, 1, 255}
    - obstacles           -     Unique labels {0, 1, 255}

    Returns:
        class_mapping (dict): Mapping of class indices to class names.
        color_mapping (dict): Mapping of class indices to RGB color values.
        cmap (colors.ListedColormap): Colormap for the classes.
        norm (colors.BoundaryNorm): Normalization for the colormap.
        patches (list): List of patches for creating a legend.
    """
    class_mapping = {
        0: "id",
        1: "ood",
        255: "background"
    }
    # Define a consistent color mapping using RGB values normalized to [0, 1]
    color_mapping = {
        0: np.array([0, 100, 255]) / 255,      # InDsitribution     # Blue
        1: np.array([255, 0, 0]) / 255,        # OODistribution     # Red
        255: np.array([0, 0, 0]) / 255,        # Background         # Black
    }
    palette = {0: [0, 100, 255], 1: [255, 0, 0], 255: [0, 0, 0]}
    # Create a ListedColormap
    cmap = colors.ListedColormap([color_mapping[k] for k in sorted(class_mapping.keys())])
    # Define bounds to cover all class indices
    bounds = [i for i in sorted(class_mapping.keys())]
    # Create a BoundaryNorm
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # Create legend patches
    patches = [mpatches.Patch(color=cmap.colors[k], label=class_mapping[k]) for k in sorted(class_mapping.keys())]
    return class_mapping, color_mapping, cmap, norm, patches, palette


def get_ood_lostandfound_all_plotting_elements():
    """
    Generates the class mappings, color mappings, colormap, normalization, and legend patches for plotting lostandfound ood dataset.
    - lostandfound        -     Unique labels {0, 1, 2}
    
    Returns:
        class_mapping (dict): Mapping of class indices to class names.
        color_mapping (dict): Mapping of class indices to RGB color values.
        cmap (colors.ListedColormap): Colormap for the classes.
        norm (colors.BoundaryNorm): Normalization for the colormap.
        patches (list): List of patches for creating a legend.
    """
    class_mapping = {
        0: "id",
        1: "ood",
        2: "background"
    }
    palette = {0: [0, 100, 255], 1: [255, 0, 0], 2: [0, 0, 0]}
    # Define a consistent color mapping using RGB values normalized to [0, 1]
    color_mapping = {
        0: np.array([0, 100, 255]) / 255,      # InDsitribution     # Blue
        1: np.array([255, 0, 0]) / 255,        # OODistribution     # Red
        2: np.array([0, 0, 0]) / 255,          # Background         # Black
    }
    cmap = colors.ListedColormap([color_mapping[i] for i in range(len(class_mapping))])
    bounds = [i for i in range(len(class_mapping) + 1)]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    patches = [mpatches.Patch(color=cmap.colors[i], label=class_mapping[i]) for i in range(len(class_mapping))]
    return class_mapping, color_mapping, cmap, norm, patches, palette


def get_ood_streethazards_plotting_elements():
    """
    StreetHazards is a synthetic dataset for anomaly detection, created by inserting a diverse array of foreign objects 
    into driving scenes and re-render the scenes with these novel objects.
    
    Generates the class mappings, color mappings, colormap, normalization, and legend patches for plotting streethazards ood dataset.
    
    - streethazards   -     Unique labels {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    
    Returns:
        class_mapping (dict): Mapping of class indices to class names.
        color_mapping (dict): Mapping of class indices to RGB color values.
        cmap (colors.ListedColormap): Colormap for the classes.
        norm (colors.BoundaryNorm): Normalization for the colormap.
        patches (list): List of patches for creating a legend.
    """
    class_mapping = {
        1: "building",
        2: "fence",
        3: "other",
        4: "pedestrian",
        5: "pole",
        6: "road line",
        7: "road",
        8: "sidewalk",
        9: "vegetation",
        10: "car",
        11: "wall",
        12: "traffic sign",
        13: "anomaly",
    }
    palette = {
        1: [70,  70,  70], 2: [190, 153, 153], 3: [250, 170, 160], 4: [220, 20, 60], 5: [153, 153, 153],
        6: [157, 234,  50], 7: [128, 64, 128], 8: [244, 35, 232], 9: [107, 142, 35], 10: [0, 0, 142],
        11: [102, 102, 156], 12: [220, 220, 0], 13: [60, 250, 240]
    }
    # Define a consistent color mapping using RGB values normalized to [0, 1]
    color_mapping = {
        1: np.array([70,  70,  70]) / 255,       # 1 - building       # Gray
        2: np.array([190, 153, 153]) / 255,      # 2 - Fence          # Greyish Light Red
        3: np.array([250, 170, 160]) / 255,      # 3 - other          # LIght Peach
        4: np.array([220, 20, 60]) / 255,        # 4 - pedestrian     # Pinkish Red
        5: np.array([153, 153, 153]) / 255,      # 5 - pole           # Black
        6: np.array([157, 234,  50]) / 255,      # 6 - road line      # Lemon Green
        7: np.array([128, 64, 128]) / 255,       # 7 - road           # Greyish Pink
        8: np.array([244, 35, 232]) / 255,       # 8 - sidewalk       # Magenta
        9: np.array([107, 142, 35]) / 255,       # 9 - vegetation     # Parrot Green
        10: np.array([0, 0, 142]) / 255,         # 10 - car           # Dark Blue
        11: np.array([102, 102, 156]) / 255,     # 11 - wall          # Lavendor Blue
        12: np.array([220, 220, 0]) / 255,       # 12 - traffic sign  # Lime Yellow
        13: np.array([60, 250, 240]) / 255,      # 13 - anomaly       # Cyan      
    }
    # Create a ListedColormap
    cmap = colors.ListedColormap([color_mapping[i] for i in sorted(class_mapping.keys())])
    # Define bounds to cover all class indices
    bounds = [i for i in sorted(class_mapping.keys())]
    # Create a BoundaryNorm
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # Create legend patches
    patches = [mpatches.Patch(color=color_mapping[i], label=class_mapping[i]) for i in sorted(class_mapping.keys())]
    return class_mapping, color_mapping, cmap, norm, patches, palette


def get_plotting_elements(dataset_name):
    """
    Get the cmaps for plottings
    """
    if dataset_name == "railsem19":
        return get_railsem19_plotting_elements()
    elif dataset_name == "cityscapes":
        return get_cityscapes_plotting_elements()
    elif dataset_name in ["lostandfound", "fishyscapes", "obstacles", "anamoly"]:
        return get_ood_plotting_elements()
    elif dataset_name == "lostandfound_all":
        return get_ood_lostandfound_all_plotting_elements()
    elif dataset_name == "streethazards":
        return get_ood_streethazards_plotting_elements()
    else:
        raise ValueError(f"Unable to retrieve Plotting Elements. Invalid dataset name: {dataset_name}")


def unnormalize_image(img, dataset_name, clip=False):
    """
    Unnormalize the images before plotting
    """
    if dataset_name == "railsem19":
        img *= 255
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device).view(3, 1, 1)
        img = img * std + mean
        if clip:
            img = torch.clamp(img, 0, 1)
    return img


def get_flat_palette(palette_dict):
    """Converts a palette dictionary to a flat list for PIL."""
    flat_palette = [0] * 768  # 256 colors, 3 values each
    for idx, color in palette_dict.items():
        flat_palette[idx * 3: idx * 3 + 3] = color
    return flat_palette


def apply_color_palette(mask, flat_palette):
    """
    Convert a single-channel mask to an RGB image using the color palette.
    """
    mask_image = PIL.Image.fromarray(mask.numpy().astype(np.uint8))
    mask_image.putpalette(flat_palette)
    return np.array(mask_image.convert('RGB'))


def plot_validation_predictions(val_data, writer, epoch, dataset_name, clip=False):
    original_images, targets, pred_segmaps = zip(*val_data)
    original_images = torch.stack(original_images).detach().cpu()
    targets = torch.stack(targets).detach().cpu()
    pred_segmaps = torch.stack(pred_segmaps).detach().cpu()
    _, _, _, _, _, palette = get_plotting_elements(dataset_name)
    flat_palette = get_flat_palette(palette)

    for i, (img, gt_mask, pred_mask) in enumerate(zip(original_images, targets, pred_segmaps)):
        img = unnormalize_image(img, dataset_name, clip)
        img_np = img.permute(1, 2, 0).numpy()  # Convert to HWC format
      
        gt_mask_color = apply_color_palette(gt_mask, flat_palette)
        pred_mask_color = apply_color_palette(pred_mask, flat_palette)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        axes[1].imshow(gt_mask_color, interpolation='nearest')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        axes[2].imshow(pred_mask_color, interpolation='nearest')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        plt.tight_layout()

        # Convert matplotlib figure to numpy array
        fig.canvas.draw()
        fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        fig_np = fig_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        # Log the figure to TensorBoard
        writer.add_image(f"SelectedIndices/Image_{i}", fig_np, epoch, dataformats='HWC')


# Test Related Plots
def plot_risks_curves(x_vals, y_vals, aurrc, x_title="Rejection Rates", y_title="Risks"):
    """
    Function to plot Area Under Risk Rejection Curves for a test set.

    - Risks vs Rejection Rates
    - Risks vs Selection Thresholds
    """
    print(x_vals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=f"{x_title}_vs_{y_title}"))
    # Update layout
    fig.update_layout(title=f"{x_title} vs {y_title}", xaxis_title=x_title, yaxis_title=y_title)
    fig.add_annotation(text=f"AURRC: {aurrc}", x=0.5, y=0.5, xanchor="center", yanchor="middle")
    return fig


def plot_cnfmat(cnfmat, labels=None):
    """
    Plot the confusion matrix for a test set.
    """
    # Ensure labels are provided
    if labels is None:
        labels = [str(i) for i in range(cnfmat.shape[0])]
    fig = go.Figure()
    # Convert tensor to numpy for compatibility with plotly
    cnfmat = cnfmat.cpu().numpy()
    # Create the heatmap
    heatmap = go.Heatmap(z=cnfmat, x=labels, y=labels, hoverongaps=False, colorscale='Viridis', showscale=True)
    fig.add_trace(heatmap)
    # Add annotations
    annotations = []
    for i in range(cnfmat.shape[0]):
        for j in range(cnfmat.shape[1]):
            annotations.append(
                go.layout.Annotation(x=labels[j], y=labels[i], text=str(round(cnfmat[i, j].item(), 2)),
                                     showarrow=False, font=dict(color='white' if cnfmat[i, j] > cnfmat.max() / 2 else 'black'), xref="x1", yref="y1")
            )
    # Update layout
    fig.update_layout(
        xaxis=dict(title='Predicted Labels', tickmode='array', tickvals=list(range(len(labels))), ticktext=labels, tickangle=45, side='bottom'),
        yaxis=dict(title='True Labels', tickmode='array', tickvals=list(range(len(labels))), ticktext=labels, automargin=True),
        title='Confusion Matrix', annotations=annotations, autosize=False, width=1000, height=1000, margin=dict(l=150, r=150, b=200, t=150)
    )
    return fig

# Unseen and Test set related plots
def plot_error_map(pred_segmentation, true_segmentation, return_fig=False):
    """Binary Error map for a prediction
    """
    # Convert predictions and ground truth to numpy arrays
    pred_np = pred_segmentation.cpu().numpy()
    true_np = true_segmentation.cpu().numpy()
    # Create error map (1 where there is an error, 0 where correct)
    error_map = (pred_np != true_np).astype(int)
    if return_fig:
        # Create Plotly figure
        fig = go.Figure()
        # Add heatmap trace for error map
        fig.add_trace(go.Heatmap(z=error_map, colorscale=[[0, 'white'], [1, 'red']], showscale=True, colorbar=dict(title="Error", tickvals=[0, 1], ticktext=["Correct", "Error"])))
        # Update layout
        fig.update_layout(title="Error Map", xaxis_visible=False, yaxis_visible=False)
        return fig
    return error_map


def plot_max_softmax_probability_map(max_pred_probs, return_fig=False):
    """Plot maximum softmax probability map for a prediction
    """
    # Get maximum softmax probability across classes
    max_probs = max_pred_probs.cpu().numpy()
    # Create Plotly figure
    if return_fig:
        fig = go.Figure()
        # Add heatmap trace for max softmax probabilities
        fig.add_trace(go.Heatmap(z=max_probs, colorscale='Viridis', showscale=True, colorbar=dict(title="Max Softmax Probability")))
        # Update layout
        fig.update_layout(title="Maximum Softmax Probability Map", xaxis_visible=False, yaxis_visible=False)
        return fig
    return max_probs


def plot_predictions(image, gt_mask_array, pred_mask, binary_error_map, max_probs_map, entropy_map, uncertainty_map,
                     belief, original_height, original_width, dataset_name):
    """
    Used for predicting the test images or the new unseen image prediction.

    Plots the image, ground truth mask if exists, Prediction Mask, Binary Error Map if ground truth exists, Maximum Softmax Probability Map, Entropy Map, Uncertainty Map of MCD if exists
    """
    print("[INFO]: Plotting predictions")
    _, _, cmap, norm, _, _ = get_plotting_elements(dataset_name)
    fig, axes = plt.subplots(1, 8, figsize=(25, 5))
    # Convert to (H, W, C) format
    img_transposed = image.squeeze(dim=0)
    img_transposed = unnormalize_image(img_transposed, dataset_name=dataset_name)
    img_transposed = img_transposed.permute(1, 2, 0)
    axes[0].imshow(img_transposed)
    axes[0].set_title('Original Image')
    if gt_mask_array is not None:
        print(f"gt_mask_array: {gt_mask_array.shape}")
        palette = get_cityscapes_plotting_elements()[5]  # Get the palette from the function
        colored_mask = np.zeros((gt_mask_array.shape[0], gt_mask_array.shape[1], 3), dtype=np.uint8)
        
        # print(np.unique(gt_mask_array))  # [  2   3   5   8  10 255]
        # Apply palette to mask
        for label, color in palette.items():
            colored_mask[gt_mask_array == label] = color

        axes[1].imshow(colored_mask, interpolation="nearest") 
        
        # axes[1].imshow(gt_mask_array, cmap=cmap, norm=norm, interpolation="nearest")
        axes[1].set_title('Ground Truth Mask')
        axes[3].imshow(binary_error_map, cmap='gray', interpolation="nearest")
        axes[3].set_title('Binary Error Map')
    axes[2].imshow(pred_mask, cmap=cmap, norm=norm, interpolation="nearest")
    axes[2].set_title('Predicted Mask')
    axes[4].imshow(max_probs_map, cmap='viridis', interpolation="nearest")
    axes[4].set_title('Max Softmax Probability Map')
    axes[5].imshow(entropy_map, cmap='jet', interpolation="nearest")
    axes[5].set_title('Entropy Map')
    if uncertainty_map is not None:
        axes[6].imshow(uncertainty_map, cmap='hot', interpolation="nearest")
        if belief:
            axes[6].set_title('EDL Uncertainty')
        else:
            axes[6].set_title('Predictive Variance Map')
    for ax in axes:
        ax.axis('off')
    return fig

def plot_predictions_individually(image, gt_mask_array, pred_mask, binary_error_map, max_probs_map, entropy_map,
                                  uncertainty_map, belief, original_height, original_width, dataset_name):
    """
    Generates figures for predictions and returns them as a dictionary of matplotlib figures. This function 
    helps to save every figure as separate plot instead of saving everything to a same figure object.
    
    Parameters:
    - image, gt_mask_array, pred_mask, binary_error_map, max_probs_map, entropy_map, uncertainty_map, belief, dataset_name:
      Same as original function parameters.

    Returns:
    - A dictionary of matplotlib figures for each component.
    """
    print("[INFO]: Plotting predictions individually")
    _, _, cmap, norm, _, _ = get_plotting_elements(dataset_name)
    img_transposed = image.squeeze(dim=0)
    img_transposed = unnormalize_image(img_transposed, dataset_name=dataset_name)
    img_transposed = img_transposed.permute(1, 2, 0)
    
    figures = {}
    
    # Original Image
    fig_original = plt.figure(figsize=(10, 8))
    # Unpad Image
    img_unpadded = img_transposed[:original_height, :original_width]
    plt.imshow(img_unpadded)  # img_transposed
    plt.axis('off')
    figures['original_image'] = fig_original
    
    if gt_mask_array is not None:
        # Unpad Ground Truth Mask
        gt_mask_unpadded = gt_mask_array[:original_height, :original_width]
        # Ground Truth Mask
        palette = get_cityscapes_plotting_elements()[5]
        colored_mask = np.zeros((gt_mask_unpadded.shape[0], gt_mask_unpadded.shape[1], 3), dtype=np.uint8)
        for label, color in palette.items():
            colored_mask[gt_mask_unpadded == label] = color

        fig_gt = plt.figure(figsize=(10, 8))
        plt.imshow(colored_mask, 
                   cmap=cmap,  # Use the colormap defined for cityscapes
                    norm=norm, # Map class labels to specific colors
                    interpolation="nearest", )
        plt.axis('off')
        figures['ground_truth_mask'] = fig_gt

        # Binary Error Map
        binary_error_unpadded = binary_error_map[:original_height, :original_width]
        fig_error = plt.figure(figsize=(10, 8))
        plt.imshow(binary_error_unpadded, cmap='gray', interpolation="nearest")
        plt.axis('off')
        figures['binary_error_map'] = fig_error

    # Predicted Mask
    pred_mask_unpadded = pred_mask[:original_height, :original_width]
    fig_pred = plt.figure(figsize=(10, 8))
    plt.imshow(pred_mask_unpadded, cmap=cmap, norm=norm, interpolation="nearest")
    plt.axis('off')
    figures['predicted_mask'] = fig_pred
    
    # Max Softmax Probability Map
    max_probs_unpadded = max_probs_map[:original_height, :original_width]
    fig_max_prob, ax = plt.subplots(figsize=(10, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='2%', pad=0.04)
    msp_plot = ax.imshow(max_probs_unpadded, cmap='viridis', vmin=0, vmax=1) # # vmax=np.log2(pred_mask.max())
    ax.set_axis_off()
    cbar = fig_max_prob.colorbar(msp_plot, cax=cax, orientation='horizontal', location="top")
    # cbar.set_label('Max Softmax Probability')
    figures['max_softmax_probability_map'] = fig_max_prob
    
    # Entropy Map
    entropy_unpadded = entropy_map[:original_height, :original_width]
    fig_entropy, ax = plt.subplots(figsize=(10, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='2%', pad=0.04)
    entropy_plot = ax.imshow(entropy_unpadded, cmap='jet', interpolation="nearest", vmin=0, vmax=1)
    ax.set_axis_off()
    cbar = fig_entropy.colorbar(entropy_plot, cax=cax, orientation='horizontal')
    # cbar.set_label('Entropy')
    # Set ticks above the colorbar
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')  # Set label position to top if necessary
    figures['entropy_map'] = fig_entropy

    if uncertainty_map is not None:
        # Unpad Uncertainty Map
        uncertainty_map = uncertainty_map.astype(float)
        uncertainty_unpadded = uncertainty_map[:original_height, :original_width]
        print(f"uncertainty_unpadded dtype: {uncertainty_unpadded.dtype}, {uncertainty_map.dtype}, {np.sum(np.isnan(uncertainty_map))}")
        print(f"Array dtype: {uncertainty_map.dtype}")
        print(f"uncertainty_unpadded dtype: {uncertainty_unpadded.dtype}")
        # Uncertainty Map
        fig_uncertainty, ax = plt.subplots(figsize=(10, 8))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='4%', pad=0.1)
        uncertainty_plot = ax.imshow(uncertainty_unpadded, cmap='hot', interpolation="nearest", vmin=0, vmax=1)
        ax.set_axis_off()
        cbar = fig_uncertainty.colorbar(uncertainty_plot, cax=cax, orientation='horizontal')
        ## cbar.set_label('Uncertainty')
        # Set ticks above the colorbar
        cbar.ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # Set 6 ticks for the colorbar
        cbar.ax.xaxis.set_major_formatter(plt.ScalarFormatter("%.1f"))  # Format tick labels with 2 decimal places 
        # Set ticks above the colorbar
        cbar.ax.xaxis.set_ticks_position('top')  # Set tick positions to top
        cbar.ax.xaxis.set_label_position('top')  # Set label position to top if necessary
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Ensure numeric values
        cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])  # Convert to explicit string labels
        ## plt.title('EDL Uncertainty' if belief else 'Predictive Variance Map')
        figures['uncertainty_map'] = fig_uncertainty
    
    return figures


# OOD Plotting functions
def plot_roc_curve(roc_curve_data, auroc, fpr_at_recall_level, recall_level):
    """
    Plot AUROC curve for OOD Test.
    """
    print("[INFO]: Plotting ROC curve...")
    fig = go.Figure()
    # ROC Curve data Unpacking
    fpr, tpr, _ = roc_curve_data
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (area = {auroc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=[fpr_at_recall_level], y=[recall_level], mode='markers',
                             marker=dict(color='red', size=10),
                             name=f'FPR@Recall ({recall_level}): {fpr_at_recall_level:.2f}'))
    # Annotation for AUROC
    fig.add_annotation(x=0.95, y=0.1, text=f'AUROC = {auroc:.3f}', showarrow=False, font=dict(size=12, color="black"))

    # Update layout
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",)  # template="plotly_white"
    return fig


def plot_roc_curve_multiple(roc_data_list, auroc_list, fpr_recall_list, recall_level=0.95):
    """
    Plot combined ROC curves for multiple datasets and include AUROC and FPR@95 in the legend.
    Args:
        roc_data_list: List of tuples (roc_curve_data, label) for each dataset.
        auroc_list: List of tuples (auroc, label) for each dataset.
        fpr_recall_list: List of tuples (fpr_at_recall_level, label) for each dataset.
        recall_level: Recall level used for evaluation (default set to 95%).
    Returns:
        Combined figure with ROC curves for all datasets.
    """
    print("[INFO]: Plotting combined ROC curves...")
    
    fig = go.Figure()

    # Add ROC curve for each dataset
    for (roc_curve_data, label), (auroc, _), (fpr_at_recall_level, _) in zip(roc_data_list, auroc_list, fpr_recall_list):
        fpr, tpr = roc_curve_data
        
        # Add ROC curve trace to the figure
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{label}: AUROC={auroc:.2f}, FPR@{recall_level*100:.0f}={fpr_at_recall_level:.2f}'
            )
        )
        
        # Add FPR marker for this dataset
        fig.add_trace(
            go.Scatter(
                x=[fpr_at_recall_level],
                y=[recall_level],
                mode='markers',
                marker=dict(color='red', size=5),
                name=f'{label} FPR@Recall ({recall_level}): {fpr_at_recall_level:.2f}'
            )
        )

    # Add Random Classifier baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash')
        )
    )

    # Update layout to configure legend and axes
    fig.update_layout(
        title="Combined ROC Curves for Multiple Datasets",
        xaxis=dict(
            title="False Positive Rate",
            showline=True, linewidth=1, linecolor="black", mirror=True,
            showgrid=True, gridcolor="lightgray",
        ),
        yaxis=dict(
            title="True Positive Rate",  # ylabel
            showline=True, linewidth=1, linecolor="black", mirror=True,
            showgrid=True, gridcolor="lightgray",
            title_font=dict(size=18),  # Increase ylabel font size
            tickfont=dict(size=14),  # Increase tick font size for y-axis
            range=[0.6, 1]
        ),
        showlegend=True,  # Enable legend
        legend=dict(
            title="<b>Legend</b>",
            x=0.99, y=0.01,  # Place legend in bottom-right corner
            xanchor="right", yanchor="bottom",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=12)
        ),
        plot_bgcolor="white",  # Set background to white
        paper_bgcolor="white"  # Set paper background to white
    )

    return fig


def plot_aupr_curve(aupr_curve_data, aupr, title="in"):
    """
    Plot AUPR curve for OOD Test.
    """
    print("[INFO]: Plotting AUPR curve...")
    fig = go.Figure()
    # AUPR Curve data Unpacking
    precision, recall, _ = aupr_curve_data
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'AUPR-{title} Curve'))
    # Annotation for AUPR and AUROC
    fig.add_annotation(x=0.5, y=0.1, text=f'AUPR-{title} = {aupr:.3f}', showarrow=False, font=dict(size=12, color="black"))
    # Update layout
    fig.update_layout(title=f"AUPR-{title} Curve", xaxis_title="Recall", yaxis_title="Precision", )  # template="plotly_white"
    return fig


def plot_entropy_distribution(ood_labels_dir, ood_entropy_baseline_dir, ood_entropy_improv_baseline_dir, baseline_mdl_title="Baseline", improv_mdl_title="MCD_OHEM", title="Entropy Distribution for In-Distribution and Out-of-Distribution Pixels"):
    """
    Plots the OOD Dataset Entropy Distribution of Baseline Model and Improved Model as Violin Plots for comparison.
    """
    print("[INFO]: Plotting entropy distribution...")
    # Load and entropies for Baseline model and Our Model
    labels = load_npz_files(ood_labels_dir, convert_to_torch=False)
    entropies_baseline = load_npz_files(ood_entropy_baseline_dir, convert_to_torch=False)
    entropies_improv = load_npz_files(ood_entropy_improv_baseline_dir, convert_to_torch=False)
    # Filter and Flatten entropies
    id_entropies_baseline = entropies_baseline[labels == 0].flatten()
    id_entropies_improv = entropies_improv[labels == 0].flatten()
    ood_entropies_baseline = entropies_baseline[labels == 1].flatten()
    ood_entropies_improv = entropies_improv[labels == 1].flatten()
    
    # Sample a subset for plotting to avoid too much data
    max_sample_size = 500000  # Adjust based on your memory limits
    
     # For ID (In-Distribution) - Get random indices
    id_sample_size = min(len(id_entropies_baseline), max_sample_size)
    id_indices = np.random.choice(len(id_entropies_baseline), id_sample_size, replace=False)

    # For OOD (Out-of-Distribution) - Get random indices
    ood_sample_size = min(len(ood_entropies_baseline), max_sample_size)
    ood_indices = np.random.choice(len(ood_entropies_baseline), ood_sample_size, replace=False)

    # Sample the same indices for baseline and improved models
    id_entropies_baseline = id_entropies_baseline[id_indices]
    id_entropies_improv = id_entropies_improv[id_indices]
    ood_entropies_baseline = ood_entropies_baseline[ood_indices]
    ood_entropies_improv = ood_entropies_improv[ood_indices]
    
    # Create violin plot
    fig = go.Figure()
    # In-Distribution data
    fig.add_trace(go.Violin(x=["In-Distribution"] * len(id_entropies_baseline), y=id_entropies_baseline,
                            legendgroup=baseline_mdl_title, scalegroup=baseline_mdl_title, name=baseline_mdl_title,  # 'Baseline'
                            side='negative', line_color='orange', fillcolor='orange'))
    fig.add_trace(go.Violin(x=["In-Distribution"] * len(id_entropies_improv), y=id_entropies_improv,
                            legendgroup=improv_mdl_title, scalegroup=improv_mdl_title, name=improv_mdl_title,
                            side='positive', line_color='blue', fillcolor='rgba(0,0,255,0.5)'))
    # Out-Distribution data
    fig.add_trace(go.Violin(x=["Out-Distribution"] * len(ood_entropies_baseline), y=ood_entropies_baseline,
                            legendgroup=baseline_mdl_title, scalegroup=baseline_mdl_title, name=baseline_mdl_title,
                            side='negative', line_color='orange', fillcolor='orange', showlegend=False))
    fig.add_trace(go.Violin(x=["Out-Distribution"] * len(ood_entropies_improv), y=ood_entropies_improv,
                            legendgroup=improv_mdl_title, scalegroup=improv_mdl_title, name=improv_mdl_title,
                            side='positive', line_color='blue', fillcolor='rgba(0,0,255,0.5)', showlegend=False))
    fig.update_traces(meanline_visible=True)  #  box_visible=True
    
    fig.update_layout(
        title="",  # title
        # yaxis_title="Entropy",
        violingap=0, violinmode='overlay',
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom", y=1.02,  # Slightly above the plot
            xanchor="center", x=0.5,  # Centered horizontally
            bordercolor="gray",   # Border color
            borderwidth=1,         # Border width in pixels
        ),
        template="plotly_white",  # Base template for a clean white background
        plot_bgcolor="white",     # White plotting area
        paper_bgcolor="white",    # White outer area
        xaxis=dict(
            showline=True, linewidth=1, linecolor='black', mirror=True,
            showgrid=True,        # Make gridlines visible
            gridcolor="lightgray", # Color of gridlines
            title_font=dict(size=18),  # Increase ylabel font size
            tickfont=dict(size=18),
        ),
        yaxis=dict(
            title="Entropy",
            showline=True, linewidth=1, linecolor='black', mirror=True,
            showgrid=True,        # Make gridlines visible
            gridcolor="lightgray", # Color of gridlines
            title_font=dict(size=18),  # Increase ylabel font size
            tickfont=dict(size=18)  
        ),
        font=dict(size=14),  # General font size to enforce on all text elements
    )
    try:
        print("[INFO]: Trying to save the entropy distribution image")
        write_image(fig=fig, file=f"./dist_images/{baseline_mdl_title}_vs_{improv_mdl_title}_entropy_dist.pdf", format="pdf")
    except Exception as e:
        print(f"Failed to save the entropy dustribution image: {e}")
    return fig


def calculate_distances(points):
    """Calculate the distance between each point."""
    dist_matrix = np.linalg.norm(points[:, np.newaxis] - points, axis=2)
    return dist_matrix


def plot_per_class_metrics(per_cls_metric1, per_cls_metric2, metric1_title: str, metric2_title: str, class_labels: List[str], threshold=0.05):
    """
    Plot the Per Category Metrics as a scatter plot for a test set.

    For example:
    - Per Class Entropy vs Per Class Frequency
    - Per Class Uncertainty vs Per Class Frequency
    - Per Class Entropy vs Per Class Accuracy
    - Per Class Uncertainty vs Per Class Frequency
    """
    per_cls_metric1 = per_cls_metric1.cpu().numpy()
    per_cls_metric2 = per_cls_metric2.cpu().numpy()

    points = np.column_stack((per_cls_metric1, per_cls_metric2))
    distances = calculate_distances(points)

    fig = go.Figure(
        data=go.Scatter(
            x=per_cls_metric1, y=per_cls_metric2, mode='markers', marker=dict(size=10), text=class_labels,
            hoverinfo='text+x+y',  # Include text, x, and y values in the hover info
            hovertemplate='<b>%{text}</b><br>x: %{x}<br>y: %{y}<extra></extra>', textfont=dict(size=1)
        )
    )
    # Label positions Handling
    used_positions = []

    for i, (x, y) in enumerate(points):
        label_pos_above = (x, y + 0.02)
        label_pos_below = (x, y - 0.02)

        # Check if there are any close points
        close_points = np.where((distances[i] < threshold) & (distances[i] > 0))[0]

        if len(close_points) > 0:
            # If the point is close to others, check if it's above or below the closest point
            closest_point_idx = close_points[np.argmin(distances[i, close_points])]
            if y < points[closest_point_idx, 1]:
                fig.add_annotation(x=x, y=y, text=class_labels[i], showarrow=False, yshift=10)
                used_positions.append(label_pos_above)
            else:
                fig.add_annotation(x=x, y=y, text=class_labels[i], showarrow=False, yshift=-10)
                used_positions.append(label_pos_below)
        else:
            # If there are no close points, place the label above
            fig.add_annotation(x=x, y=y, text=class_labels[i], showarrow=False, yshift=10)
            used_positions.append(label_pos_above)

    fig.update_layout(
        title=f'{metric1_title} vs. {metric2_title}', xaxis_title=f'{metric1_title}', yaxis_title=f'{metric2_title}', plot_bgcolor='rgb(240, 240, 240)', height=600, width=1200
    )
    return fig


def plot_per_img_results(save_path):
    if not os.path.exists(save_path):
        raise ValueError(f"[ERROR]: {save_path} doesnt exist. Failed to Save the per image plots")
    # Load the CSV file
    df = pd.read_csv(save_path)
    
    # Exclude the 'test_img_names' column and create a long-form DataFrame
    metrics = [col for col in df.columns if col != 'test_img_names']

    # Create a long-form DataFrame for all metrics
    df_long = pd.melt(df, id_vars=['test_img_names'], value_vars=metrics, 
                    var_name='Metric', value_name='Value')

    # Create a horizontal violin plot with all metrics in the same plot
    fig = px.violin(df_long, y='Metric', x='Value', points='all', hover_data=['test_img_names'], orientation='h')

    # Customize the layout
    fig.update_layout(
        title="Combined Violin Plot of All Metrics per Test Image",
        xaxis_title="Metric Values",
        yaxis_title="Metric"
    )
    # try:
    #     print("[INFO]: Trying to save the Per Image Metrics Plots")
    #     write_image(fig=fig, file=f"./{save_path}/per_image_metrics.png", format="png")
    # except Exception as e:
    #     print(f"Failed to save the Per Image Metrics image in the path {save_path}: {e}")
    return fig
