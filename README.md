<center>

# Segmentation Study

Ciampitti Lab, 2025

</center>

<hr>

<center>

**List of Contents**

</center>

- [Segmentation Study](#segmentation-study)
  - [Idealization Framework](#idealization-framework)
    - [Vision](#vision)
    - [Mission](#mission)
    - [Knowledge Gap](#knowledge-gap)
    - [Hypothesis](#hypothesis)
    - [Objectives](#objectives)
  - [Overview](#overview)
  - [Repository Structure](#repository-structure)
  - [Data](#data)
    - [Raw Images and Masks](#raw-images-and-masks)
    - [Metadata](#metadata)
  - [Models](#models)
  - [Experimental Design](#experimental-design)
    - [Baseline: All Data](#baseline-all-data)
    - [Collection Date Experiments](#collection-date-experiments)
    - [Genotype Experiments](#genotype-experiments)
    - [Data-Reducing Experiments](#data-reducing-experiments)
    - [Transfer Learning Across Dates](#transfer-learning-across-dates)
    - [Evaluation](#evaluation)
  - [Metrics](#metrics)
  - [Statistical Analysis](#statistical-analysis)
    - [Overall Performance (All Experiments)](#overall-performance-all-experiments)
    - [Cross-Genotype](#cross-genotype)
    - [Cross-Date](#cross-date)
    - [Data Reduction](#data-reduction)
    - [Transfer Learning](#transfer-learning)
  - [Reproducibility](#reproducibility)
    - [Environment](#environment)
    - [Typical Workflow](#typical-workflow)
  - [Extending the Study](#extending-the-study)
  - [References](#references)


<hr>

## Idealization Framework

### Vision
To develop robust, AI-driven computer vision tools that automate and standardize the phenotyping of diverse cereal crops, enabling high-throughput and precise agricultural research for global food security.

### Mission
To benchmark and evaluate state-of-the-art deep learning segmentation architectures (UNet, SegNet, DeepLabV3+, SegFormer, MaskFormer) across multiple crops (corn, sorghum, wheat), genotypes, and phenological stages, quantifying their generalization capabilities and data efficiency through rigorous experimental design.

### Knowledge Gap
While deep learning has shown promise in plant phenotyping, there is a lack of systematic comparisons regarding how modern segmentation models handle the high intra-species variability (genotypes) and temporal changes (collection dates) in field conditions across different cereal crop organs (ears, panicles, spikes). Furthermore, the trade-off between training data volume and performance in these specific agricultural contexts remains under-explored.

### Hypothesis
State-of-the-art transformer-based models (SegFormer, MaskFormer) will outperform traditional CNN-based architectures (UNet, SegNet) in generalizing across different genotypes and collection dates due to their superior global context understanding, though their performance will be more sensitive to significant reductions in training data volume compared to established baselines.

### Objectives

The main objective is to compare the outputs and performance of several state-of-the-art segmentation models (UNet, SegNet, DeepLabV3+, SegFormer, MaskFormer, and related backbones) on corn ears, sorghum panicles, and wheat spikes.

Concretely, the study addresses the following questions:

1. What is the performance of the models considering the different genotypes and collection dates on the different crops?
2. How do metrics change when models are trained on one collection date and evaluated on other dates (cross-date generalization)?
3. How do metrics change when models are trained on some genotypes and evaluated on held-out genotypes (cross-genotype generalization)?


## Overview

This repository contains a complete comparative study of modern image segmentation models on field images of three cereal crops: wheat spikes, sorghum panicles, and corn ears. The aim is to quantify and compare model performance across:

- Different crops
- Multiple genotypes within each crop
- Multiple collection dates
- Cross-date and cross-genotype generalization

The study is implemented with a combination of Python (for model training and evaluation) and R (for statistical analysis of results).


## Repository Structure

- analysis/
  - R Markdown workflows for aggregating, cleaning and analyzing model evaluation results.
- data/
  - images_metadata.csv: metadata for images (crop, genotype, collection date, etc.).
  - corn/, sorghum/, wheat/: crop-specific image/mask datasets, each split into train/test and used by the Python training notebooks.
- docs/
  - LiteratureReview.md: summary of segmentation architectures used in related crop & plant studies.
  - MetricsReview.md: summary of segmentation evaluation metrics.
- models/
  - Pretrained weights for each crop and model variant (e.g., SegFormer, UNet, SegNet, DeepLabV3+, MaskFormer) including per-date, per-genotype, reduced-data, and transfer-learning variants.
- notebooks/
  - Jupyter notebooks implementing data preparation, model training, and evaluation for different experimental settings (all data, per-date, per-genotype, reduced data, transfer learning).
- results/
  - CSVs and plots with quantitative results (e.g., model_evaluation_results.csv, summary_statistics_*.csv) generated by notebooks and R analyses.
- utils/
  - Python utility modules for models, and helpers.


## Data

### Raw Images and Masks

The data/ folder is organized by crop:

- data/corn/
- data/sorghum/
- data/wheat/

Each crop subfolder is expected to contain train/ and test/ subfolders with images/ and masks/, for example:

- data/corn/train/images
- data/corn/train/masks
- data/corn/test/images
- data/corn/test/masks

Images are RGB field images, and masks are single-channel (binary) segmentation masks where positive pixels correspond to the target organ (ear, panicle, spike) and background pixels are zero.

Image and masks data are available at [this link](https://zenodo.org/records/18474786)

### Metadata

The file data/images_metadata.csv holds metadata per image. Typical fields include:

- crop (wheat, sorghum, corn)
- genotype
- collectiondate (or similar date/stage identifier)
- individual ID
- file names or paths

This metadata was used to analyze data balance for subsequent augmentation.


## Models

The study compares a suite of segmentation architectures frequently used in plant and crop phenotyping:

- UNet
- SegNet
- DeepLabV3+
- SegFormer
- MaskFormer

Literature context, original publications, and architectural descriptions are given in docs/LiteratureReview.md.

Trained model weights are stored in models/, (not available in the repository) with filenames that encode:

- crop (wheat, sorghum, corn)
- architecture (e.g., UNET, SegNet, DeepLabV3Plus, SegFormer, MaskFormer)
- experimental condition (date-specific, genotype-specific, data-reduced, transfer-learning, etc.).

Examples:

- models/wheat_U-NET_seg.pt
- models/sorghum_SegFormer_20less_seg.pt
- models/corn_SegFormer_date3_seg.pt

These models are loaded in the evaluation notebooks to generate predictions and metrics.


## Experimental Design

The study is structured as a set of complementary experiments, each implemented as one or more notebooks in notebooks/ and corresponding R analysis scripts in analysis/.

### Baseline: All Data

- Notebook: notebooks/trainingModels_allData.ipynb
- Data: all available images for each crop.
- Goal: train each model using the full dataset for a given crop and quantify baseline performance (IoU, Precision, Recall, F1, etc.).

In this setting, the SegmentationDataset class loads images and masks from the appropriate train/ and test/ folders, normalizes images, and binarizes masks.


### Collection Date Experiments

- Notebook: notebooks/trainingModels_collectionDates.ipynb
- Goal: train and evaluate models separately per collection date, and study how performance varies with stage and environmental conditions.

Corresponding trained weights are stored as per-date model files in models/ (e.g., *date1_seg.pt, *date2_seg.pt, ...).


### Genotype Experiments

- Notebook: notebooks/trainingModels_genotypes.ipynb
- Goal: evaluate cross-genotype generalization by training on a subset of genotypes and testing on held-out genotypes.

Metadata in data/images_metadata.csv is used to define genotype-based splits, and evaluation results are later aggregated in analysis/dataAnalysis_crossGenotype.rmd.


### Data-Reducing Experiments

- Notebook: notebooks/trainingModels_dataReducing.ipynb
- Goal: quantify how performance degrades as training data is progressively reduced (e.g., 50%, 40%, 30%, 20%, 10% of the full dataset), and evaluate the effect of data augmentation.

Model weights reflecting these regimes are saved with suffixes like 10less, 20less, 30less, 40less, and 50less.


### Transfer Learning Across Dates

- Notebook: notebooks/trainingModels_transferLearning.ipynb
- Goal: investigate whether initializing from a model trained on an earlier collection date and fine-tuning on a later date improves performance versus training from scratch.

Models such as sorghum_SegFormer_transfer_seg.pt store the resulting transfer-learning weights.


### Evaluation

- Notebook: notebooks/modelsEvaluation.ipynb

This notebook:

- Loads the trained weights from models/ for each crop and architecture.
- Uses a dedicated evaluation dataset class (SegEvalDataset) to iterate over test images and masks.
- Computes segmentation metrics per image using scikit-learn and custom utilities:
  - IoU
  - Precision
  - Recall
  - F1 score
- Aggregates results into a unified CSV (results/model_evaluation_results.csv), including columns for:
  - crop
  - genotype
  - collection date
  - model name
  - metrics (IoU, Precision, Recall, F1)

These results serve as the input for the downstream R-based statistical analyses.


## Metrics

The main metrics used in this study are:

- Intersection over Union (IoU)
- Precision
- Recall
- F1 Score

Additional metrics such as Pixel Accuracy, Dice Coefficient, MAE, and Hausdorff Distance are described in docs/MetricsReview.md, and may be used for robustness checks or specific analyses.


## Statistical Analysis

Statistical and exploratory analyses are implemented in R Markdown files in analysis/.

### Overall Performance (All Experiments)

- File: analysis/dataAnalysis_all.Rmd

Workflow:

1. Load results/model_evaluation_results.csv.
2. Define the subsets of interest:
   - crops: wheat, sorghum, corn
   - models: DeepLabV3Plus, U-NET, SegNet, SegFormer, MaskFormer
   - metrics: IoU, Precision, Recall, F1
3. Remove outliers per (crop, model, metric) using an IQR-based rule.
4. Summarize performance per crop and model:
   - Mean, max, min, variance, and coefficient of variation (CV) for each metric.
5. Export summary tables to results/:
   - summary_statistics_by_crop_model_all.csv
   - summary_stats_short_by_crop_model.csv

These outputs provide a compact view of how each model performs across crops.


### Cross-Genotype

- File: analysis/dataAnalysis_crossGenotype.rmd

Focus:

- Compare model performance when genotypes are held out at evaluation time.
- Evaluate stability of IoU, Precision, Recall, and F1 across genotypes.


### Cross-Date

- File: analysis/dataAnalysis_crossDate.Rmd

Focus:

- Evaluate models trained on one collection date and tested on other dates.
- Quantify degradation or improvement across phenological stages.


### Data Reduction

- File: analysis/dataAnalysis_dataReducing.Rmd

Focus:

- Relate training set size (e.g., 100%, 50%, 40%, 30%, 20%, 10%) to performance.
- Quantify how metrics drop as data is removed and how augmentation mitigates this drop.


### Transfer Learning

- File: analysis/dataAnalysis_transfer.Rmd

Focus:

- Compare performance of models trained from scratch versus transfer learning across dates.
- Evaluate whether fine-tuning from an earlier date improves metrics on later dates.


## Reproducibility

### Environment

Python:

- PyTorch and torchvision for model implementation and training.
- numpy, pandas, scikit-image, scikit-learn for preprocessing and metrics.
- tqdm for progress bars.

R:

- tidyverse for data wrangling.
- patchwork for combining plots.

### Typical Workflow

To reproduce the study or adapt it to new data:

1. **Prepare Data**
   - Organize your images and masks under data/{crop}/train and data/{crop}/test, following the same folder structure.
   - Update or generate data/images_metadata.csv with crop, genotype, collection date, and file names.

2. **Train Models**
   - Run the appropriate training notebooks in notebooks/:
     - trainingModels_allData.ipynb
     - trainingModels_collectionDates.ipynb
     - trainingModels_genotypes.ipynb
     - trainingModels_dataReducing.ipynb
     - trainingModels_transferLearning.ipynb
   - Inspect and, if needed, adjust hyperparameters (learning rate, batch size, epochs, image size, etc.) and the CROP setting in each notebook.

3. **Evaluate Models**
   - Run notebooks/modelsEvaluation.ipynb to:
     - Load the best model checkpoints from models/.
     - Generate segmentation predictions on test data.
     - Compute metrics and save results/model_evaluation_results.csv.

4. **Analyze Results in R**
   - Knit or run the R Markdown scripts in analysis/ (e.g., dataAnalysis_all.Rmd) to:
     - Filter and remove outliers.
     - Compute summary statistics and plots.
     - Export summary tables in results/.


## Extending the Study

To extend this study to new crops, organs, or models:

- Add new image/mask datasets under data/{new_crop}/ with the same structure.
- Update data/images_metadata.csv with the new crop, genotypes, and dates.
- Implement or import the new model architecture under utils/models/.
- Create new training/evaluation notebooks reusing the existing Dataset and evaluation pipelines. If performing the same type of analysis, just re-use the notebooks, changing the paths to match the new crops.
- Add new analysis scripts in analysis/ or extend existing ones to incorporate the additional factors.


## References

- See docs/LiteratureReview.md for a curated list of segmentation architectures and plant science applications relevant to this study.
- See docs/MetricsReview.md for definitions and interpretations of segmentation metrics used throughout the analyses.
