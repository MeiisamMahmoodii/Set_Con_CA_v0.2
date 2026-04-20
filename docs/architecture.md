# Architecture Documentation

## Overview
This document outlines the architectural decisions and implementation details for the ConCAE framework.

## Input Representation
All inputs are processed as sets of feature vectors.
- **Structure**: A collection of $N$ sets, where each set contains $m$ vectors.
- **Vector Dimension**: Each vector has dimension $d$ (representing $d$-dimensional features).
- **Input Tensor Shape**: $(N, m, d)$.

## Data Loading & Preprocessing
- **Preprocessing**: No preprocessing is applied to the input features (no normalization, no scaling). The raw hidden states from the backbone are used directly.
- **Feature Extraction**: Features are extracted from a pre-trained backbone (e.g., Llama, Mistral) at a specific layer.
- **Coordinate/Identity Preservation**: If temporal or sequential information is relevant, positional encodings from the backbone are assumed to be part of the feature vectors.

## Architecture Details
### ConCAE Module
- **Core Mechanism**: ConCAE uses a set-based approach to aggregate and transform features across the $m$ elements of each set.
- **Invariance**: The architecture is designed to be permutation-invariant with respect to the $m$ elements within each set.

## Experiment Tracking
For every experiment, the following metadata must be recorded:
- **Backbone**: The model used for feature extraction (e.g., Llama-3-8B).
- **Layer Index**: The specific layer from which features were extracted.
- **Set Size ($m$)**: The number of elements per set.
- **Feature Dimension ($d$)**: The dimensionality of the input vectors.
- **Preprocessing Flag**: Confirmation that no preprocessing was applied.
- **Dataset**: The name and version of the dataset used.
