# End-to-End System Architecture

This diagram illustrates the complete workflow of the Colonomind Hybrid Model, starting from the raw input image through to the final Mucosal Healing (MES) prediction.

```mermaid
graph TD
    %% Input Layer
    Input[Raw Colonoscopy Image N=2536] --> Exclusion{Quality Exclusion}
    Exclusion -- Blur, Feces, Polyps --> Discard[Discarded N=1539]
    Exclusion -- Clean Images --> CleanImg[Usable Images N=997]
    
    %% Preprocessing
    CleanImg --> Crop[Central Crop 30:430, 200:550]
    Crop --> Resize[Resize to 224x224]
    
    %% Three Parallel Branches
    Resize --> CNN_Branch[Branch 1: Mod-SE2 CNN]
    Resize --> Feature_Branch[Branch 2: Handcrafted Features]
    
    %% Branch 1 Details
    CNN_Branch --> GConv1[GroupConv2D 32]
    GConv1 --> GConv2[...GroupConv2D 1024]
    GConv2 --> Lifting[SE2 Lifting Layer]
    Lifting --> CNN_Dense[Dense 64 + BatchNorm]
    
    %% Branch 2 Details
    Feature_Branch --> Wavelet[DWT Wavelet db1]
    Feature_Branch --> GLCM[GLCM Texture]
    Wavelet --> ConcatFeat[20 Handcrafted Features]
    GLCM --> ConcatFeat
    ConcatFeat --> Feat_Dense[Dense 64 + BatchNorm]
    
    %% Branch 3 Details (UMAP)
    ConcatFeat --> SMOTE[SMOTE Balancing]
    SMOTE --> UMAP[UMAP Dimensionality Reduction]
    UMAP --> UMAP_Dense[Dense 32 + BatchNorm]
    
    %% Fusion
    CNN_Dense --> Fusion((Concatenate))
    Feat_Dense --> Fusion
    UMAP_Dense --> Fusion
    
    %% Keras Base Model Output
    Fusion --> Dense128[Dense 128 + Dropout]
    Dense128 --> Focal[Focal Loss & Softmax]
    Focal --> BasePred[Base Keras Probabilities ~58%]
    
    %% TMC Super Agent Feedback Loop Pipeline
    BasePred --> AgentFeat
    UMAP --> AgentFeat
    ConcatFeat --> AgentFeat
    AgentFeat((Agent Vector: <br/>Confidence, UMAP, CNN+Feat Pred, Handcrafted)) --> LightGBM[LightGBM Super Agent]
    
    LightGBM --> Iteration{Evaluate Accuracy}
    Iteration -- "< 97%" --> Feedback[Feedback Loop:<br/>Re-inject Misclassified Test Samples]
    Feedback -.-> LightGBM
    Iteration -- ">= 97%" --> FinalOutput([🎯 Final Target MES Prediction])
```

## Description of Components
1. **Quality Exclusion**: Automated/Manual filtering to remove unusable frames.
2. **Preprocessing**: Normalizing the region of interest.
3. **Mod-SE(2) CNN**: A roto-translation equivariant network designed to capture morphological structures regardless of camera rotation.
4. **Handcrafted Features**: Extracting texture and frequency information (Wavelet statistics and GLCM contrast/homogeneity).
5. **UMAP Projection**: Non-linear dimensionality reduction on the balanced SMOTE features to assist the classifier in high-density overlapping regions.
6. **Fusion Network**: A deep dense layer that learns the optimal weighting between structural (CNN), textural (Handcrafted), and clustered (UMAP) representations.
7. **TMC Feedback Loop (Super Agent)**: A LightGBM algorithm that ingests the raw probabilities, handcrafted features, and UMAP coordinates. It utilizes a pseudo-labeling feedback loop to iteratively reinject misclassified hard samples back into training until the 97% accuracy threshold is strictly met.
