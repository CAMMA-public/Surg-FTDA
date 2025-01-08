# Surg-FTDA: Few-shot Text-driven Adaptation of Foundation Models for Surgical Workflow Analysis

Surg-FTDA (Few-shot Text-driven Adaptation) is a novel approach designed to adapt multimodal foundation models to surgical workflow analysis tasks. By leveraging a two-stage process, Surg-FTDA effectively minimizes the reliance on large-scale annotated datasets and bridges the modality gap between image and text embeddings.

### Key Features:

1. **Few-shot Data Anchor Selection**: Reduces the need for extensive paired datasets by selecting a small subset of diverse and representative image-text pairs.
2. **Modality Alignment**: Aligns image embeddings with text embeddings using an MLP to transform them into a shared semantic space.
3. **Text-driven Adaptation**: A text-only training method to train a decoder, enabling the model to generalize across generative tasks like captioning and discriminative tasks such as phase and triplet recognition.

Surg-FTDA has been validated on two foundation models, CLIP and SurgVLP, demonstrating its flexibility and robustness in surgical workflow analysis. Surg-FTDA supports generation tasks(image caption) and discrimination tasks(triplet recognition, phase recognition). 

![Workflow Overview](others/model_archi.png "Surg-FTDA Workflow")

![Workflow Overview](others/method_pipeline.png "Surg-FTDA Workflow")


## File Structure

- **`Caption/`**: Contains code for generative tasks, such as generating captions for surgical images.
- **`Discrimination/`**: Contains code for discriminative tasks, including triplet recognition and phase recognition.

## Installation

To set up the environment for Surg-FTDA, install the dependencies using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate surg-ftda
```

## Usage

### Prerequisites: Adjust File Paths

Before running the code, ensure you adjust the paths for the following files in the relevant configuration files:

1. **JSON Files**: Specify the dataset paths (e.g., image-text pairs, task-specific labels).
2. **PKL Files**: Include pre-generated embeddings for text and image encoders.
3. **PT Files**: Provide paths to the pre-trained model weights (e.g., CLIP, SurgVLP).

### Caption Tasks

### Step 1: Generate Embedding Spaces

Generate embeddings using CLIP or SurgVLP models:

- **CLIP Embeddings:**
    
    ```bash
    python Caption/embedding_generator_clip.py
    ```
    
- **SurgVLP Embeddings:**
    
    ```bash
    python Caption/embedding_generator_surgvlp.py
    ```
    

### Step 2: Train the Model

Train the model using either text embeddings or image embeddings:

- **Training with CLIP:**
    
    ```bash
    python Caption/train_clip.py
    ```
    
- **Training with SurgVLP:**
    
    ```bash
    python Caption/train_surgvlp.py
    ```
    

### Step 3: Perform Modality Alignment

Use `modality_alignment.py` to align image embeddings to the text embedding space:

```bash
python Caption/modality_alignment.py --config Caption/config_alignment.yaml
```

### Step 4: Generate Captions

Use the prediction scripts to generate captions:

- **Predict with CLIP:**
    
    ```bash
    python Caption/predict_clip.py
    ```
    
- **Predict with SurgVLP:**
    
    ```bash
    python Caption/predict_surgvlp.py
    ```
    

### Step 5: Evaluate Captions

Evaluate the quality of the generated captions:

```bash
python Caption/evaluate_caption.py
```

### Discrimination Tasks

Discrimination tasks involve recognizing triplets and surgical phases. The pipeline supports training models individually for triplet text, phase text, or a unified dataset that combines both types of text, enabling the creation of a single model capable of handling multiple tasks.

### Step 1: Generate Embedding Spaces

Generate embeddings using CLIP or SurgVLP models for triplet tasks, phase tasks, or a combination of both:

- **CLIP Embeddings:**
    
    ```bash
    python Discrimination/embedding_generator_clip.py
    ```
    
- **SurgVLP Embeddings:**
    
    ```bash
    python Discrimination/embedding_generator_surgvlp.py
    ```
    

### Step 2: Train the Model

Train the model using embeddings generated for triplet, phase, or combined datasets. The training process allows the model to learn task-specific or shared representations based on the input text type.

- **Training with CLIP:**
    
    ```bash
    python Discrimination/train_clip.py
    ```
    
- **Training with SurgVLP:**
    
    ```bash
    python Discrimination/train_surgvlp.py
    ```
    

### Step 3: Perform Modality Alignment

Use `modality_alignment.py` to align image embeddings to the text embedding space for triplet text, phase text. 

```bash
python Discrimination/modality_alignment.py 
```

### Step 4: Prediction

Use the prediction scripts to generate outputs for triplet or phase recognition tasks:

- **Predict with CLIP:**
    
    ```bash
    python Discrimination/predict_clip.py
    ```
    
- **Predict with SurgVLP:**
    
    ```bash
    python Discrimination/predict_surgvlp.p
    ```
    

### Step 5: Evaluate Discrimination Tasks

Evaluate the model’s performance on triplet or phase recognition tasks:

- **Evaluate triplet recognition:**
    
    ```bash
    python Discrimination/evaluate/evaluate_triplet.py
    ```
    
- **Evaluate phase recognition:**
    
    ```bash
    python Discrimination/evaluate/evaluate_phase.py
    ```