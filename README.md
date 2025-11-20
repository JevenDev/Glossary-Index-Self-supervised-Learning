# Self-Supervised Learning: Contrastive Learning Demo

[See it in action](https://jevendev.github.io/Glossary-Index-Self-supervised-Learning/)

This interactive demo visualizes **Contrastive Learning**, a core concept in Self-Supervised Learning (SSL). It demonstrates how machines can learn robust visual features from data without human labels by solving "pretext tasks" like the one proposed in the **SimCLR** framework (Chen et al., 2020).

## How It Works

The network learns by comparing images:
*   **Maximize Agreement (Positive Pairs)**: The model pulls together embeddings of the same image, even if it is augmented (e.g., cropped, blurry, or discolored). It learns that these are the "same" object.
*   **Minimize Agreement (Negative Pairs)**: The model pushes apart embeddings of completely different images (e.g., a dog vs. a car). It learns that these are distinct objects.

## Features

*   **Real-Time Embeddings**: Uses **TensorFlow.js** and **MobileNet** to extract 1024-dimensional feature vectors from images right in your browser.
*   **Cosine Similarity**: Calculates the actual mathematical similarity between image embeddings to drive the visualization.
*   **Interactive Visualization**: Watch the "embedding orbs" attract or repel based on the model's understanding of the images.
*   **Augmentation Simulation**: Visualizes how the model sees different versions of the same image.

## Quickstart

1.  **Load Images**: Click "Use Default Images" to load a diverse set (Cat, Dog, Car, Bird, Flower) or upload your own.
2.  **Set Anchor**: Click any thumbnail to set it as the "Anchor Image".
3.  **Compare**:
    *   **Positive Pair**: Compares the anchor to an augmented version of itself. Watch the high similarity score pull them together.
    *   **Negative Pair**: Compares the anchor to a different object. Watch the low similarity score push them apart.

## Note on Local Usage

If you run this locally by opening `index.html` directly, browser security (CORS) may block the model from reading image pixels. The app includes a fallback mode that simulates the scores so you can still see how the concept works. To see the *real* model inference, run a local web server (e.g., `python3 -m http.server`).

## AI Usage

Generative AI (ChatGPT Codex 5.0) was used for tab autocomplete and to help solve a CORS error.
