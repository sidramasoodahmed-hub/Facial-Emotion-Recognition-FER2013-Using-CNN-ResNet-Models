# Facial-Emotion-Recognition-FER2013-Using-CNN-ResNet-Models
This project implements two deep learning models Multiple Custom CNN and a ResNet-based transfer learning model to classify facial expressions from the FER2013 dataset into seven emotion categories. It includes full training code (via notebook), saved models, and a multi-model inference script that compares predictions from both architectures.

**1. Custom CNN Model**

Built from scratch using multiple convolutional blocks

Uses Conv2D → BatchNorm → MaxPool → Dropout

Trained on 48×48 grayscale images

Lightweight and fast

Suitable for embedded systems and low-resource environments

**2. ResNet-Based Transfer Learning**

Uses a pretrained ResNet backbone

Adds a custom classifier head

Better generalization

Higher accuracy compared to CNN

Takes advantage of residual learning

This dual-approach setup highlights the strengths and limitations of both standard CNNs and deeper architectures.

**Emotion Classes**

The FER2013 dataset has 7 emotion classes:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

These labels remain consistent across both models.
