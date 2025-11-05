# Self-Supervised Pseudo-Labeling Playground

[See it in action](https://jevendev.github.io/Glossary-Index-Self-supervised-Learning/)

This demo shows how you can skip human annotations and still gather useful labels. We rely on
ml5.js' MobileNet feature extractor — a network that has already taught itself rich visual
representations. When you drop images into the page, the model generates its own pseudo-labels
(top ImageNet categories plus confidences) so you can inspect or export machine-generated tags.

## Quickstart

1. Open the page and wait for the "Model ready" status.
2. Drag in your own images or click "Use Default Images" to load the bundled cat photos.
3. Press **Label Random Image** or click any thumbnail to see the machine's top predictions.
4. Use **Label All Images** to generate a full pseudo-label log that can seed downstream datasets.

No training loop is run locally; instead we demonstrate how a pre-trained network can bootstrap
labels that would otherwise require human effort. Adjust the UI or plug in a different ml5 model
to explore other self-supervised labeling strategies.
