Test set: 1441 images (lesion-level split, never used for model selection).

| Model | Best epoch | Val bal. acc | Test acc | Test bal. acc [95% CI] | Test macro F1 | Train min/epoch |
|---|---|---|---|---|---|---|
| Majority class (nv) | - | - | 0.676 | 0.143 | - | - |
| ResNet-50 | 4/5 | 0.722 | 0.773 | 0.706 [0.648, 0.764] | 0.607 | 3.2 |
| ViT-Base/16 | 3/5 | 0.750 | 0.745 | 0.728 [0.670, 0.784] | 0.610 | 8.2 |

Per-class test recall (support in brackets):

| Class | ResNet-50 | ViT-Base/16 |
|---|---|---|
| akiec (34) | 0.765 | 0.765 |
| bcc (89) | 0.820 | 0.674 |
| bkl (168) | 0.500 | 0.583 |
| df (9) | 0.556 | 0.667 |
| mel (146) | 0.493 | 0.616 |
| nv (974) | 0.856 | 0.793 |
| vasc (21) | 0.952 | 1.000 |

Test balanced accuracy, ResNet-50 minus ViT: -0.022 (95% paired bootstrap CI -0.088 to +0.048). If the interval contains 0, this run does not show a reliable difference.

Single run, single seed, single split. 95% CIs are bootstrap over test images and do not include seed-to-seed training variance.
