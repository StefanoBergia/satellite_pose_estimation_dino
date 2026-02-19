# Domain Adaptation Experiments — Results

## Techniques Overview

### Baseline
Standard DINOv3 ViT-L/16 model trained on the SPEED+ lightbox+sunlamp synthetic domain (style split, ~80% of test set) without any domain adaptation. Serves as the reference for all comparisons.

### Test-Time Adaptation (TTA)
At inference time, the model's batch normalization statistics are updated using the unlabeled target images before predicting. This adapts the internal feature distribution to the test domain without any retraining, relying only on entropy minimization or norm-layer updates.

### Self-Supervised Training
The model is further trained on unlabeled target-domain images using self-supervised objectives (e.g., consistency regularization, pseudo-labels). The backbone learns target-domain representations without requiring pose annotations for the new domain.

### DSU + MixStyle (Domain Generalization)
Two augmentation techniques applied during training to improve generalization across domains:
- **DSU (Distribution Shift Uncertainty)**: perturbs feature statistics (mean/variance) to simulate unseen styles at the feature level.
- **MixStyle**: mixes instance-level feature statistics between samples from different domains, encouraging the model to be invariant to style shifts.

These are applied during standard supervised training — no target-domain data is needed at train time.

---

## Experiment 1 — Style Split (~80% of test, no overlap with eval)

> Evaluation sets: lightbox (5392 samples), sunlamp (2333 samples)

| Method                            | Split    | loss   | px_err  | px_rmse | pck    | epnp_slab | epnp_ori | epnp_pos(%) | epnp_rot(°) | epnp_t(m) | solved% | dropped |
|-----------------------------------|----------|--------|---------|---------|--------|-----------|----------|-------------|-------------|-----------|---------|---------|
| **Baseline**                      | val      | 0.0002 | 7.1736  | 11.7923 | 0.9968 | 0.0336    | 0.0253   | 0.0083      | 1.45        | 0.0475    | 0.9957  | 52      |
| **Baseline**                      | lightbox | 0.0088 | 26.1600 | 56.1054 | 0.8679 | 0.1531    | 0.1192   | 0.0339      | 6.83        | 0.2345    | 0.9688  | 168     |
| **Baseline**                      | sunlamp  | 0.0125 | 36.0756 | 71.2796 | 0.7990 | 0.1953    | 0.1509   | 0.0444      | 8.65        | 0.2975    | 0.9360  | 143     |
|                                   |          |        |         |         |        |           |          |             |             |           |         |         |
| **TTA**                           | lightbox | 0.0091 | 26.0778 | 56.8651 | 0.8697 | 0.1422    | 0.1145   | 0.0276      | 6.56        | 0.1839    | 0.9688  | 168     |
| **TTA**                           | sunlamp  | 0.0129 | 35.4930 | 72.2447 | 0.8101 | 0.1787    | 0.1430   | 0.0358      | 8.19        | 0.2365    | 0.9355  | 144     |
|                                   |          |        |         |         |        |           |          |             |             |           |         |         |
| **Self-Supervised**               | val      | 0.0002 | 6.5826  | 10.8533 | 0.9975 | 0.0305    | 0.0229   | 0.0076      | 1.31        | 0.0434    | 0.9962  | 45      |
| **Self-Supervised**               | lightbox | 0.0069 | 23.2937 | 49.0491 | 0.8844 | 0.1320    | 0.1032   | 0.0287      | 5.91        | 0.1938    | 0.9716  | 153     |
| **Self-Supervised**               | sunlamp  | 0.0105 | 33.0077 | 65.1677 | 0.8156 | 0.1671    | 0.1253   | 0.0418      | 7.18        | 0.2790    | 0.9431  | 127     |
|                                   |          |        |         |         |        |           |          |             |             |           |         |         |
| **Self-Supervised + TTA**         | lightbox | 0.0072 | 23.2486 | 49.9812 | 0.8866 | 0.1301    | 0.1044   | 0.0257      | 5.98        | 0.1684    | 0.9753  | 133     |
| **Self-Supervised + TTA**         | sunlamp  | 0.0110 | 32.6489 | 66.4964 | 0.8256 | 0.1577    | 0.1211   | 0.0365      | 6.94        | 0.2484    | 0.9445  | 124     |
|                                   |          |        |         |         |        |           |          |             |             |           |         |         |
| **DSU + MixStyle**                | val      | 0.0002 | 6.4310  | 10.3626 | 0.9979 | 0.0304    | 0.0226   | 0.0078      | 1.29        | 0.0446    | 0.9966  | 41      |
| **DSU + MixStyle**                | lightbox | 0.0069 | 21.7814 | 49.5067 | 0.9035 | 0.1246    | 0.1000   | 0.0246      | 5.73        | 0.1640    | 0.9766  | 126     |
| **DSU + MixStyle**                | sunlamp  | 0.0091 | 28.6210 | 59.2719 | 0.8554 | 0.1656    | 0.1218   | 0.0438      | 6.98        | 0.3158    | 0.9584  | 93      |
|                                   |          |        |         |         |        |           |          |             |             |           |         |         |
| **DSU + MixStyle + TTA**          | lightbox | 0.0068 | 21.5969 | 48.9913 | 0.9044 | 0.1260    | 0.1018   | 0.0241      | 5.83        | 0.1601    | 0.9779  | 119     |
| **DSU + MixStyle + TTA**          | sunlamp  | 0.0091 | 28.3574 | 59.1388 | 0.8582 | 0.1519    | 0.1192   | 0.0327      | 6.83        | 0.2102    | 0.9597  | 90      |
|                                   |          |        |         |         |        |           |          |             |             |           |         |         |
| **DSU + MixStyle + Self-Sup.**    | val      | 0.0001 | 6.1021  | 9.9722  | 0.9980 | 0.0285    | 0.0213   | 0.0072      | 1.22        | 0.0408    | 0.9972  | 34      |
| **DSU + MixStyle + Self-Sup.**    | lightbox | 0.0065 | 21.6231 | 48.0357 | 0.9015 | 0.1249    | 0.0982   | 0.0267      | 5.63        | 0.1793    | 0.9772  | 123     |
| **DSU + MixStyle + Self-Sup.**    | sunlamp  | 0.0086 | 28.5811 | 57.7028 | 0.8495 | 0.1539    | 0.1185   | 0.0354      | 6.79        | 0.2412    | 0.9566  | 97      |
|                                   |          |        |         |         |        |           |          |             |             |           |         |         |
| **DSU + MixStyle + Self-Sup. + TTA** | lightbox | 0.0064 | 21.2751 | 47.7129 | 0.9039 | 0.1174    | 0.0933   | 0.0240      | 5.35        | 0.1594    | 0.9774  | 122     |
| **DSU + MixStyle + Self-Sup. + TTA** | sunlamp  | 0.0087 | 28.0524 | 57.9191 | 0.8570 | 0.1423    | 0.1123   | 0.0300      | 6.43        | 0.1934    | 0.9597  | 90      |

---

## Experiment 1 — By Split

### Val

| Method                               | loss   | px_err | px_rmse | pck    | epnp_slab | epnp_ori | epnp_pos(%) | epnp_rot(°) | epnp_t(m) | solved% | dropped |
|--------------------------------------|--------|--------|---------|--------|-----------|----------|-------------|-------------|-----------|---------|---------|
| **Baseline**                         | 0.0002 | 7.1736 | 11.7923 | 0.9968 | 0.0336    | 0.0253   | 0.0083      | 1.45        | 0.0475    | 0.9957  | 52      |
| **Self-Supervised**                  | 0.0002 | 6.5826 | 10.8533 | 0.9975 | 0.0305    | 0.0229   | 0.0076      | 1.31        | 0.0434    | 0.9962  | 45      |
| **DSU + MixStyle**                   | 0.0002 | 6.4310 | 10.3626 | 0.9979 | 0.0304    | 0.0226   | 0.0078      | 1.29        | 0.0446    | 0.9966  | 41      |
| **DSU + MixStyle + Self-Sup.**       | 0.0001 | 6.1021 | 9.9722  | 0.9980 | 0.0285    | 0.0213   | 0.0072      | 1.22        | 0.0408    | 0.9972  | 34      |

### Lightbox

| Method                               | loss   | px_err  | px_rmse | pck    | epnp_slab | epnp_ori | epnp_pos(%) | epnp_rot(°) | epnp_t(m) | solved% | dropped |
|--------------------------------------|--------|---------|---------|--------|-----------|----------|-------------|-------------|-----------|---------|---------|
| **Baseline**                         | 0.0088 | 26.1600 | 56.1054 | 0.8679 | 0.1531    | 0.1192   | 0.0339      | 6.83        | 0.2345    | 0.9688  | 168     |
| **TTA**                              | 0.0091 | 26.0778 | 56.8651 | 0.8697 | 0.1422    | 0.1145   | 0.0276      | 6.56        | 0.1839    | 0.9688  | 168     |
| **Self-Supervised**                  | 0.0069 | 23.2937 | 49.0491 | 0.8844 | 0.1320    | 0.1032   | 0.0287      | 5.91        | 0.1938    | 0.9716  | 153     |
| **Self-Supervised + TTA**            | 0.0072 | 23.2486 | 49.9812 | 0.8866 | 0.1301    | 0.1044   | 0.0257      | 5.98        | 0.1684    | 0.9753  | 133     |
| **DSU + MixStyle**                   | 0.0069 | 21.7814 | 49.5067 | 0.9035 | 0.1246    | 0.1000   | 0.0246      | 5.73        | 0.1640    | 0.9766  | 126     |
| **DSU + MixStyle + TTA**             | 0.0068 | 21.5969 | 48.9913 | 0.9044 | 0.1260    | 0.1018   | 0.0241      | 5.83        | 0.1601    | 0.9779  | 119     |
| **DSU + MixStyle + Self-Sup.**       | 0.0065 | 21.6231 | 48.0357 | 0.9015 | 0.1249    | 0.0982   | 0.0267      | 5.63        | 0.1793    | 0.9772  | 123     |
| **DSU + MixStyle + Self-Sup. + TTA** | 0.0064 | 21.2751 | 47.7129 | 0.9039 | 0.1174    | 0.0933   | 0.0240      | 5.35        | 0.1594    | 0.9774  | 122     |

### Sunlamp

| Method                               | loss   | px_err  | px_rmse | pck    | epnp_slab | epnp_ori | epnp_pos(%) | epnp_rot(°) | epnp_t(m) | solved% | dropped |
|--------------------------------------|--------|---------|---------|--------|-----------|----------|-------------|-------------|-----------|---------|---------|
| **Baseline**                         | 0.0125 | 36.0756 | 71.2796 | 0.7990 | 0.1953    | 0.1509   | 0.0444      | 8.65        | 0.2975    | 0.9360  | 143     |
| **TTA**                              | 0.0129 | 35.4930 | 72.2447 | 0.8101 | 0.1787    | 0.1430   | 0.0358      | 8.19        | 0.2365    | 0.9355  | 144     |
| **Self-Supervised**                  | 0.0105 | 33.0077 | 65.1677 | 0.8156 | 0.1671    | 0.1253   | 0.0418      | 7.18        | 0.2790    | 0.9431  | 127     |
| **Self-Supervised + TTA**            | 0.0110 | 32.6489 | 66.4964 | 0.8256 | 0.1577    | 0.1211   | 0.0365      | 6.94        | 0.2484    | 0.9445  | 124     |
| **DSU + MixStyle**                   | 0.0091 | 28.6210 | 59.2719 | 0.8554 | 0.1656    | 0.1218   | 0.0438      | 6.98        | 0.3158    | 0.9584  | 93      |
| **DSU + MixStyle + TTA**             | 0.0091 | 28.3574 | 59.1388 | 0.8582 | 0.1519    | 0.1192   | 0.0327      | 6.83        | 0.2102    | 0.9597  | 90      |
| **DSU + MixStyle + Self-Sup.**       | 0.0086 | 28.5811 | 57.7028 | 0.8495 | 0.1539    | 0.1185   | 0.0354      | 6.79        | 0.2412    | 0.9566  | 97      |
| **DSU + MixStyle + Self-Sup. + TTA** | 0.0087 | 28.0524 | 57.9191 | 0.8570 | 0.1423    | 0.1123   | 0.0300      | 6.43        | 0.1934    | 0.9597  | 90      |

---

## Experiment 2 — Full Test Dataset (style + test splits combined)

| Method             | Split    | loss   | px_err  | px_rmse | pck    | epnp_slab | epnp_ori | epnp_pos(%) | epnp_rot(°) | epnp_t(m) | solved% | dropped |
|--------------------|----------|--------|---------|---------|--------|-----------|----------|-------------|-------------|-----------|---------|---------|
| **Baseline**       | val      | 0.0002 | 7.1736  | 11.7923 | 0.9968 | 0.0336    | 0.0253   | 0.0083      | 1.45        | 0.0475    | 0.9957  | 52      |
| **Baseline**       | lightbox | 0.0089 | 26.2228 | 56.0619 | 0.8682 | 0.1497    | 0.1165   | 0.0331      | 6.68        | 0.2278    | 0.9669  | 223     |
| **Baseline**       | sunlamp  | 0.0127 | 36.6810 | 72.5671 | 0.7940 | 0.1982    | 0.1533   | 0.0448      | 8.78        | 0.3000    | 0.9341  | 184     |
|                    |          |        |         |         |        |           |          |             |             |           |         |         |
| **DSU + MixStyle** | val      | 0.0002 | 6.4310  | 10.3626 | 0.9979 | 0.0304    | 0.0226   | 0.0078      | 1.29        | 0.0446    | 0.9966  | 41      |
| **DSU + MixStyle** | lightbox | 0.0069 | 21.7713 | 49.0754 | 0.9040 | 0.1225    | 0.0977   | 0.0248      | 5.60        | 0.1643    | 0.9766  | 158     |
| **DSU + MixStyle** | sunlamp  | 0.0092 | 29.1080 | 60.0266 | 0.8520 | 0.1664    | 0.1250   | 0.0414      | 7.16        | 0.2926    | 0.9556  | 124     |
