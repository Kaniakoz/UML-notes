> **Out-of-Distribution Detection**: The process of identifying inputs that are dissimilar to the data the model was trained on.

key question: How do we detect when a model is given inputs that are very different from its training set?
### Model Extrapolation
- **Interpolation**: When a model is given input similar to its training set, it interpolates well, especially with neural networks.
- **Extrapolation**: When a model encounters dissimilar input outside of its training set, it extrapolates (guesses), which often leads to failure. Detecting this at inference time is crucial.
### Closed Set Assumption
Most machine learning models operate under a significant implicit assumption:
> The training, validation, and test sets contain a pre-defined set of classes or a distribution of values, and no other distributions or classes exist.

This is known as the **closed set assumption**. The opposite is the **open set assumption**.
### Model vs. World Knowledge
- **Known Knowns**: Basic and only knowledge in the closed set assumption.
- **Known Unknowns**: The model is aware of knowledge it lacks, such as more classes than those in the training set. This is the basic open set recognition setting.
- **Unknown Knowns**: The model is unaware of things it knows. This relates more to the designer of the model.
- **Unknown Unknowns**: The most difficult setting, where knowledge is unknown, and the model is completely unaware of this lack of knowledge.

|Knowledge Type|Description|
|---|---|
|Known Knowns|Basic knowledge within the closed set assumption.|
|Known Unknowns|Model is aware of unknown knowledge, like additional classes.|
|Unknown Knowns|Model is unaware of some of its knowledge; pertains to model designers.|
|Unknown Unknowns|Most difficult: knowledge is unknown, and the model is unaware. Partially covered by open set recognition, as there may be infinite classes to enumerate.|

### Types of Anomalies
- **Anomaly Detection**: Detects anomalous samples during testing by defining what is normal. Only normal samples are available during training.
- **Novelty Detection**: Detects test input samples that belong to new classes. Usually multi-class, while anomaly detection is generally single class.
- **Open Set Recognition**: Classifies known classes accurately while detecting samples from unknown classes.    
- **Outlier Detection**: Detects outliers, samples that differ significantly from a whole set of data points.
- **Out of Distribution Detection**: Rejects input samples with labels that do not overlap with the training set (label shift) or any input dissimilar to the training set.

|Anomaly Type|Description|Training Data|Testing Data|
|---|---|---|---|
|Anomaly Detection|Detects anomalous samples by defining "normal."|Only normal samples|Normal and anomalous samples|
|Novelty Detection|Detects test inputs from new classes.|Normal samples|Samples from normal and novel classes|
|Open Set Recognition|Classifies known classes accurately, detects samples from unknown classes.|Known classes|Samples from known and unknown classes|
|Outlier Detection|Detects samples that differ significantly from the entire dataset.|All observed data points|Comparison across all observed data points|
|OOD Detection|Rejects inputs with non-overlapping labels or that are dissimilar to the training set. Requires accurate epistemic uncertainty.|Training data defines what is in-distribution|Process of rejecting any input that is dissimilar with the training set|
### Out of Distribution - ID vs OOD Data
- **In Distribution Data**: Samples and semantic meaning implied by the training set.
- **Out of Distribution Data**: Anything not inside the In Distribution set.
### Out of Distribution - Covariate/Feature Shift
> **Covariate Shift**: The input distribution changes. New semantic classes might be present, or trained classes are visually very different from the training samples.

The feature distribution changes, but the label distribution is the same (the model has not been retrained).
### Out of Distribution - Label Shift
> **Label Shift**: The feature distribution is the same, but the label distribution changes.

This can be a change in the domain that generates data and labels.
### Out of Distribution - Concept Shift
> **Concept Shift**: The meaning of labels might change over time or regions.
### In Summary - OOD Shifts
OOD detection is crucial because models can fail in unexpected ways when presented with data outside their training distribution. Examples include:
- Models overfitting to the color of the sky when trained to detect tanks.
- Medical image classification models identifying X-ray machine features or hospital annotations instead of actual medical conditions.
## Evaluation of OOD Performance
### Concept - Datasets
For evaluation, we need at least two datasets:
- **In Distribution Dataset (ID)**: Where your model is trained and tested.    
- **Out of Distribution Datasets (OOD)**: Where the model is evaluated for OOD detection performance. This dataset must:
    - Be out of distribution.
    - Cover different semantic classes.
    - Have some kind of corruption.
    - Lack intersection with the ID dataset.
    - Be the same format (color images, greyscale images, same input size).
### Concept - Datasets Examples
- **Fashion MNIST vs MNIST**: Both are 28x28 grayscale images, but semantic classes are very different (fashion items vs. digits).
- **CIFAR10 vs SVHN**: Both are 32x32 color images (RGB), with different semantic classes (animals/vehicles vs. digits in house numbers).
- **Split Across Classes**: Train only on a subset of classes and use the remaining classes as OOD data.
### Evaluation Labels
For evaluation, we usually need labels. Since OOD detection is formulated like a binary classification problem, we need binary labels:
- **Class 0**: In-Distribution data.
- **Class 1**: Out of Distribution data.
These labels are separate from the ones in the main task (classification or regression).
### Evaluation Protocol
1. Decide on your ID and OOD datasets.
2. Train your model on the train split of the ID dataset.
3. Make confidence/probability/uncertainty predictions on the ID and OOD dataset test splits. We will call these y^idy^​id​ and y^oody^​ood​.
4. Form virtual labels, consisting of 0’s for the ID dataset, and 1’s for the OOD dataset    
5. Use any evaluation metrics to compute an evaluation score.
### Confidences for OOD Detection
- **Maximum Probability**: The inverse of the maximum softmax probability, (1−max)⁡, is used for OOD detection.
- **Entropy**: Shannon entropy for classification or differential entropy for regression.
- **Output Standard Deviation**: For regression, the standard deviation output is a confidence that can be used for OOD.
### Evaluation Metrics
For OOD detection evaluation, the goal is for confidence scores to be low for ID samples and high for OOD samples.
### Binary Classification Evaluation
In binary classification, there are four conditions of correct/incorrect classification that define four different performance quantities.
- **True Positives (TP) and True Negatives (TN)**: These are correct classifications.
$TPR=\frac{\text{num of True Positives}}{\text{num of Positives}}=1−FNR$
$TNR=\frac{\text{num of True Negatives}}{\text{num of Negatives}}=1−FPR$
- **False Positives (FP) and False Negatives (FN)**: These are incorrect classifications.
$FPR=\frac{\text{num of False Positives}}{\text{num of Negatives}}=1−TNR$
$TNR=\frac{\text{num of False Negatives}}{\text{num of Positives}}=1−TPR$
One way to summarize these quantities is the F1 score: $F1=\frac{2TP}{2TP+FP+FN}​$
### ROC Curves
> **ROC Curve**: A Receiver Operating Characteristics Curve is a plot of the True Positive Rate (TPR) versus the False Positive Rate (FPR) of a binary classifier as the discrimination threshold is varied.

The TPR and FPR change as the discrimination threshold is varied, characterizing the discrimination power of the scores that the classifier produces.
### ROC Curve - Procedure
1. Select a number of steps N and discretize the range of input confidence scores into S.
2. For each confidence threshold s∈S:
    1. Apply the threshold for scores: $\hat{y}≤s$ to produce 0-1 predictions.
    2. Compute TPR and FPR from the 0-1 predictions using the ground truth labels.
    3. Add the values of TPR and FPR to the plot (these are the X and Y axis values).
3. Display the plot; your ROC curve is ready.
### ROC Curve - Interpretation
The best binary classifier would produce TPR=1.0 and FPR=0.0. Classifiers with ROC curves closer to (1,0) are better.
### AUROC - Area Under the ROC Curve
> **AUROC (AUC)**: Area under the ROC Curve.

An AUC of 0.5 indicates that the classifier is just guessing. Any AUC > 0.5 is a classifier working better than chance. The ideal value of the AUC is 1.0.
One interpretation of the AUC is that it is the probability that a positive class example has a higher score than a negative class example.

## Out of Distribution Detection Methods
### Epistemic Uncertainty
The premier method for out-of-distribution detection is **epistemic uncertainty**.
- Epistemic uncertainty should be low inside samples of the training set.
- Epistemic uncertainty should be high outside of the training set.
Entropy $-\Sigma_{c}P(y=c|x)log$ or the maximum probability $(1−max_c​P(y=c∣x))$, can be used for classification, while the standard deviation output is used for regression.
### Training on OOD Data
One approach for OOD Detection is to use an additional output head (contrastive loss, maximum discrepancy, etc.) and tune the extra output head to produce a high score on OOD data and a low score on ID data.
- **Additional Head**: Train using binary cross-entropy so the additional head learns to discriminate ID and OOD data.
- **Maximum Discrepancy**: Learn a model that maximizes discrepancy between features or softmax scores between ID and OOD data.
### ODIN - OOD Detection in Image Networks 
ODIN makes use of temperature scaling, with the observation that a high temperature (T>100) improves OOD detection performance when using the maximum probability. In the paper, they use T=1000.
This is because for high temperatures, the softmax output is flatter, closer to the input logit space, where it is easier to distinguish ID from OOD samples.
Variations include Generalized ODIN, which trains an additional head to output a per-sample temperature T. In ODIN, T is learned from OOD data.
### Energy-based OOD Detection 
score based on the softmax outputs:
$E(x,f)=−Tlog⁡∑_{i}^{C}exp⁡(f_i(x)/T)$
This represents the free energy of the prediction, based on the softmax denominator. OOD inputs usually get a lower energy score than ID inputs.
### Generative Models for OOD Detection
A generative model models the density $P(x)$ instead of discriminative models that model density $P(y∣x)$. Examples of generative models are GANs and Variational Autoencoders.
In theory, a well-trained generative model would assign lower probabilities (or zero) for OOD data than ID data, but this does not always happen in practice.
One way to improve this for OOD detection is to use a likelihood ratio , which includes information from another generative model that models background information ($P_{θ_o}(x)$) for the same modality:
$LLR(x)=log⁡\frac{P_θ(x)}{P_{θ_o}(x)}=log⁡P_θ(x)−log⁡P_{θ_o}(x)$
### OOD Detection in Regression
Unfortunately, there are not many OOD detection methods particularly designed for regression tasks.
But still, epistemic uncertainty works very well. For this purpose, we use the variance/standard deviation head. The higher its output, the more likely the input is out of distribution.
One important detail is that variance is unbounded, unlike entropy or probability for classification.
### Out of Distribution Detection - Pitfalls
- It is not easy to completely separate ID and OOD examples, as some ID examples still have high uncertainty, and sometimes OOD examples have low uncertainty due to variability in classes.
- Choosing a threshold is not easy, as lots of analysis has to be performed.
- There are no guarantees on OOD performance, and there are known cases of bad effects.
- Uncertainty should be used as additional information from where further human analysis can be decided, instead of enabling fully automatic processing.