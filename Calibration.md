> Calibration compares task performance as the confidence of predictions changes.
   For example, predictions with 10% confidence should be correct 10% of the time.

**Overconfidence**: Confidence is higher than accuracy, giving a false sense of security.
**Underconfidence**: Confidence is lower than accuracy, so confidence should be higher.
### Calibration - Reliability Plots 
- Predictions are divided by confidence values $conf(B_i)$ into bins $B_i$.
- For each bin, the accuracy $acc(B_i)$ is computed.
- Values $(conf(B_i), acc(B_i))$ are plotted.
- $conf(B_i) < acc(B_i)$: model is underconfident.
- $conf(B_i) > acc(B_i)$: model is overconfident.
- $conf(B_i) = acc(B_i)$: perfect calibration.
Calibration can be observed using a Reliability plot:
1. Take the predictions of a model over a dataset.
2. Divide the predictions into bins $B_i​$ by confidence values conf(B$_i$).
3. Compute the accuracy $acc(B_i)$ for each bin.
4. Plot the values $(conf(B_i),acc(B_i))$.
Regions where $conf(Bi)<acc(Bi)$ indicate **underconfidence**, while regions $conf(Bi)>acc(Bi)$ indicate **overconfidence**. The line $conf(Bi)=acc(Bi)$ indicates perfect calibration.

### Calibration - Metrics

| Metric                         | Description                                                      |
| ------------------------------ | ---------------------------------------------------------------- |
| **Calibration Error**          | $CE = \sum_{i} \mid acc(B_i)-conf(B_i)$\|                        |
| **Expected Calibration Error** | $ECE = \sum_i N^{-1} \mid B_{i}\mid \mid acc(B_i)-conf(B_i)\mid$ |
| **Maximum Calibration Error**  | $MCE = \underset{i}max \mid acc(B_i)-conf(B_i)\mid$              |
- **Calibration Error (CE)** -> Affected by variations in the number of samples in each bin.
- **Expected Calibration Error (ECE)** 
    ECE=1N∑i∣Bi∣∣acc(Bi)−conf(Bi)∣ECE=N1​∑i​∣Bi​∣∣acc(Bi​)−conf(Bi​)∣
    Weights the error by the proportion of samples in each bin, making it more stable and less prone to outliers.
- **Maximum Calibration Error (MCE)**:
    MCE=maxi∣acc(Bi)−conf(Bi)∣MCE=maxi​∣acc(Bi​)−conf(Bi​)∣
    Computes the maximum level of miscalibration for risk-averse applications.
### Calibration in Regression
We have confidence intervals $[l, u]$ with a confidence level $\alpha$ associated with them:
$P(l<=y<=u)=α$
For a predictive interval $[l, u]$, the probability that it contains the true value is $\alpha$.
We compute $u$ and $l$ based on an assumption on the distribution of $y$, like Gaussian or uniform.
### Gaussian Confidence Intervals 
Assuming $F$ is the CDF of a Gaussian distribution, the computation of an $\alpha$ confidence interval has a pseudo-closed form:
$l=μ−\mid z_{\frac{\beta}{2}}\mid \sigma$
$u=μ+\mid z_{\frac{\beta}{2}}\mid \sigma$
Where $\beta = 1 + \alpha$ and $z_{\frac{\beta}{2}}$ is the z score$^2$ corresponding to the quantile $\frac{\beta}{2}$.

### Calibration in Regression (Steps)
1. Define a set of confidences $S = [0.0, 0.1, 0.2, ..., 0.9, 1.0]$.
2. For each confidence $\alpha \in S_{\alpha}$:
    1. Define an $\alpha$ confidence interval for the per-sample distribution with parameters $\mu_1$ and $\sigma^2_i$, with $i$ being the sample index.
    2. Compute the coverage using the previously computed per-sample confidence intervals.
    3. Add $(\alpha, acc)$ to the plot.
3. Display the plot.

## Other Evaluation Plots 
Uncertainty should be proportional to error or misclassification.
- Predictions with low confidence: large error or misclassified.
- Predictions with high confidence: small error or correctly classified.
#### Steps
1. Take predictions with uncertainty over a dataset and divide the range into steps $U$.
2. For each step $u \in U$:
    1. Threshold all predictions with $u$.
    2. Compute some error metric $e$ for the remaining predictions.
    3. Add $(u, e)$ to the plot.
3. Display the plot.
#### Error Metrics
- **Classification**: Loss, accuracy, error (1 - accuracy), brier score.
- **Regression**: Mean squared error, mean absolute error, $R^2$ score.
#### Interpretation
- Increasing confidence should lead to a lower error/higher accuracy.
- Decreasing confidence should lead to a higher error/lower accuracy.
### Alternative Error vs. Confidence Scatterplot
Plot each sample in a dataset with a measure of error on the Y-axis and confidence/uncertainty on the X-axis.
## Confidence in Classification and Regression Models
### Classification Confidence
In classification, **confidence** measures help understand the certainty of a model's predictions. Two common measures are:
1. **Maximum Probability**: The highest probability assigned to any class. Expressed as:
    confidence=max(pi)confidence=max(pi​)
2. **Entropy (h)**: A measure of uncertainty in the probability distribution. Lower entropy indicates higher confidence, and vice versa. The formula is:
    h=−∑ipilog(pi)h=−∑i​pi​log(pi​)
    Note that entropy goes the _other way_; low entropy is high confidence, high entropy is low confidence.
### Regression Confidence
For regression models, **standard deviation** (the square root of variance) can serve as a confidence measure, typically provided as an output head.
## Loss Functions in Classification
### Categorical Cross-Entropy
Used for **multi-class classification** problems. Labels ycyc​ are one-hot encoded, and model predictions y^cy^​c​ are class probabilities that sum to 1. The loss function is:
L(y,y^)=−∑i∑cyiclog(y^ic)L(y,y^​)=−∑i​∑c​yic​log(y^​ic​)
### Binary Cross-Entropy
Used for **binary classification** problems with labels yi∈{0,1}yi​∈{0,1}. The loss function is:
L(y,y^)=−∑i[yilog(y^i)+(1−yi)log(1−y^i)]L(y,y^​)=−∑i​[yi​log(y^​i​)+(1−yi​)log(1−y^​i​)]
## Negative Log-Likelihood (NLL)
Log-likelihoods are a family of loss functions. For regression with uncertainty, NLL is commonly used. The equation is:
logp(y∣x)=∑n[−log(2πσ2(xi))2−(μ(xi)−yi)22σ2(xi)+C]logp(y∣x)=∑n​[−2log(2πσ2(xi​))​−2σ2(xi​)(μ(xi​)−yi​)2​+C]
Here, the model outputs both a **mean** μ(x)μ(x) and **variance** σ2(x)σ2(x), which are weighted. High variance reduces the impact of squared error, while the logarithm of variance provides a counteracting effect. This loss assumes the model outputs parameters of a Gaussian distribution.
## Kullback-Leibler (KL) Divergence
KL Divergence measures the distance between probability distributions pp and qq. Cross-entropy is a simplified version of this loss with certain assumptions:
L(p,q)=∫p(x)log(p(x)q(x))dx≈∑iyilog(yiy^i)L(p,q)=∫p(x)log(q(x)p(x)​)dx≈∑i​yi​log(y^​i​yi​​)
## Loss Functions with Uncertainty
Certain loss functions inherently consider uncertainty:
- **Cross-Entropy**: A special case of NLL for classification, considering class probabilities/confidences.
- **Gaussian NLL**: A special case of NLL for regression with Gaussian distributed output, modeling uncertainty via variance.
- **KL Divergence**: Measures distance between probability distributions, implicitly modeling uncertainty.
## Reproducibility in Model Training
When training the same neural network architecture on the same dataset multiple times:
- **Models**: Each model is a unique instance due to different weight values.
- **Predictions**: Predictions will differ due to different weights.
- **Aleatoric Uncertainty**: Should remain relatively unchanged.
- **Epistemic Uncertainty**: Will vary to different degrees.
A model is considered a specific instance where:
1. The dataset/task it was trained on.
2. The weight values.
3. The model architecture (layers and equations).
This assumes the model is properly trained, with loss decreasing and converging to similar values.
## Interaction Between Aleatoric and Epistemic Uncertainty
- There is some interaction when both sources are predicted by a model.
- **Aleatoric Uncertainty**: Has its own Epistemic Uncertainty because we estimate it with a model (e.g., variance head trained with Gaussian NLL).
- **Epistemic Uncertainty**: Directly related to the model, so no additional interaction.
## Proper Scoring Rules
### Definition
**Scoring functions** evaluate the quality of predicted probability distribution outputs. A scoring rule S(p,(y,x))S(p,(y,x)) evaluates the quality of a predicted probability distribution p(y∣x)p(y∣x) relative to a true distribution q(y∣x)q(y∣x). The expected scoring rule is:
S(p,q)=∫q(y,x)S(p,(y,x))dydxS(p,q)=∫q(y,x)S(p,(y,x))dydx
A **proper scoring rule** is one where:
S(p,q)≤S(q,q)S(p,q)≤S(q,q)
Equality holds only if p(y∣x)=q(y∣x)p(y∣x)=q(y∣x) for all pp and qq. This means the best value is obtained only by predicting the true distribution qq. A scoring rule can be made into a loss function by taking L=−S(p,q)L=−S(p,q).
### Examples of Proper Scoring Rules
- General log-loss: S(p,(y,x))=logp(y∣x)S(p,(y,x))=logp(y∣x)
- Cross-entropy loss
- KL Divergence
- Brier score
- Gaussian Negative Log-Likelihood
### Improper Scoring Rules
- Standard accuracy:
    Acc(y,y^)=1N∑i1[yi=y^i]Acc(y,y^​)=N1​∑i​1[yi​=y^​i​]
    It does not consider prediction confidence, and 100% accuracy can be achieved without predicting the right distribution.
- Mean squared error and mean absolute error: They do not consider distribution outputs.
- Custom metrics (e.g., geometric means): M(y,y^)=yiy^iM(y,y^​)=yi​y^​i​
- ​, which can be minimized by predicting zeros.
### Implications
- **Proper scoring rules** lead to predicting the true distribution.
- **Improper scoring rules** can mislead and do not always lead to predicting the true distribution.
- Distribution outputs are needed for proper uncertainty quantification, so focusing on proper scoring rules is essential.
## Coverage in Regression
This is a continuous version of an accuracy metric for regression with uncertainty. Given confidence intervals [li,ui][li​,ui​], coverage is computed as:
Cov(y,y^)=1N∑i1[y^il≤yi≤y^ih]Cov(y,y^​)=N1​∑i​1[y^​il​≤yi​≤y^​ih​]
Where y^ily^​il​ is the i-th predicted lower bound, and y^ihy^​ih​ is the corresponding upper bound.
## Differential Entropy in Regression
Entropy can be extended to continuous distributions for a PDF f(x)f(x):
h(X)=−∫f(x)logf(x)dxh(X)=−∫f(x)logf(x)dx
For a Gaussian distribution, entropy is 12log(2πσ2)+1221​log(2πσ2)+21​. It depends only on the variance σ2σ2 of the distribution.
## Brier Score in Classification
This is the mean squared error applied to output probabilities versus the target probability distribution (one-hot encoding or per-class probabilities):
Brier(y,y^)=1N∑i(yi−y^i)2Brier(y,y^​)=N1​∑i​(yi​−y^​i​)2
The Brier score measures how close the predicted and true probabilities are and is a proper scoring rule.
## Entropy in Classification
Entropy measures the "information content" in a probability distribution. For classification, using a discrete probability distribution conditioned by each class P(x∣c)P(x∣c), entropy is:
H=−∑cP(x∣c)logP(x∣c)H=−∑c​P(x∣c)logP(x∣c)
Where cc is each of the class indices c∈[0,C−1]c∈[0,C−1]. Entropy is directly related to uncertainty. The uniform distribution maximizes entropy. For a fixed mean and variance, the Gaussian distribution maximizes entropy.
## Calibration
Calibration indicates how much we can trust the confidence of a model by comparing task performance (e.g., accuracy) as confidence changes.
For example:
- A prediction made with 10% confidence should be correct 10% of the time.
- A prediction made with 90% confidence should be incorrect only 10% of the time.
### Overconfidence and Underconfidence
- **Overconfidence**: Confidence is higher than accuracy, giving a false sense of security.
- **Underconfidence**: Confidence is lower than accuracy.
A model can be both under and over confident in different regions of the confidence space.
### Calibration in Regression
In regression, confidence intervals [l,u][l,u] have a confidence level αα associated with them, such that:
P(l≤y≤u)=αP(l≤y≤u)=α
For a predictive interval [l,u][l,u], the probability that it contains the true value is αα. To compute uu and ll given a fixed αα, assumptions must be made (e.g., yy is Gaussian or uniformly distributed).
P(a≤X≤b)=F(b)−F(a)P(a≤X≤b)=F(b)−F(a)
Where FF is the cumulative distribution function (CDF).
### Gaussian Confidence Intervals
Assuming FF is the CDF of a Gaussian distribution, a αα confidence interval is computed as:
l=μ−zα2σl=μ−z2α​​σ
u=μ+zα2σu=μ+z2α​​σ
Where z=F−1(1+α2)z=F−1(21+α​) is the z-score corresponding to the 1+α221+α​ quantile.
### Calibration in Regression Steps
1. Define a set of confidences S=[0.0,0.1,0.2,...,0.9,1.0]S=[0.0,0.1,0.2,...,0.9,1.0].
2. For each confidence α∈Sα∈S:
    1. Define an αα confidence interval for the per-sample distribution with parameters μμ and σ2σ2.
    2. Compute the coverage (confidence interval accuracy) using the computed per-sample confidence intervals.
    3. Add (α,acc)(α,acc) to the plot.
3. Display the plot.
## Error vs. Confidence Plots
Uncertainty should be proportional to error or misclassification. Evaluate this by plotting the relationship between confidence and error.
Predictions with low confidence should have large error or be misclassified, while predictions with high confidence should have small error or be correctly classified.
### Steps to Create the Plots
1. Take predictions with uncertainty over a dataset, take the min and max uncertainty/confidence, divide the range into steps UU.
2. For each step u∈Uu∈U:
    1. Threshold all predictions with uu, discarding predictions that do not meet the threshold.
    2. Compute an error metric ee for the remaining predictions using μμ and ground truth labels yy.
    3. Add (u,e)(u,e) to the plot.
3. Display the plot.
### Error Metrics
- **Classification**: Loss, accuracy, error (1 - accuracy), Brier score, etc.
- **Regression**: Mean squared error, mean absolute error, R2 score.
### Interpretation
Increasing confidence should lead to lower error/higher accuracy, and decreasing confidence should lead to higher error/lower accuracy. The plot can compare different models or uncertainty quantification methods.
### Alternative Error vs. Confidence Plot
Plot each sample in a dataset, with error on the Y-axis and confidence/uncertainty on the X-axis. There should be a monotonic (increasing) relationship between error and confidence.
## Calibration of Probabilities
### What does calibrated mean?
According to Merriam-Webster dictionary, calibrate means:
1. To ascertain the caliber of (something).
2. To determine, rectify, or mark the graduations of (something, such as a thermometer tube).
3. To standardize (something, such as a measuring instrument) by determining the deviation from a standard so as to ascertain the proper correction factors.
4. To adjust precisely for a particular function.
5. To measure precisely.
### Calibration of Forecasters
Calibration can only be measured on a dataset, not directly in each prediction. It is fundamentally a frequentist measure.
### Probability Calibration
Confidence = Accuracy (1) #Probability predicted by your Model=Observed frequency of correct predictions
### Probability Calibration - Classification
P(y|ŷ∝) = 1/|ŷ∝| ∑1[ŷi =yi] (2) ŷ∝
### Probability Calibration - Regression
P(l∝ <= y <= u∝)=α (3) Where [l∝,u∝] are the bounds of the confidence interval, which is a function of the confidence level α, and predicted mean μ̂i and variances σ̂i 2. ŷ∝,i ∑1[ŷl∝,i <= y <= ŷh∝,i] = α (4)
### Why Calibration Matters?
For probabilities, you might be producing numbers in [0,1], but these are not probabilities, only when you can give them a probabilistic interpretation, and that they represent the probability of a prediction being correct, you can use these as probabilities and confidences.
### Predicting Bad Calibrated Probabilities
Basically, there are ways to cheat calibration, so calibration by itself is not the only goal during model training.
### Why models are Miscalibrated? - Classification
- Loss
- Multi-Class
- Not Probabilities
### Why models are Miscalibrated? - Regression
- Loss
- Confidence Levels The overall message is that models are not trained to be well calibrated as an explicit learning goal.
But there are methods to correct confidences after training(post-hoc) so they are better calibrated.
### Calibration and Sources of Uncertainty
In general epistemic uncertainty is better expected to be calibrated, since it relates to the model and how it makes a prediction, so it should gauge the confidence level of the prediction being correct.
Aleatoric uncertainty has an e↵ect of modeling the label noise, which might produce incorrect predictions (due to noise), or increase the size of confidence intervals.
### Over and Under Confidence
Regions where overall confidence is higher than accuracy indicate an overconfident model, so confidences should be lower than they are. This is the worst case as a high confidence gives a false sense of security.
Regions where overall confidence is lower than accuracy indicate that the model is underconfident, which means that the confidence should actually be higher than it is, in order to match accuracy.
Note that a model can be both under and over confident at the same time, but in di↵erent regions of the confidence space.
### Types of Classification Calibration
- Confidence Calibration maxpi
- Classwise Calibration
- Multi-class Calibration Confidence < Classwise < Multiclass
### Regression Calibration
P(Y <= g(x,τ))=τ (6)
### Setting Number of Bins
One option is tune this parameter on the training set, select the value of N that minimizes the ECE.
Another option is to visually look at the calibration plot (including bins), and vary N, balancing smoothing and noise.
The e↵ect of changing N is that a small N will oversmooth the plot, while a large N might have lots of bins with no or zerosamples.
### Decomposition of Proper Scoring Rules
E[d(S,Y)] = E[d(S,C)] + E[d(C,Y)]
### Concept
For this there are two major types of methods: Post-Hoc Modify or recompute predicted probabilities to improve their calibration. A priori Improve model structure or training to produce a better calibrated model.
### Calibration Maps
A calibration map is a function g : S -> P, where the mapping is applied to the confidences produced by a model, and the mapping g is learned in a way that improves calibration of the predicted confidences.
### Platt Scaling
g(x) = 1/1+exp(Ax+B) (7) y+ = N++1/N++2; y- = 1/N-+2 (8)
### Platt Scaling - Training
To use test data, the calibration map should be trained on thetraining set, and evaluated on the validation set, including anyhyper-parameter tuning. Then finally, a single evaluation shouldbe made on test data. Missing any of these steps will causeleakage.
### Empirical Binning
g(x) = acc(Bi) if x ∈ Bi (9)
### Isotonic Calibration
In Isotonic calibration, we use the same setup as empiricalbinning, but now the bin edges B = [li,ui] with li < ui, arelearned from the data instead of being fixed values that areusually set from a decided number of bins N. The bin sizes donot have to be equal.
### Temperature Scaling
Softmax(x) = exp(xi/T) ; i ∈ [0,C-1] / Σexp(xj/T)
exp(xi/Ti)/ Σ exp(xj/Tj)
