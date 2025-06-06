## Bayesian Neural Networks (BNNs) 🧠

### Flipout vs. BBB - Regression 📊

When comparing **Bayes by Backprop (BBB)** and **Flipout** in regression tasks, **Flipout** exhibits reduced variance and converges faster with less noise compared to **BBB**.

### Architectural Considerations 🏗️

- Bayesian NNs don't need to use all layers with weight distributions (**Bayesian Layers**).
- The concept of **Bayesian layers** also applies to other methods like **MC-Dropout/DropConnect, Ensembling**, etc.
- Adjusting the number of Bayesian layers affects the trade-off between computation and uncertainty quality.
    - Fewer layers: faster model, poor uncertainty
    - More layers: slower model, high-quality uncertainty
- The number of Bayesian layers primarily influences **epistemic uncertainty**.

### Making Predictions 🎯

To make predictions with a **VI-approximate Bayesian neural network**, the **Monte Carlo approximation** is used to the **Predictive Posterior Distribution** with M samples:

P(y∣x)≈1M∑iMP(y∣wi0,x)P(wi0∣x)P(y∣x)≈M1​∑iM​P(y∣wi0​​,x)P(wi0​​∣x) where wi0∼P(w∣D)wi0​​∼P(w∣D)

This involves sampling the posterior distribution of the weights P(w∣x)P(w∣x) to produce weights ww, which are used to make a forward pass (P(y∣wi,x)P(y∣wi​,x)), and then taking a weighted average.

A common approximation is:

P(y∣x)≈1M∑iMP(y∣wi0,x)P(y∣x)≈M1​∑iM​P(y∣wi0​​,x) where wi0∼P(w∣D)wi0​​∼P(w∣D)

This simplified version doesn't weight using the weight posterior, useful when weight probabilities P(wi0∣x)P(wi0​​∣x) cannot be directly computed.

## Uncertainty Disentanglement 🧩

### Concept 💡

> Predictive uncertainty is usually a combination of **aleatoric** and **epistemic uncertainty**.

Predictive Uncertainty=Epistemic+AleatoricPredictive Uncertainty=Epistemic+Aleatoric

The goal is to obtain these uncertainties separately and train models accordingly.

### Applications Benefiting from Disentanglement 🚀

- **Out-of-Distribution Detection**: requires only **epistemic uncertainty**.
- **Measurements from Noisy Ground Truth**: model should distinguish between measurement noise (**aleatoric uncertainty**) and model uncertainty (**epistemic uncertainty**).
- **Active Learning**: needs accurate **epistemic uncertainty** estimates, ignoring **aleatoric uncertainty**, to select which samples to label next.

### Uncertainty Disentanglement - Regression 📈

- **Aleatoric Uncertainty**: can be estimated using the **Gaussian Negative Log-Likelihood** and a **Two-Head Model**.
- **Epistemic Uncertainty**: can be produced using UQ methods like **MC-Dropout/DropConnect**, **Ensembles**, **Bayes by Backprop**, or **Flipout**.

Combining both sources of uncertainty, we use a **Gaussian mixture model** (similar to an Ensemble):

p(y∣x)∼N(μ∗(x),σ∗2(x))p(y∣x)∼N(μ∗​(x),σ∗2​(x))

where

μ∗(x)=1M∑iμi(x)μ∗​(x)=M1​∑i​μi​(x)

σ∗2(x)=1M∑i(σi2(x)+μi2(x))−μ∗2(x)σ∗2​(x)=M1​∑i​(σi2​(x)+μi2​(x))−μ∗2​(x)

Here, $i$ iterates through the samples/ensembles, and $M$ is the number of samples or ensemble members.

#### Deeper Look at Variance 🔎

The variance $ \sigma^2*$ can be broken down:

σ∗2(x)=1M∑iσi2(x)+1M∑iμi2(x)−μ∗2(x)σ∗2​(x)=M1​∑i​σi2​(x)+M1​∑i​μi2​(x)−μ∗2​(x)

=Ei[σi2(x)]+Ei[μi2(x)]−Ei[μi(x)]2=Ei​[σi2​(x)]+Ei​[μi2​(x)]−Ei​[μi​(x)]2

=Ei[σi2(x)]+Vari[μi(x)]=Ei​[σi2​(x)]+Vari​[μi​(x)]

Aleatoric Uncertainty+Epistemic UncertaintyAleatoric Uncertainty+Epistemic Uncertainty

- **Aleatoric uncertainty**: mean of the per-sample/ensemble variances.
- **Epistemic uncertainty**: variance of the per-sample/ensemble means.

#### Intuition 💭

- **Aleatoric Uncertainty**: combines all variances into a single variance.
- **Epistemic Uncertainty**: variation or disagreement between models/samples; if all means are the same, the model(s) is very confident; if means differ, the model(s) disagree, indicating epistemic uncertainty.



## Evaluation of Uncertainty Quantification 📊

### Concept 💡

Specific methods are required to evaluate the quality of uncertainty produced by models. Standard evaluation formulations generally don't consider uncertainty at the output.

### Importance of Uncertainty Evaluation Methods 🌟

> Error and misclassification should be proportional to output uncertainty made by a model.

This makes output uncertainty useful for the end user.

### Warning ⚠️

Selection of **loss functions** and **metrics** is crucial. Losses should be selected according to the task, and metrics should be selected based on the desired knowledge and performance measurements.

### Learning Performance 📈

||Definition|
|---|---|
|**Losses**|Objective function guiding learning; defines the task and quality of solutions; usually differentiable.|
|**Metrics**|Measurements of quality to evaluate learning; usually non-differentiable. Losses can be used as metrics.|

### Probabilistic Classifiers 🤖

Most classifiers output a probability vector $p$ of length $C$. The class integer index $c$ can be recovered by:

c=argmaxipic=argmaxi​pi​

For binary classification, only a single probability is required:

f(x)=P(y=1)=1−P(y=0)f(x)=P(y=1)=1−P(y=0)

### Obtaining Confidences 🤝

||Description|
|---|---|
|**Classification**||
|Confidence|$= max_i p_i$|
|Entropy|$h = -\sum_i p_i log\ p_i$, where low entropy is high confidence, and high entropy is low confidence|
|**Regression**||
|Confidence|Standard deviation (square root of variance) from an output head.|

### Loss Functions - Classification 📉

|Loss|Description|
|---|---|
|**Categorical Cross-Entropy**|Used for multi-class classification with one-hot encoded labels $y_c$ and class probabilities $ \hat{y}_c$ summing to 1. L(y,y^)=−∑i∑cyiclog(y^ic)L(y,y^​)=−∑i​∑c​yic​log(y^​ic​)|
|**Binary Cross-Entropy**|Used for binary classification with labels yi∈0,1yi​∈0,1. L(y,y^)=−∑i[yilog(y^i)+(1−yi)log(1−y^i)]L(y,y^​)=−∑i​[yi​log(y^​i​)+(1−yi​)log(1−y^​i​)]|

### Loss Functions - Negative Log-Likelihood (NLL) 📉

For regression with uncertainty, NLL is commonly used:

log p(y∣x)=∑i(μ(xi)−yi)22σ2(xi)+12log(2πσ2(xi))+Clog p(y∣x)=∑i​2σ2(xi​)(μ(xi​)−yi​)2​+21​log(2πσ2(xi​))+C

The model outputs a mean $ \mu(x)$ and variance $ \sigma^2(x)$. Uncertainty (large variance) reduces the impact of squared error. This assumes the model outputs parameters of a Gaussian distribution.

### Other Loss Functions 📉

|Loss|Description|
|---|---|
|**KL Divergence**|Distance measure between probability distributions $p$ and $q$. Cross-Entropy is a simplified version.|

### Loss Functions with Uncertainty 📉

- **Cross Entropy**: Special case of NLL for classification, considers probabilities/confidences of the correct class.
- **Gaussian NLL**: Special case of NLL for regression with Gaussian distributed output. Models uncertainty through the variance output.
- **KL Divergence**: Measures distance between probability distributions, implicitly modeling uncertainty.

### Reproducibility ♻️

When training the same neural network architecture on the same dataset five times:

- Each model will be unique due to different weight values.
- Predictions will differ.
- Aleatoric Uncertainty should be similar.
- Epistemic Uncertainty will vary.

This assumes the model is properly trained and converged.

### Interaction Between Aleatoric and Epistemic Uncertainty 🤝

They interact when both are predicted by a model:

- **Aleatoric Uncertainty**: has a degree of its own Epistemic Uncertainty due to estimating it with a model.
- **Epistemic Uncertainty**: directly related to the model with no additional interaction.

### Proper Scoring Rules 🥇

> A scoring rule is a function $S(p,(y,x))$ that evaluates the quality of a predicted probability distribution $p(y|x)$ relative to an event $y \sim q(y|x)$, where $q(y|x)$ is the true distribution.

The expected scoring rule is given by:

S(p,q)=∫q(y,x)S(p,(y,x))dydxS(p,q)=∫q(y,x)S(p,(y,x))dydx

A **proper scoring rule** satisfies:

S(p,q)<=S(q,q)S(p,q)<=S(q,q)

Equality holds only if $p(y|x) = q(y|x)$ for all $p$ and $q$. Best value is obtained by predicting the true distribution $q$.

#### Examples of Proper Scoring Rules ✅

- Log-loss: $S(p, (y, x)) = log\ p(y|x)$
- Cross-entropy loss
- KL Divergence
- Brier score
- Gaussian Negative Log-Likelihood

#### Improper Scoring Rules ❌

- Standard accuracy
- Mean squared error and mean absolute error
- Custom metrics using geometric means

#### What Does This Mean? 🤔

- **Proper scoring rules**: lead to predicting the true distribution.
- **Improper scoring rules**: can mislead and don't always lead to predicting the true distribution.

### Coverage - Regression 🎯

Given confidence intervals $[l_i, u_i]$, coverage is computed as:

Cov(y,y^)=1N∑i1[y^l<=yi<=y^h]Cov(y,y^​)=N1​∑i​1[y^​l​<=yi​<=y^​h​]

### Differential Entropy - Regression ♾️

Entropy extended to continuous distributions:

h(X)=∫f(x)log f(x)dxh(X)=∫f(x)log f(x)dx

For a Gaussian distribution, entropy is $ \frac{1}{2}log(2\pi\sigma^2) + 1$, depending only on the variance $ \sigma^2$.

### Brier Score - Classification 🔢

Mean squared error applied to output probabilities vs the target probability distribution:

Brier(y,y^)=1N∑i(yi−y^i)2Brier(y,y^​)=N1​∑i​(yi​−y^​i​)2

Measures how close predicted and true probabilities are.

### Entropy - Classification 📊

Measurement of "information content" in a probability distribution. For classification:

H=−∑cP(x∣c)log P(x∣c)H=−∑c​P(x∣c)log P(x∣c)

Where $c$ are the class indices $c \in [0, C-1]$. Uniform distribution maximizes entropy.