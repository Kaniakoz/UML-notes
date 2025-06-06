## Uncertainty Quantification Methods

## Bayesian Neural Networks (BNNs)

### Bayesian Thinking
Bayesian statistics involves applying **Bayes' Rule** to update beliefs. It starts with prior information, updates it with observations, and obtains new information. This process can be repeated with new observations. This approach is fundamental to building Bayesian Neural Networks.
Given a **hypothesis (H)** and **evidence (E)**:
- We have prior information about the hypothesis validity.
- We observe evidence that informs us about the hypothesis $(E|H)$.
- We want to find out how this evidence influences our belief about the hypothesis $(H|E)$.
$P(H∣E)=\frac{P(E∣H)P(H)}{P(E)}$
This is **Bayes' Rule**.
- **Likelihood**: $P(E∣H)$
- **Prior**: $P(H)$
- **Posterior**: $P(H∣E)$
- **Evidence**: $P(E)$
Simplified:
> Posterior = (Likelihood * Prior) / Evidence
### Evidence
The probability of evidence is typically hard to compute conceptually but can be inferred using the law of total probability:
$P(E)=P(E∣H)P(H)+P(E∣¬H)P(¬H)$
This works well for binary hypotheses.
### Concepts
- **Prior**
    > Your previous belief about the hypothesis, or a general assumption if no information is available.
- **Likelihood**
    > The probability of observing the evidence given the hypothesis, measuring the compatibility of the evidence with the hypothesis. This is a function of the evidence.
- **Posterior**
    > Your updated belief about the hypothesis after seeing the evidence. This is a function of the hypothesis.
    

### Example: SARS-CoV-2 Test
Assume you take a SARS-CoV-2 test and it returns positive.
- **Hypothesis (H)**: You are infected with SARS-CoV-2 (binary: yes/no).
- **Evidence (E)**: Your test is positive.
- **Accuracy of the test**: 80% (P(E|H) = 0.8).
- **Prior**: Ratio of infected people in the population = 0.1% (P(H) = 0.001).
	Compute $P(E)$:
	$P(E)=P(E∣H)P(H)+P(E∣¬H)P(¬H)=0.8⋅0.001+0.2⋅0.999=0.2$
	Compute the posterior probability $P(H|E)$:
	$P(H∣E)=\frac{P(E∣H)P(H)}{P(E)}=\frac{0.8⋅0.001}{0.2006}≈0.00398$
	The posterior probability of being infected is only 0.4%, which is only four times higher than the prior probability.
If the test is negative:
	$P(¬E∣¬H)=0.8P$
	$P(¬H∣¬E)=\frac{P(¬E∣¬H)P(¬H)}{P(¬E)}=\frac{0.8⋅(1−0.001)}{1−0.2006}≈0.999$
	This is the probability of not being infected given a negative test, which is less than 99% due to the low prevalence of the disease.

### Bayesian View of Uncertainty
Bayesian statistics offers a unique perspective on uncertainty. When learning a model, we aim to learn the probability distribution $P(y|x)$, the probability of some output given the input.
However, this doesn't account for model parameters w (weights), which indirectly encode uncertainty. Thus, we want to learn $P(y|x, w)$, which directly considers the model parameters. Since model parameters are learned from data D, we aim to learn $P(w|D)$, which is what training a model does.
$P(w∣D)=\frac{P(D∣w)P(w)}{P(D)}$
Where $D = \{x_i,y_i​\}$ is a labeled training set.
Bayes' Rule allows us to decompose the probability of the weights given the data $(P(w|D))$ into the probability of the data given weights ($P(D|w)$), computed by the model.
### Interpretation
- **Prior**: $P(w)$, the prior distribution on the weights, generally assumed. This is connected to random weight initializers in ML/NN models.
- **Likelihood**: $P(D|w)$, a forward pass of the model, computing $P(y|x, w)$. This requires weights w and is defined by the model architecture/equations.
- **Posterior**: $P(w|D)$, the posterior distribution of the weights given the data. This is the learning algorithm, used for making predictions.
From a Bayesian perspective, a model is defined by both its structure (model equations) and its parameters. Model parameters encode model uncertainty about each prediction. Variations in model parameters can be encoded as a probability distribution. Bayesian statistics can estimate these distributions $P(w|D)$.
### Bayesian Neural Networks
A Bayesian Neural Network should have two components:
- **Weights/Parameters**: The weights of layers in the model are probability distributions (not necessarily in all layers).
- **Bayesian Learning**: The learning algorithm considers that it learns probability distribution weights instead of point-wise weights.
### Bayesian Learning
- **Maximum Likelihood Estimation (MLE)**: Maximize the log-likelihood given model parameters.
$w^{MLE}​=arg \underset{w}max$ $​logP(D∣w)=arg \underset{w} max​∑_{i}​logP(y_i​∣x_i​,w)$
- **Maximum A Posteriori (MAP)**: MLE formulation plus a regularization term $P(w)$, which is a prior on model parameters.
$w^{MAP}​=arg​\underset{w} max$ $logP(w∣D)=arg \underset{w}​ max[logP(D∣w)+logP(w)]$
Both formulations learn point-wise estimates of the parameters, ignoring $P(D)$ as it is constant.
### Bayesian Predictive Posterior
If the distribution over weights is obtained, output distributions can be computed with the Bayesian predictive posterior distribution:
$P(y∣x)=∫_wP(y∣w,x)P(w∣D) dw$
This equation marginalizes over all possible model parameters w. It computes predictions y with different model parameters w, weighted by the probability of those parameters given the input x. It's a form of Bayesian Model Averaging.
### Interpretation
$P(y∣x)=∫P(y∣w,x)P(w∣D)dw$
- **Forward Pass** $P(y∣w,x)$: One forward pass through the neural network, outputting a probability distribution given inputs and weights.
- **Posterior of Weights** $P(w∣D)$: Learned using a learning algorithm, the result of learning using Bayes' Rule.
This equation integrates the product of a forward pass (given a specific set of weights) and the probability of that set of weights, over all weight values.
### Monte Carlo Approximation
In general, we cannot compute the integral for the Bayesian Predictive Posterior distribution, but it can be approximated using Monte Carlo with M samples.
$P(y∣x)~M^{-1}∑_{i}^{M}P(y∣θ_i,x)P(\theta_i|x)$, where $θ_i∼P(w∣D)$
If only forward passes $P_i​$ that produce samples of the posterior are available, a rougher approximation is:
$P(y∣x)~M^{-1}​∑_{i}^M​P_i​(y∣w,x)$
In this case, we are not marginalizing over weights but over abstract stochastic passes, which is a worse approximation but computationally tractable. Larger M means better approximation.
### Intractability of BNNs
In Bayes' Rule, the term $P(E)$ is generally difficult to compute. In a Bayesian Neural Network, the equivalent term is $P(D)$, the probability distribution of the whole data, which is impossible to compute. For example, if we use images of a given size, $P(D)$ is the distribution of all images of that size, which we cannot estimate.
### Other Issues
 - encoding or representing posterior probability distributions over the weights. Typical models have millions of parameters, making the distribution very high-dimensional.
 - computational complexity. Integrating over millions of parameters and performing multiple predictions for each of these parameters is computationally infeasible.
 - there are no closed-form representations for the posterior distribution over weights, and consequently, no closed-form computations of the Bayesian predictive posterior distribution. The only alternative is to represent the distribution with samples (histograms) and use Monte Carlo methods to sample from the posterior distribution.

## Direct Uncertainty Estimation 🎯

### Maximum Likelihood Estimation (MLE)
a statistical method to find parameters for a distribution.
It starts with the likelihood function:
$L(θ;y)=∏_if_x(yi;θ)$
Where $f_x$ is the PDF of a selected distribution with parameters $\theta$. This function gives the joint probability of all data points y under the assumed distribution. We find the best parameters $θ$ by maximizing the likelihood function under $θ$: $θ^∗=arg\underset{θ}max​L(θ;y)$
The log-likelihood is easier to optimize:
$log⁡L(θ;y)=log⁡∏_if_x(y_i;θ)logL(θ;y)=∑_i=​logf_x(y_i​;θ)$
Optimal parameters are found with gradient-based optimization:
$\frac{∂L}{∂θ}=0\leftrightarrow \frac{∂log⁡L}{∂θ}=-\infty$  (approx)
### Two-Headed Models in Regression 
A simple idea is to add a second output head that outputs a confidence measure.
- **Mean Head**: $μ(x)$
- **Variance Head**: $σ^2(x)$
The question is how to train such a network. Assuming we want to output a Gaussian distribution, we can design a loss to maximize likelihood:

L(y,μ(x),σ2(x))=∏i(2πσi2)−12e−12(yi−μi)2σi2L(y,μ(x),σ2(x))=∏i​(2πσi2​)−21​e−21​σi2​(yi​−μi​)2​

log⁡L(y,μ(x),σ2(x))=∑ilog⁡((2πσi2)−12e−12(yi−μi)2σi2)logL(y,μ(x),σ2(x))=∑i​log((2πσi2​)−21​e−21​σi2​(yi​−μi​)2​)

Maximizing L is equivalent to minimizing -L.

### Gaussian Negative Log-Likelihood (NLL)

log⁡L(y,μ(x),σ2(x))=∑i(0.5(log⁡2π+log⁡σi2)+0.5(yi−μi)2σi2)logL(y,μ(x),σ2(x))=∑i​(0.5(log2π+logσi2​)+0.5σi2​(yi​−μi​)2​)

Removing constant terms:

log⁡L(y,μ(x),σ2(x))=∑i(log⁡σi2+(yi−μi)2σi2)logL(y,μ(x),σ2(x))=∑i​(logσi2​+σi2​(yi​−μi​)2​)

This loss is called the Gaussian Negative Log-Likelihood.

### Interpretation

log⁡L(y,μ(x),σ2(x))=∑i(log⁡σi2+(yi−μi)2σi2)logL(y,μ(x),σ2(x))=∑i​(logσi2​+σi2​(yi​−μi​)2​)

- The **mean** μiμi​ is supervised by the label yiyi​.
- The **variance** σi2σi2​ does not have direct supervision, but the loss influences it.

> - σi2→1σi2​→1: The logarithm term goes to 1 and the squared error term tends to 0.
> - σi2→0σi2​→0: The logarithm term goes to 1 and the squared error term tends to 1.

This loss is also called variance attenuation. When the model predicts correctly, the variance is also small. When the model predicts incorrectly (large squared error), the only way to minimize the loss is to increase the variance.

### Learning

- **Aleatoric Uncertainty**: The Gaussian NLL learns aleatoric uncertainty.
    
    > If there is noisy data, it will be hard to predict correctly, but the variance head will cover all noisy data points, estimating its aleatoric uncertainty (noise). For clean data, the variance will be low.
    

### Ensembles

Ensembling involves training M instances of the same model with different randomly drawn initial weights and then combining their predictions.

For regression, each ensemble member has two output heads:

- Mean μ(x)μ(x)
- Variance σ2(x)σ2(x)

A special loss is used for training:

log⁡p(y∣x)=(μn(xi)−yi)22σi2(xi)+12log⁡σi2(xi)+Cnlogp(y∣x)=2σi2​(xi​)(μn​(xi​)−yi​)2​+21​logσi2​(xi​)+Cn​

This is a negative log-likelihood with heteroscedastic variance, where the model predicts a variance for each data point.

### Combination

- **Classification**: Ensemble output is the average of the probabilities:

pe(y∣x)=1M∑ipi(y∣x)pe​(y∣x)=M1​∑i​pi​(y∣x)

- **Regression**: Ensemble output is a Gaussian mixture model:

pe(y∣x)∼N(μ∗(x),σ∗2(x))pe​(y∣x)∼N(μ∗(x),σ∗2(x))

μ∗(x)=1M∑iμi(x)μ∗(x)=M1​∑i​μi​(x)

σ∗2(x)=1M∑i(σi2(x)+μi2(x))−μ∗2(x)σ∗2(x)=M1​∑i​(σi2​(x)+μi2​(x))−μ∗2(x)

### DUQ - Direct Uncertainty Quantification

It is a special kind of output layer that uses a radial basis function with learnable centroids, with distance being proportional to uncertainty.

Predictions are made with:

K(f(x),c)=exp⁡(−∣∣f(x)−c∣∣22σ2)K(f(x),c)=exp(−2σ2∣∣f(x)−c∣∣2​)

Class predictions are made with:

K(f(x),c)cargmax​K(f(x),c)

The loss is binary cross-entropy. The per-class weight matrix W is learned with gradient descent, while centroids c are learned with a running mean over input features. For small numbers of samples per class, centroids c can also be learned using gradient descent. This method is only formulated for classification.

### Gradient Uncertainty

It has been proposed to use the gradient of loss with respect to input as an uncertainty measure:

Uncertainty(x)=∣∣∇xL(f(x),y)∣∣Uncertainty(x)=∣∣∇x​L(f(x),y)∣∣

At inference time, no labels y are available, so for the cross-entropy loss, y=one hot(y^)y=one hot(y^​), where y^y^​ is the class prediction made by the model. ∣∣⋅∣∣∣∣⋅∣∣ is a norm over the gradient vector to get a scalar value (e.g., L1, L2, mean, standard deviation, minimum, maximum).

## Sampling-Based Uncertainty Quantification 📊

These methods approximate posterior probability distributions by sampling or representing these distributions with histograms of samples. They are generally expensive but provide very good estimates of uncertainty. A large number of samples (100-1000) are generally required to well approximate these posterior distributions.

### Monte Carlo Dropout

It's an interpretation of applying Dropout in a neural network at inference time. Yarin Gal proved that using Dropout at inference time is approximately equivalent to sampling the predictive posterior distribution under some assumptions.

This has connections to Bayesian model averaging, as each forward pass using Dropout samples a different model, which may make different predictions. The mean and variance can be estimated from multiple samples as:

μ(x)=1N∑ifi(x)μ(x)=N1​∑i​fi​(x)

σ2(x)=1N−1∑i(fi(x)−μ(x))2σ2(x)=N−11​∑i​(fi​(x)−μ(x))2

Dropout is a well-known technique for regularization of Neural Networks. During training, a mask mi∼Bernoulli(p)mi​∼Bernoulli(p) is drawn and multiplied with the input activations, effectively making some of them zero.

### Monte Carlo DropConnect

DropConnect is a variation of Dropout, where instead of applying a mask to the activations of a layer, it is applied to the weights of a layer. It has been proven to also produce an approximation of the predictive posterior distribution. It requires the implementation of new layers that use DropConnect internally. In some cases, it outperforms MC Dropout in both task and uncertainty performance, but not always.

### What about Monte Carlo?

Enabling Dropout or DropConnect at inference transforms the neural network into a stochastic model. Each forward pass produces a different result, a sample from the predictive posterior distribution.

The model with uncertainty can be evaluated by combining the predictions from M forward passes.

p(y∣x)≈1M∑ip(y∣x,θi)p(y∣x)≈M1​∑i​p(y∣x,θi​), where θi∼θi​∼