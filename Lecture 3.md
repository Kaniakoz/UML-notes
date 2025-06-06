## Uncertainty Quantification Methods - Part II 🚀

### Today's Agenda

1. Variational Inference for BNNs
    
2. Uncertainty Disentanglement
    
3. Comparison of UQ Methods
    

---

## Variational Inference for BNNs 💡

### Concept

To learn a Bayesian NN, we need to learn the posterior probability of the weights:  
$P(w \mid D)$.  
For this, we cannot use the standard Bayesian framework due to its intractability. Instead, we learn approximate distributions for $P(w \mid D)$.

---

### Variational Inference (VI)

In VI, we approximate $P(w \mid D)$ with another distribution $q_\theta(w \mid D)$, such that some distance is minimized (to obtain the best approximation).  
As a distance metric, the **Kullback-Leibler Divergence** is used. This process happens during training, so the KL divergence is added to the standard training loss.

---

### Kullback-Leibler Divergence

The KL divergence between two distributions is given by:

KL(q,p)=∫q(x)log⁡q(x)p(x) dx\text{KL}(q, p) = \int q(x) \log \frac{q(x)}{p(x)} \, dx

Some important properties:

- $\text{KL}(q, p) \geq 0$ for all $q$ and $p$
    
- $\text{KL}(q, p) = 0$ if and only if $q = p$
    

---

### Variational Inference

We aim to minimize:

θ∗=arg⁡min⁡θ KL(qθ(w∣D),P(w∣D))\theta^* = \arg\min_\theta \, \text{KL}(q_\theta(w \mid D), P(w \mid D))

We use the following approximation:

θ∗=arg⁡min⁡θ∫qθ(w∣D)log⁡qθ(w∣D)P(w) dw−Eqθ(w∣D)[log⁡P(D∣w)]\theta^* = \arg\min_\theta \int q_\theta(w \mid D) \log \frac{q_\theta(w \mid D)}{P(w)} \, dw - \mathbb{E}_{q_\theta(w \mid D)}[\log P(D \mid w)]

We replaced $P(w \mid D)$ with $\frac{P(w)P(D \mid w)}{P(D)}$ and omitted $P(D)$, so this produces unnormalized probabilities.

---

### The Trick

Let $\tilde{p}(x)$ be our unnormalized probability, and $Z(\theta)$ the normalizing constant (evidence). Then:

J(q)=∫q(x)log⁡p~(x)Z dx=∫q(x)log⁡q(x)p(x) dx−log⁡Z(θ)J(q) = \int q(x) \log \frac{\tilde{p}(x)}{Z} \, dx = \int q(x) \log \frac{q(x)}{p(x)} \, dx - \log Z(\theta)

Rewriting:

log⁡Z(θ)=KL(q,p)−J(q)≥J(q)\log Z(\theta) = \text{KL}(q, p) - J(q) \geq J(q)

So $J(q)$ is a lower bound on the log-evidence $\log Z(\theta)$.

---

### Evidence Lower Bound (ELBO)

Visual representation of relationships for the ELBO.

---

### Variational Inference Loss

L(θ)=KL(qθ(w∣D),P(w))−Eqθ(w∣D)[log⁡P(D∣w)]\mathcal{L}(\theta) = \text{KL}(q_\theta(w \mid D), P(w)) - \mathbb{E}_{q_\theta(w \mid D)}[\log P(D \mid w)]

- $\text{KL}(q_\theta(w \mid D), P(w))$: KL divergence between approximate posterior and prior
    
- $\mathbb{E}_{q_\theta(w \mid D)}[\log P(D \mid w)]$: expected negative log-likelihood (model loss)
    

---

### Sampling for $\mathbb{E}_{q_\theta(w \mid D)}[\log P(D \mid w)]$

To train a neural network with VI:

- Select a probability distribution for the weights
    
- Use $\mathcal{L}(\theta)$ as the loss
    
- Implement $\mathbb{E}_{q_\theta(w \mid D)}[\cdot]$ using Monte Carlo sampling
    
- Train using stochastic gradient descent
    

---

### Approximation Quality

The posterior obtained through VI is approximate. The ELBO provides only a loose lower bound.

Factors affecting quality:

- Chosen distribution
    
- Network architecture
    
- Which layers are Bayesian
    
- Type of layers (RNN, CNN, etc.)
    
- Number of samples used in loss
    

---

### Implementation Details

To implement VI:

1. Treat weights/biases as distributions
    
2. Implement stochastic forward pass
    
3. Add KL term to loss
    

---

### Bayes by Backprop [Blundell et al., 2015]

Each weight is modeled as a Gaussian: $w \sim \mathcal{N}(\mu, \rho)$  
The parameters $(\mu, \rho)$ are updated via gradient descent.  
Monte Carlo gradients are used due to stochasticity.

---

### Issues with Bayes by Backprop

- Stochastic loss makes training unstable
    
- Training time increases
    
- Doesn’t scale to large models
    

---

### Implementation Example

A forward pass looks like:

y=a(Wx+b),W∼P(w∣D),b∼P(b∣D)y = a(Wx + b), \quad W \sim P(w \mid D), \quad b \sim P(b \mid D)

Each batch gets one sampled $W$, which reduces diversity.

---

### Flipout for Variational BNNs [Wen et al., 2018]

Instead of sampling one weight per batch, Flipout generates per-sample perturbations:

w=μ+z,z∼N(0,1),w∼N(μ,σ)w = \mu + z, \quad z \sim \mathcal{N}(0, 1), \quad w \sim \mathcal{N}(\mu, \sigma)

Then:

W=W^⋅rsTW = \hat{W} \cdot r s^T

where $r$ and $s$ are Rademacher variables.

---

## Uncertainty Disentanglement - Classification 📊

- **Aleatoric uncertainty**: modeled via logits with variance, using sampling softmax
    
- **Epistemic uncertainty**: from MC-Dropout, ensembles, BBB, Flipout
    

---

### Logits with Uncertainty for Classification 🧮

Gaussian logits with mean $\mu(x)$ and variance $\sigma^2(x)$:

z^j∼N(μ(x),σ2(x))\hat{z}_j \sim \mathcal{N}(\mu(x), \sigma^2(x)) P(y∣x)=1N∑j=1Nsoftmax(z^j)P(y \mid x) = \frac{1}{N} \sum_{j=1}^N \text{softmax}(\hat{z}_j)

---

### Inference-Time Uncertainty Disentanglement ⏱️

Assuming $i \in [1, M]$ samples:

σAle2(x)=Ei[σi2(x)]σEpi2(x)=Vari[μi(x)]\sigma^2_{\text{Ale}}(x) = \mathbb{E}_i[\sigma^2_i(x)] \\ \sigma^2_{\text{Epi}}(x) = \text{Var}_i[\mu_i(x)]

---

### Classification: Probabilities 🎲

pAle(y∣x)=sampling_softmax(μ(x),σAle2(x))HAle(y∣x)=entropy(pAle(y∣x))p_{\text{Ale}}(y \mid x) = \text{sampling\_softmax}(\mu(x), \sigma^2_{\text{Ale}}(x)) \\ H_{\text{Ale}}(y \mid x) = \text{entropy}(p_{\text{Ale}}(y \mid x)) pEpi(y∣x)=sampling_softmax(μ(x),σEpi2(x))HEpi(y∣x)=entropy(pEpi(y∣x))p_{\text{Epi}}(y \mid x) = \text{sampling\_softmax}(\mu(x), \sigma^2_{\text{Epi}}(x)) \\ H_{\text{Epi}}(y \mid x) = \text{entropy}(p_{\text{Epi}}(y \mid x))

where:

μ(x)=1M∑iμi(x)\mu(x) = \frac{1}{M} \sum_i \mu_i(x) entropy(p)=−∑ipilog⁡pi\text{entropy}(p) = -\sum_i p_i \log p_i

---

### Important Details 🔑

In regression:

Predictive Var=Epistemic Var+Aleatoric Var\text{Predictive Var} = \text{Epistemic Var} + \text{Aleatoric Var}

In classification: predictive **logits** can be summed, not the **probabilities**:

Predictive Logits=Epistemic Logits+Aleatoric Logits\text{Predictive Logits} = \text{Epistemic Logits} + \text{Aleatoric Logits}

---

## Overall Concept + Architectures 💡

- **Epistemic uncertainty**: via MC-Dropout, BBB, Flipout
    
- **Aleatoric uncertainty**: via Gaussian NLL / Sampling Softmax
    
- **Predictive uncertainty**: combination of both
    

---

## Comparison of UQ Methods ⚖️

### Impact of Training Set Size 🏋️‍♀️

As training data size changes, **epistemic uncertainty** should also change.