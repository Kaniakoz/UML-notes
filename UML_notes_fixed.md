### I. **Fundamentals of Probability**
- **Definition**:
    $P(E) = \frac{\#\text{ of ways }E\text{ can happen}}{\#\text{ of total outcomes}}$
    - Probabilities lie in the interval $[0,1][0, 1]$.
- **Event Properties**:
    - _Mutual Exclusivity_: $P(A \cup B) = P(A) + P(B)$
    - _Independence_: $P(A \cap B) = P(A)P(B)$
- **Conditional Probability**:
    $P(A|B) = \frac{P(A \cap B)}{P(B)}$
    - If $P(A|B) = P(A)$, then A and B are independent.
- **Bayes Rule**:
    $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
### II. **Probability Distributions**
- **Concept**: Function that maps outcomes to probabilities.
- **Random Variable**: Denoted $X: A \rightarrow \mathbb{R}$; can be:
    - _Discrete_: Countable outcomes
    - _Continuous_: Infinite uncountable outcomes
- **Probability Function (PF)**:
    $f_X(x) = P(X = x)$
    - For discrete variables: called **Probability Mass Function (PMF)**.
- **Kolmogorov Axioms**:
    - Non-negativity: $P(X \in E) \geq 0$       
    - Normalization: $\sum P(X) = 1$
    - Additivity over disjoint sets
### III. **Continuous Distributions**
- **Probability Density Function (PDF)**:
    $P(a \leq X \leq b) = \int_a^b f_X(x) dx$
- **Cumulative Distribution Function (CDF)**:
    $F_X(x) = P(X \leq x) = \int_{-\infty}^x f_X(u) du$
- **Quantile Function**:
    $F^{-1}(q) = \inf \{x : F(x) > q\}$
    - $F^{-1}(0.5)$ = median
### IV. **Statistical Moments**
- **Expectation**:
    - Discrete: $E[X] = \sum x_i P(x_i)$
    - Continuous: $E[X] = \int x f_X(x) dx$
- **Variance**:
    $\text{Var}[X] = E[X^2] - (E[X])^2$
- **Covariance**:
    $\text{Cov}[X, Y] = E[(X - E[X])(Y - E[Y])]$
- **Correlation**:
    $\text{Corr}(X, Y) = \frac{\text{Cov}[X, Y]}{\sigma_X \sigma_Y}$
### V. **Key Distributions**
- **Gaussian Distribution**:
    $f_X(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2}$
- **Uniform Distribution**:
    $f_X(x) = \begin{cases} \frac{1}{b - a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}$
- **Central Limit Theorem (CLT)**:
    $\bar{X}_n \sim N\left(\mu, \frac{\sigma^2}{n}\right) \text{ as } n \to \infty$
# **Intro to Uncertainty Quantification (UQ)**
### I. **Motivation**
- ML models often give overconfident predictions, even for out-of-distribution (OOD) inputs.
- Key applications:
    - Autonomous driving        
    - Medical diagnostics
    - Robotics
### II. **Types of Uncertainty**
- **Aleatoric (Data) Uncertainty**:
    - Caused by noise in observations.
    - Cannot be reduced with more data.
- **Epistemic (Model) Uncertainty**:
    - Due to model ignorance (e.g., lack of training data).
    - Can be reduced with more or better data.
- **Predictive Uncertainty**:
    $\text{Predictive Uncertainty} = \text{Aleatoric} + \text{Epistemic}$
- **Distributional Uncertainty**:
    - Represents model’s ignorance of the correct output distribution.
### III. **Representing Uncertainty**
- **Regression**:
    - Use confidence intervals or mean ± standard deviation.
    - Example: $f(x) \in [a, b]$
- **Classification**:
    - Use discrete probability distribution (e.g., softmax over logits).
    - Softmax often gives overconfident outputs → needs calibration.
- **Entropy**:
    $H = -\sum P(x)\log P(x)$
    - High entropy = high uncertainty.
# **Uncertainty Quantification Methods**

## I. **Bayesian Neural Networks (BNNs)**
### Bayesian Thinking
- **Bayesian inference** updates beliefs based on observed evidence.
- **Bayes’ Rule**:
    $P(H|E) = \frac{P(E|H)P(H)}{P(E)}​$
    where:
    - **Prior (P(H))**: Belief before evidence.
    - **Likelihood (P(E|H))**: How probable evidence is under hypothesis.
    - **Posterior (P(H|E))**: Updated belief after seeing evidence.
    - **Evidence (P(E))**: Normalization term.
### Bayesian in Machine Learning
- We model $P(y|x, w)$ and train to learn $P(w∣D)$.
- Apply Bayes’ Rule to model parameters:
    $P(w|D) = \frac{P(D|w)P(w)}{P(D)}​$
### Interpretation
- **Prior (P(w))**: Initial weight distribution.
- **Likelihood (P(D|w))**: Modeled by network forward pass.
- **Posterior (P(w|D))**: Learned distribution over weights.
### Bayesian Neural Networks
- Model weights are distributions, not fixed values.
- Predictive distribution:
    $P(y|x) = \int P(y|x,w)P(w|D) dw$
### Monte Carlo Approximation
- Exact integral is intractable → use Monte Carlo:
    $P(y|x) \approx \frac{1}{M} \sum_{i=1}^{M} P(y|x, w_i),\quad w_i \sim P(w|D)$
### Challenges
- **Intractable evidence term (P(D))**.
- **High-dimensional posterior** over millions of weights.
- **Monte Carlo** is necessary, but computationally expensive.
## II. **Direct Uncertainty Estimation**
### Two-Headed Networks for Regression
- Outputs: μ(x)\mu(x)μ(x) (mean), σ2(x)\sigma^2(x)σ2(x) (variance).
- Assumes Gaussian output distribution.
### Gaussian Negative Log-Likelihood (NLL)
$-\log L(y, \mu(x), \sigma^2(x)) = \log \sigma^2 + \frac{(y - \mu)^2}{\sigma^2}​$
- Minimizing this encourages:
    - High variance for uncertain predictions.
    - Low variance for confident ones.
### What is learned?
- **Aleatoric uncertainty** (data noise).
- Epistemic uncertainty is not captured.
## III. **Ensembles**
### Training:
- Train M identical models with different initializations.
### Outputs:
- Each model predicts $\mu_i(x)$, $\sigma_i^2(x)$
- Final predictions:
    $\mu^*(x) = \frac{1}{M} \sum_i \mu_i(x)$
	$\sigma^{2*}(x) = \frac{1}{M} \sum_i \left(\sigma_i^2(x) + \mu_i^2(x)\right) - \mu^*(x)^2$
### Pros:
- Combines epistemic + aleatoric uncertainty.
- Outperforms single model methods in OOD settings.
## IV. **DUQ**
### Direct Uncertainty Quantification
- Uses **Radial Basis Functions (RBFs)** centered on class centroids.
- Prediction confidence: inverse of distance to class centroid.
- Formula:
    $K_c(f(x), e_c) = \exp\left(-\frac{\|W_c f(x) - e_c\|^2}{2\sigma^2}\right)$
- **Only for classification**.
## V. **Gradient Uncertainty**
### Idea:
- Use gradient magnitude of the loss wrt input as uncertainty:
    $\sigma(x) = \|\nabla_x L(f(x), y)\|$
- No labels at inference → use predicted class in loss.
## VI. **Sampling-Based Methods**
### Monte Carlo Dropout
- Apply **dropout at inference** to simulate sampling from posterior.
### Monte Carlo DropConnect
- Similar to MC-Dropout but applies dropout to **weights**, not activations.
- Generally better but more complex to implement.
### VII. **Variational Inference (VI)**
- Goal: Approximate intractable posterior $P(w|D)$ with $q_\theta(w|D)$
- Minimize KL divergence:
    $\text{KL}(q, p) = \int q(x)\log\frac{q(x)}{p(x)}dx$
- **ELBO (Evidence Lower Bound)**:
    $\log P(D) \geq \mathbb{E}_{q}[ \log P(D|w) ] - \text{KL}(q(w)||P(w))$
- **Loss Function**:
    $L(\theta) = \text{KL}(q_\theta(w|D)||P(w)) - \mathbb{E}_{q_\theta}[\log P(D|w)]$
- Training via **Monte Carlo estimation** and **stochastic gradient descent**.
### VIII. **Bayesian Neural Networks (BNNs)**
- **Bayes by Backprop**:
    - Each weight ww has Gaussian distribution $N(\mu, \rho)$
    - Training is stochastic, often slow and unstable for large models.
- **Flipout**:
    - Uses per-sample weight perturbations to reduce gradient variance.
    - Improves training stability and convergence.
### IX. **Uncertainty Disentanglement**
- **Objective**: Separate aleatoric and epistemic uncertainties.
- **Regression Setting**:
    $\text{Total Variance} = \mathbb{E}[\sigma_i^2(x)] + \text{Var}[\mu_i(x)]$
    - First term: Aleatoric
    - Second term: Epistemic
- **Classification Setting**:
    - Logits have both mean and variance.
    - Use _Sampling Softmax_:
        $z_j \sim \mathcal{N}(\mu(x), \sigma^2(x))$, $P(y|x) = \frac{1}{N} \sum_{j=1}^{N} \text{softmax}(z_j)$
### X. **Final Thoughts**
- More Bayesian layers → better epistemic uncertainty
- More data → reduces epistemic but not aleatoric uncertainty
- **Active Learning**:
    - Uses epistemic uncertainty to guide which samples should be labeled.
# **Evaluation of Uncertainty Quantification**
## I. **Key Concepts**
- Need **task-specific losses and metrics** to evaluate UQ.
- Ideal: Higher uncertainty ↔ higher error/misclassification.
## II. **Loss Functions**
### Classification:
- **Cross-Entropy**: Considers output probabilities.
- **BCE**: For binary classification.
### Regression with Uncertainty
- **Gaussian NLL**: $\log \sigma^2 + \frac{(y - \mu)^2}{\sigma^2}​$
### Others
- **KL Divergence**: Measures distance between distributions.
## III. **Proper vs Improper Scoring Rules**
### Proper Scoring Rules
- Encourage accurate distributions.
- Examples: log-likelihood, Brier score, KL divergence.
###  Improper Scoring Rules
- Do not account for distributional uncertainty.
- Examples: accuracy, MSE, MAE.
## IV. **Regression Metrics**
### Coverage
$Cov = \frac{1}{N} \sum 1[l_i \le y_i \le u_i]$
### Entropy
- Differential entropy for Gaussians: $h(X) = \frac{1}{2} \log(2\pi e \sigma^2)$
## V. **Classification Metrics**
### Brier Score
$Brier = \frac{1}{N} \sum (y_i - \hat{y}_i)^2$
### Entropy
$H = -\sum p_i \log p_i$
## VI. **Calibration**
### Definition:
- Confidence ≈ observed accuracy.
- Poor calibration misleads users even with good predictions.
### Reliability Plot
- Plot confidence vs accuracy per bin.
- Perfect calibration: diagonal line.
### Metrics
- **Calibration Error (CE)**:
    $CE = \sum |acc(B_i) - conf(B_i)|$
- **Expected Calibration Error (ECE)**:
    $ECE = \sum \frac{|B_i|}{N} |acc(B_i) - conf(B_i)|$
- **Maximum Calibration Error (MCE)**: Largest deviation bin.
## VII. **Regression Calibration**
### Confidence Intervals
- For Gaussian:
    $[l, u] = [\mu - z_{\alpha/2}\sigma, \mu + z_{\alpha/2}\sigma]$
### Calibration Process:
1. Choose set of confidence levels.
2. For each $\alpha$, calculate CI and check if label lies within.
3. Plot ($\alpha$, coverage).
## VIII. **Error vs Confidence**
### Plots
- Group predictions by confidence level.
- Measure average error (e.g., MSE) in each bin.
- Ideal: higher confidence → lower error.
### Alternative Visualization
- Scatter plot: x = confidence, y = error.
- Expect monotonic relationship.
# **Calibration of Probabilities**

> **Calibration Definition:** Confidence predicted by the model should match the actual observed accuracy. For regression, the confidence interval should match the frequency that the true label lies within that interval.
> **Misinterpretation Risk:** Probabilities from models are not inherently calibrated; they must be empirically validated.
### Calibration in Classification
- **Formal Definition:**  
    $P(y|\hat{y} = \alpha) = \alpha$  
    This means if a model predicts class A with 70% confidence, it should be correct 70% of the time when making that prediction.
- **Why Models Are Miscalibrated:**
    - Cross-entropy loss encourages high confidence.
    - Multi-class classification introduces complexity (e.g., one-vs-all).
    - Models like SVMs don't inherently output probabilities.
- **Overconfidence vs Underconfidence:**
    - Overconfident: Model's confidence > actual accuracy.
    - Underconfident: Model's confidence < actual accuracy.
### Calibration in Regression
- **Interval Calibration:**
    $P(l_\alpha \le y \le u_\alpha) = \alpha$
    Requires confidence intervals to capture the correct proportion of true values.
- **Why Miscalibrated:**
    - Gaussian NLL loss only captures aleatoric uncertainty.
    - Calibration error is not a differentiable function (makes training harder).
### Calibration Methods
#### Post-Hoc Methods
- **Platt Scaling:** Fits a sigmoid (logistic) function to map uncalibrated scores to probabilities. Requires tuning parameters A and B via logistic regression.
- **Temperature Scaling:** Adjusts logits by a scalar T before softmax; increasing T reduces confidence. Can use a single T or class-specific TiT_i.
- **Isotonic Regression:** Non-parametric, fits a monotonic function to calibrate scores. More flexible than Platt but can overfit.
- **Empirical Binning:** Uses bins to group confidence levels and assigns them the average accuracy of each bin. Simple and interpretable.
#### Calibration Map:
A function $g : S_Y \rightarrow P_Y$ maps model outputs to calibrated probabilities.
#### Mixup Training:
Data augmentation technique using linear combinations of inputs and labels. Leads to smoother decision boundaries and improved calibration.
$\tilde{x} = \lambda x_i + (1-\lambda) x_j,\quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$
### Evaluation
- **Reliability Diagrams:** Visualize calibration.
- **Expected Calibration Error (ECE):** Measures the difference between predicted confidence and observed accuracy.
- **Sharpness:** Degree of confidence; high sharpness = confident predictions.
- **Scoring Rule Decomposition:**
    $\text{Score} = \text{Calibration Loss} + \text{Refinement Loss}$

# **Out of Distribution Detection (OOD)**
### Core Idea
- **Goal:** Detect when inputs are **outside** the training distribution.
- **Motivation:** Models fail silently on unfamiliar inputs. Need to flag these cases.
### Key Terminology
- **Closed Set Assumption:** Assumes test data contains only known classes.
- **Open Set:** Allows unknown classes during testing.
- **Model/World Knowledge Matrix:**
    - **Known-Knowns:** What model has seen and understands.
    - **Known-Unknowns:** Aware it hasn't seen all classes.
    - **Unknown-Knowns:** Properties present in model but not understood by designers.
    - **Unknown-Unknowns:** Model doesn’t know what it doesn’t know.
### Related Concepts
- **Anomaly Detection:** Train only on normal data; detect deviations.
- **Novelty Detection:** Detect new classes in testing.
- **Open Set Recognition (OSR):** Classify knowns, detect unknowns.
- **Outlier Detection:** Statistical detection over observed dataset.
- **Out of Distribution Detection:** Reject samples not from the training distribution.
### Types of OOD Shifts
- **Covariate Shift:** Input distribution changes; label distribution same.
- **Label Shift:** Labels change but input distribution same.
- **Concept Shift:** Meaning of labels changes geographically or over time.
### Real-World Failure Examples
- Tank recognition overfitted to sky color.
- Medical imaging models learned machine artifacts.
### Evaluation of OOD Detection
- **Datasets:** Need ID (train + test) and OOD (test-only), same modality.
    - e.g., CIFAR-10 (ID) vs SVHN (OOD), Fashion MNIST vs MNIST.
- **Labels:** Binary (0 = ID, 1 = OOD).
- **Metrics:**
    - ROC Curve & AUC.
    - TPR, FPR, Precision, Recall, F1.
###  OOD Detection Methods
#### Epistemic Uncertainty
- **High** outside training data, **low** within.
- Estimation Techniques:
    - **Entropy** of softmax.        
    - **1 - max softmax probability**.
    - **Standard deviation** (regression).
#### Model-Based Methods
- **Dropout, DropConnect, Ensembles, Flipout:** Introduce variability during inference to estimate uncertainty.
#### Training-Based Methods
- **Additional Head:** Trains binary classifier for ID vs OOD. Often fails due to generalization gap.
- **Maximum Discrepancy:** Measures divergence between features for ID and OOD.
#### ODIN 
- Uses **temperature scaling** (e.g., T=1000) and input perturbations to better separate ID/OOD.
#### Energy-Based
- Score:
    $E(x, f) = -T \log \sum_i \exp(f_i(x)/T)$
    Lower energy ⇒ more likely OOD.
#### Generative Methods
- **Likelihood-based:** Train generative model on ID data (e.g., VAE, GAN).
- Problem: May assign high likelihoods to OOD.
- **Solution:** Use **Likelihood Ratio**:
    $LLR(x) = \log P_\theta(x) - \log P_{\theta_0}(x)$
### OOD in Regression
- Focus on **variance output** (epistemic uncertainty).
- Variance increases with distance from training range.
### Challenges & Pitfalls
- **No perfect separation:** Some ID = high uncertainty, some OOD = low.
- **Threshold choice:** Non-trivial and context-dependent.
- **No guarantees:** Performance can degrade unexpectedly.
- **Recommendation:** Use OOD as an alert system, not as a final decision-maker.