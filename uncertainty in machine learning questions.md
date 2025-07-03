- What are Aleatoric and Epistemic Uncertainty?
	- **Aleatoric Uncertainty**: This is data uncertainty, arising from noise inherent in the observations, such as sensor noise or stochastic events. It **cannot be reduced** by collecting more data.
	- **Epistemic Uncertainty**: This is model uncertainty, due to limited data or model mis-specification. It **can be reduced** by obtaining more or better data.
- What is Disentangling of Uncertainty?
	- Disentangling uncertainty means **separating** the total predictive uncertainty into **aleatoric and epistemic** components. This helps in understanding whether uncertainty is due to noisy data (aleatoric) or lack of model knowledge (epistemic).
- Give some real-world examples of Uncertainty that are different to the ones mentioned in the lecture.
	- **Aleatoric**: Fluctuating electricity usage in a household due to random human activity.
	- **Epistemic**: Weather prediction in a region with sparse sensor coverage.
	- **Distributional**: Autonomous vehicle encountering a novel road sign not seen during training.
- What is the purpose of uncertainty quantification?
	- Detect out-of-distribution inputs.
	- Avoid overconfident wrong predictions.
	- Guide decision-making in safety-critical systems (e.g., medical, autonomous driving).
	- Enable models to say “I don’t know.”
- What is active learning and how it is related to uncertainty quantification?
	- Active Learning allows models to **select informative samples** for labeling. UQ helps identify **high-uncertainty** samples that are most beneficial to learn from, making training more efficient. Based on maximum informational gain, human giving labels, reach higher accuracy with fewer data points.
- Explain the overall Bayesian Neural Network framework.
	- BNNs model **weights as probability distributions**, not point estimates.
	- During inference, predictions are obtained by **marginalizing** over these weight distributions.
	- This gives a predictive distribution $P(y|x) = \int P(y|x, w)P(w|D) dw$
- What is the intuition for the Gaussian Negative Log-Likelihood?
	- Gaussian NLL penalizes the model based on how well it predicts the **mean** and **variance** of the data:
	- Encourages low variance where the model is accurate.
	- Increases predicted variance in noisy regions, this helps capture **aleatoric uncertainty**.
- How does MC-DropConnect work? and how it is different from MC-Dropout?
	- **MC-DropConnect**: Randomly drops weights during inference.
	- **MC-Dropout**: Randomly drops activations (neurons) during inference.  
	    Both create a **distribution of predictions** by repeated forward passes, approximating the predictive posterior.
- What is the difference between direct and sampling-based UQ methods?
	- **Direct UQ**: Outputs uncertainty from a single forward pass (e.g., two-headed models, but not only).
	- **Sampling-based UQ**: Requires multiple stochastic forward passes (e.g., MC-Dropout), often more accurate but computationally expensive, samples the data.
- What are the disadvantages of the Bayesian framework applied to Neural Networks?
	- Computationally expensive.
	- Intractable posterior distributions.
	- Requires approximation (e.g., Variational Inference).
	- Poor scalability to large models.
- What is Variational Inference in Neural Networks (in concept)?
	- It approximates the intractable posterior $P(w∣D)$ with a simpler distribution $q(w∣D)$. The goal is to minimize the KL divergence between them, using a loss function that includes:
		- The KL term between prior and approximate posterior.
		- Expected log-likelihood of the data.
- What are some disadvantages of Variational Inference for BNNs?
	- Approximation quality is unknown.
	- Training is unstable (noisy gradients).
	- Computationally intensive.
	- Not scalable to very deep networks without tricks (e.g., Flipout).
- What is different between Bayes by Backprop and Flipout?
	- **Bayes by Backprop**: Samples one set of weights per batch.
	- **Flipout**: Samples different weights per example in a batch using perturbations, reducing variance and improving training stability.
- What is the overall concept to disentangle aleatoric/epistemic uncertainty?
	- Use models like ensembles or BNNs.
	- Compute **total predictive uncertainty** and then separate:
	    - **Aleatoric**: From predicted variance (e.g., in two-headed models).
	    - **Epistemic**: From variance across model predictions or weight samples.
- How is disentangling uncertainty in classification and regression different?
- What scores can we use for OOD detection?
	entropy, maximum probability, epistemic uncertainty, std
- How is OOD detection different from anomaly detection?
	anomaly only has data and we're looking for anomalies in it, no model required
- How is a threshold selected for OOD detection?
	roc curve, best threshold is closest to 0% FPR, 100% TRP
- Can label and covariate shift happen at the same time?
	yes
- What are the differences between a confidence and a probability?
	prob between o and 1, confidence is not normalised
- How is uncertainty different from confidence?
	they are opposites
- What is the purpose of Uncertainty Quantification?
	to get an estimate from the model about it's uncertainty
- Explain how MC-DropConnect differs from Droput?
	drop connect zeros into weights, dropout activation
- What is Monte Carlo Sampling?
	randomness used to compute sth
- Can DUQ be used for regression problems?
	direct uncertainty quantification cannot be used for that
- How to combine MC-Dropout and Gaussian NLL loss?
	mc epistemic, gaussian aleatoric
- What is the overall concept of the derivation of the Gaussian NLL loss?
	maximum likelihood estimation
- What is an accuracy vs confidence plot, and what relationship does it display?
	calibration displayed 
- Explain how a reliability/calibration plot is made.
	accuracy on y, confidence on x, confidence bins
- How is making a reliability plot for classification different from regression?
	use bins for classification, no bins for regression (use confidence intervals there)
- What is the concept of a proper scoring rule?
	no way to cheat the metric, obtain its best value only by predicting the true distribution q
- What is the purpose of calibration?
	so that confidence matches accuracy
- How does platt scaling work conceptually?
	feed a sigmoid function to a calibration plot
- How does any calibration method affect predictions and confidences?
	no effect on predictions, effect on confidences (it changes confidence to match with accuracy)

problem with multivariate gaussians - covaraince scales quadratically with the number of weights