## Definition of Probability

> A **probability** is defined for an event as the ratio of possible ways it can happen to the total number of outcomes.

$P(E) = \frac{\text{num of ways E can happen}}{\text{num of total outcomes}}$
- Real numbers in the $[0,1]$ range representing the likelihood/ chance that an event will happen.

## Event Properties

### Mutual Exclusivity

> Events A and B are mutually exclusive if when one happens, the other cannot happen.

$P(A \text{ or } B) = P(A) + P(B)$

### Independence

> Events A and B are independent if one happening does not affect the other.

$P(A \text{ and } B) = P(A)P(B)$

### Conditional Probability

> It is the probability of an event A given that another event B has already occurred, assuming both events have some relationship.

denoted by $P(A|B)$ and computed by definition as $P(A|B) = \frac{P(A \cap B)}{P(B)}$
In general, $P(A) \neq P(A|B) \neq P(B|A) \neq P(B)$.
Independence occurs **only if** $P(A|B) = P(A)$ and/or $P(B|A) = P(B)$.

## Basic Properties of Probability

| Property  | Equation                                                             | Condition                         |
| --------- | -------------------------------------------------------------------- | --------------------------------- |
| Event     | $P(E) \in [0,1]$                                                     |                                   |
| Not Event | $P(\text{not } E) = 1 - P(E)$                                        |                                   |
| A or B    | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$                            |                                   |
| A or B    | $P(A \cup B) = P(A) + P(B)$                                          | If A and B are mutually exclusive |
| A and B   | $P(A \cap B) = P(A \mid B)P(B)=P(B \mid A)P(A)$                      |                                   |
| A and B   | $P(A \cap B) = P(A)P(B)$                                             | If A and B are independent        |
| A given B | $P(A \mid B)=\frac{P(A \cap B)}{P(B)} =\frac{P(B \mid A)P(A)}{P(B)}$ | Bayes rule                        |

## Probability Distributions

### Concept

> A probability distribution is a function that maps from some input variable to probability (discrete) or probability density (continuous) values, characterizing how probability is distributed on the variable.

- The input variable is a **Random Variable**, its value depends on a random outcome or randomness.
- The possible values of the variable are events, subsets of a sample space.
- A sample space is any predefined set (real numbers, matrices, vectors, etc).

### Random Variables
> A random variable is a mathematical concept used to model randomness, where it is an object that depends on a random event.

- **Sample Space**: A random value has a set of allowable values, usually called the sample space and denoted by A.
- **Mapping**: A random variable is a mapping $X: A \rightarrow \mathbb{R}$, where A represents possible outcomes in the sample space, mapped to real numbers (their probability).
- Random variables can be discrete or continuous, which defines their sample space.

### Examples of Random Variables
- **Height**: Select a person randomly and measure their height.
- **Falling Leaves**: Final position of leaves falling from a tree.
- **Passing Bicycles**: Count the number of cyclists passing on a given hour.
### Probability Function (PFs)
> The PF is a function that usually defines a probability distribution.

$f_X(x) = P(X = x)$
- Similar to the mapping X defined previously in a random variable.
- Works well for discrete random variables, but is problematic for continuous ones, because $P(X = x) = 0$ for all x.
- For discrete distributions, this is usually called the **probability mass function**.
### Kolmogorov Axioms
All probability distributions need to follow these axioms:
- Probability values are between 0 and 1: $0 \leq P(X \in E) \leq 1 \mid \forall E \subseteq A$
- Sum of all events is 1: $\sum_{X \in A} P(X) = 1$ $\int_{A} P(X) dX = 1$
- Disjoint Family of Sets: $P(X \in \bigcup_{i} E_i) = \sum_{i} P(X \in E_i)$ for any disjoint family of sets $E_i \subseteq A$
### Probability Density Function (PDFs)
> For continuous probability distributions, the PDF is a function that usually defines a probability distribution.

Models probability density instead of plain probability, can be bigger than one without violating Kolmogorov’s Axioms.
$P(a \leq X \leq b) = \int_{a}^{b} f_X(x)dx$ For any $a \leq b$. 
Note that if $a = b$, then: $P(a \leq X \leq a) = \int_{a}^{a} f_X(x)dx = 0$

### Examples of Real Probability Distributions
- **Height**: Approximately a Gaussian distribution.
- **Falling Leaves**: Likely a Gaussian distribution under some assumptions, like no wind or other influences.
- **Passing Bicycles**: A Poisson distribution.
_Distributions can be engineered or determined from experimental research._
### Building Probability Distributions from Data
- A simple way to visualize a probability distribution is to take samples from that distribution and build a histogram.
- This is an approximation of the true distribution that becomes better with more samples.
- This is called the **empirical distribution** of a sample.
### (Random) Sampling

> Sampling is the process of generating values that follow a given probability distribution.

This is done with random number generators and mathematics.
$x \sim f_X(\text{parameters})$
- $x$ is a sample from distribution density function $f_X$ with given parameters.
- Values produced by sampling only follow the distribution in aggregation, which can be visualized by making histograms of many samples.
### Density Estimation as Opposite of Sampling
Estimate distribution $f_X$ from values or samples $x$ from an unknown distribution.
- This is unsupervised, with no labels, only data points.
- A way to learn structure from data.
- Clear use case in generative models: learn the distribution that generated some data points by only having access to those data points.
### Cumulative Distribution Function (CDFs)
The CDF is a function defined by: $F_X(x) = P(X \leq x)$
This function gives the probability that a random variable X is less than a value x. A special property is: $P(a < X \leq b) = F(b) - F(a)$
### Continuous CDFs
For continuous random variables, there are some additional dualities:
$F_X(x) = \int_{-\infty}^{x} f_X(u)du$
$f_X(x) = \frac{dF_X(x)}{dx}$
The CDF and PDF are related through the integral/derivative of each other.
### Inverse Cumulative Function or Quantile Function
$F^{-1}(q) = \inf{x : F(x) > q}$, where $q ∈ [0,1]$ is a specific quantile.
- If F(x) is a strictly increasing function, then $F^{-1}(q)$ is the unique value $x$ so $F(x) = q$ holds.
- **Quantiles** are equal divisions of the probability space.
- **Percentiles**: divide in 100 divisions.
- **Quartiles**: over 4 divisions.
- **Median**: The percentile 50 is always the median ($F^{-1}(0.5)$).
### Expectation
> Expectation or expected value is a linear operation defined for continuous distributions with PDF $f_X$ as:

$E[X] = \int_{-\infty}^{\infty} x f_X(x) dx$
And for discrete distributions as: $E[X] = \sum_{i=0}^{\infty} x_i f_X(x_i)$
The expected value $E[X]$ is associated with the mean, while the variance can be computed as $Var[X] = E[X^2] - (E[X])^2$.
### Covariance
> Covariance is a measure of the joint variation between two related variables X and Y.

$Cov[X, Y] = E[(X - E[X])(Y - E[Y])]$
If covariance is normalized with the standard deviation of each variable ($\sigma_X, \sigma_Y$), you obtain the correlation: $CORR(X, Y) = \frac{E[(X - E[X])(Y - E[Y])]}{\sigma_X \sigma_Y}$
### Transformations on Random Variables
Some important identities and transformations on random variables X and Y:
- $Var[X + Y] = Var[X] + 2Cov[X, Y] + Var[Y]$
- $Var[X - Y] = Var[X] - 2Cov[X, Y] + Var[Y]$
- $Var[X] = Cov[X, X] = E[X^2] - (E[X])^2$
For a constant k:
- $Var[X + k] = Var[X]$
- $Var[kX] = k^2 Var[X]$
### Some Special Transformations and Laws
- Law of Total Variance: $Var[Y] = E[Var[Y|X]] + Var[E[X|Y]]$
- Law of Total Expectation: $E[X] = E[E[X|Y]]$
### Addition of Probability Distributions
The addition/sum of two random variables is the convolution of their PDFs. If we have $X \sim f_X, Y \sim f_Y$, then $Z = X + Y$ has PDF: $f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z - x) dx$
### Transformations on Random Variables
Passing random variables through a function is generally difficult. If we have X with a distribution with PDF $f_X$, and a function g that is increasing in the domain of X, then the PDF of $Y = g(X)$ is given by:
$f_Y(y) = f_X(g^{-1}(y)) | \frac{dg^{-1}(y)}{dy} |$
Valid for continuous distributions only. For a discrete distribution X with probability mass function $f_X$, then the transformation is:
$f_Y(y) = \sum_{x \in g^{-1}(y)} f_X(x)$

## Common Probability Distributions

### Normal or Gaussian Distribution
> A continuous distribution defined for $x \in \mathbb{R}$ with two parameters, mean $µ$ and variance $\sigma^2$, and with PDF given by:

$f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-0.5 \frac{(x - \mu)^2}{\sigma^2}}$
The expectation and variance of the Gaussian distribution $X \sim N(\mu, \sigma^2)$ is:
$E[X] = \mu$, $Var[X] = \sigma^2$

### Uniform Distribution
> Continuous or Discrete distribution where all values in a range $[a,b]$ are equally likely.

It is parameterized by a and b and has PDF/mass function as: $f_X(x) = \frac{1}{b - a}$ if $a \leq x \leq b$, 0 otherwise
It is usually denoted as $X \sim U(a, b)$. The mean is $E[X] = 0.5(a + b)$ and variance is $Var[X] = \frac{1}{12} (b - a)^2$.
### Central Limit Theorem
> Given a sequence of random variables $X_1, X_2, ..., X_n$ that are independent and identically distributed with population mean $µ$ and variance $\sigma^2$.

Define the sample mean or average: $\bar{X} = \frac{X_1 + X_2 + ... + X_n}{n}$
The central limit theorem states that as $n \rightarrow \infty$, the distribution of $\bar{X}$ is a Gaussian distribution: $\bar{X} \sim N(\mu, \frac{\sigma^2}{n})$
### Random Numbers given a Probability Distribution
Sometimes it is needed to generate numbers not from a uniform distribution, but following another distribution with a given CDF or PDF.
#### Inversion Method
If we generate $U \sim \text{Uniform}(0, 1)$, then with the following computation using the target CDF $F_X$: $X = F_X^{-1}(U)$
$X$ will follow the distribution $P$ defined by CDF $F_X$. This only works if $F_X$ is invertible.
#### Rejection Sampling
This method works using a proposal distribution Y that has PDF g, to generate samples of distribution X with PDF f. It works like this:
1. Generate sample y from Y and a sample u from Uniform(0,1).
2. If $u < \frac{f(y)}{Mg(y)}$ Accept y as a sample drawn from X.
3. If not, reject y and return to step 2.
M is a constant that fulfills $f(x) < Mg(x)$, and $1 < M < \infty$. It is selected as a scale to make sure f and g are compatible and also decides the accept rate $1/M$, so on average, a sample is generated every M iterations of this algorithm.
### A Note on Probability Distributions
In general, probability distributions are a difficult concept to grasp because:
- They are abstract mathematical objects that model (ideal) behavior in the world.
- The only way to get a number from a probability distribution is to generate a sample from it.
- Their behavior can only be seen through obtaining multiple samples and making a histogram (to visualize the distribution).
### Importance for the Course
- The concepts presented here are the base for the rest of the course.
- Probability Distributions are key for **Uncertainty Quantification** in Machine Learning.
- Sampling is also a key concept, as a lot of UQ methods rely on sampling to produce a distribution.
### Pre-requisite Knowledge for this Course
- Concept of a Probability Distribution, related functions (PDF, CDF, etc).
- Differences between sampling and density estimation, and how they relate to real-world probability distributions.
- Concepts of Expectation and Covariance.
- The central limit theorem, and basic distributions like Gaussian and Uniform.
## Introduction to Uncertainty Quantification

### Motivation
Standard machine learning models may produce misleading **softmax confidences**.
As ML is used for real processes, it's essential to evaluate the certainty of predictions on unseen samples.
- **Autonomous Driving**: Decision-making needs reliable predictions.
- **Medical Applications**: Low confidence may require additional tests.
ML models need to answer:
- Do I know that I do not know?
- Can I refuse to provide an answer?
This relates to **out of distribution detection**, where a model can indicate input dissimilarity to the training set and refuse to answer.
Real-world datasets can be unbalanced, noisy, and lead to overconfident neural networks.
Most ML models don't explicitly model uncertainty and produce point-wise predictions instead of distributions. Neural networks are often overconfident, producing wrong predictions with high confidence.
### Practical Applications of Uncertainty
- Detecting misclassified examples.
- Rejecting outputs if uncertainty is too high (out of distribution detection).
- Informing users about the reliability of predictions.
### Types of Uncertainty
- **Aleatoric Uncertainty (AU)**: Inherent to the data, cannot be reduced by adding more information (e.g., sensor noise).
- **Epistemic Uncertainty**: Produced by the model, reducible by adding more information (e.g., model misspecification, lack of training data).

### Data/Model Uncertainty
#### Regression
- **Data Uncertainty**: Additive noise added to labels, noisy inputs from sensors.
- **Model Uncertainty**: Missing data, low density in areas, inputs outside of training range, uncertainty in model parameters.
#### Classification
- **Data Uncertainty**: Incorrect labels, noisy inputs from sensors.
- **Model Uncertainty**: Missing data, low density in areas, inputs outside of training range, unclear decision boundary.
### Predictive and Distributional Uncertainty
- **Predictive Uncertainty**: Combination of aleatoric and epistemic uncertainty.
    $\text{Predictive Uncertainty} = Aleatoric + Epistemic$
- **Distributional Uncertainty**: Lack of knowledge about the correct output distribution.
    $Predictive Uncertainty = Aleatoric + Epistemic + Distributional$
### Uncertainty Disentanglement

### Uncertainty Representation
Uncertainty information is represented in multiple ways, depending on the task.
- The most generic representation is to use a **probability distribution** on the output.
#### Regression
- **Confidence Intervals**: Output within a defined interval $[a,b]$. $f(x) \in [a, b]$
- **Mean and Variance**: Uncertainty represented as the variation of the output from the mean. $f(x) \pm \sigma$
- Equivalent interval: $f(x) \in [f(x) - \sigma, f(x) + \sigma]$
#### Classification
- The only robust representation is to use a discrete probability distribution.
- The easiest way to implement it is to use a softmax activation (for multi-class) or a sigmoid activation (for binary classification).
### Overconfidence and Calibration
Models often produce probabilities or confidence intervals that are overconfident. Probabilities or confidence intervals must represent the likelihood of correct prediction, which is measured by **calibration**.
### Entropy
> Entropy is a measurement of the "information content" in a probability distribution.

$H = - \sum_{x} P(x) \log P(x)$
- Entropy is directly related to uncertainty.
- Units of entropy are called bits if using the base-two logarithm.
- The uniform distribution maximizes entropy, being harder to predict.
- For a fixed mean and variance, the Gaussian distribution maximizes entropy.
### Challenges
**Embodiment** is the main difference between Robot Learning/Perception and Machine/Deep Learning.
- **Medical Systems and Decision Making**: Requires correct epistemic uncertainty estimates.
- **Robotics**: Useful uncertainties are not modeled, e.g., uncertainty in dynamical systems, perception, or when robot capabilities are extrapolated.
- **Reinforcement Learning**: Policies need to estimate their own epistemic uncertainty.
- **Autonomous Driving**: Engineering safer methods and using safety measures are crucial.

## Unusual Situations and Safe Learning Systems
Autonomous Driving (AD) systems must detect unusual situations with nearly 100% precision, alerting the driver. Current methods rely on humans to label usual situations, which doesn't scale and can't achieve super-human driving.
Examples of failures include:
- Experimental autonomous vehicles hitting human pedestrians.
- Biases in face recognition algorithms against certain skin colors.
AI/Robotics should be developed for social good, tuning algorithms for maximum safety.








