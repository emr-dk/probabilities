# Probabilities
One of the major cornerstones of statistics, machine learning and inference. As such I felt it relevant to revisit my knowledge on this subject. I have gone through most of the book and then revisited the parts I felt were most relevant for myself - and for later use. 

## 1. Probabilities and counting
**Naive definition of probability** (p6)

If $S$ is a finite sample space and $A$ is an event, then the probability of $A$ is the fraction of outcomes in $S$ that are in $A$.

$$
P_{naive}(A) = \frac{\lvert A \rvert}{\lvert S \rvert} = \frac{\text{No. of outcomes in favor of} A}{\text{total number of outcomes in }S}
$$

**Multiplication rule** (P8)

If we have a compound experiment consisting of Experiment A with $a$ outcomes and Experiment B with $b$ outcomes, the possible outcomes will be $a \cdot b = b \cdot a$ 

probability multiplication rule
$$
P(A \cap B) = P(A)P(B)
$$

**Addition rule** (p9)

If we have a compound experiment consisting of Experiment A with $a$ outcomes and Experiment B with $b$ outcomes, the possible outcomes will be $a + b$

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

**Complement rule** (p10)

The complement of an event $A$ is the event $A^C$ consisting of all outcomes in $S$ that are not in $A$. The probability of $A^C$ is $1-P(A)$

**Partition rule** (p10)

If $A_1,\ldots,A_n$ is a partition of $S$ then $P(A) = \sum_{i=1}^n P(A \cap A_i)$

**Sampling with replacement (p11)**

Making $k$ choices from $n$ objects has $n^k$ outcomes

**Sampling without replacement (p12)**

Making $k$ choices from $n$ has 
$n(n-1) \cdots (n-k+1), 1 \le k \le n$ outcomes

**Permutations** (p13)

The number of permutations of $n$ objects is $n!$

**Combinations** (p14)

The number of combinations of $k$ objects from $n$ objects is ${n \choose k} = \frac{n!}{(n-k)!k!}$

**Binomial theorem** (p15)

For any $n \ge 0$ and $x,y \in \mathbb{R}$ we have:
$$(x+y)^n = \sum_{k=0}^n {n \choose k} x^k y^{n-k}$$


**Binomials**

For any integer where $k \ge 0$ and $n \ge 0$ the binomial coefficient is the number of subsets of size $k$ for $S$ with size $n$. Written as ${n\choose k}$ and read as $n$ choose $k$.

Where $k \le n$ the binomial coefficient formula is written as 
$${n\choose k} = \frac{n(n-1)\cdots(n-k+1)}{k!}=\frac{n!}{(n-k)!k!}$$

# 2. Conditional probabilities 
Conditional probabilities are used to describe the probability of an event given that another event has occurred. 

**Conditional probability (p46)**

If $A$ and $B$ are events and $P(B) > 0$ then the conditional probability of $A$ given $B$ is:
$$P(A|B) = \frac{P(A\cap B)}{P(B)}$$

**Probability of the intersection of two events** (p52)

For any events A and B with positive probabilities:
$$P(A \cap B) = P(B)P(A|B) = P(A)P(B|A)$$

**Bayes rule** (p53)

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

**Odds** (p53)

The odds of A are:
$$\text{odds}(A) = \frac{P(A)}{P(A)^C}$$

**Law of total probability** (p54)

Let $A_1,\ldots,A_n$ be a partition of sample space $S$ - the events $A_i$ are disjoint and their union is $S$), then:
$$P(B) = \sum_{i=1}^n P(B|A_i)P(A_i)$$

**Inclusion-exclusion (p59)**

For any events $A$ and $B$:
$$P(A \cup B |E) = P(A|E) + P(B|E) - P(A \cap B|E)$$


**Independence of two events** (p60)

Two events $A$ and $B$ are independent if $P(A \cap B) = P(A)P(B)$

**Mutually independent events** (p61)

A collection of events $A_1,\ldots,A_n$ are mutually independent if for every subset of indices $i_1,\ldots,i_k$ we have:
$$P(A_{i_1} \cap \cdots \cap A_{i_k}) = P(A_{i_1}) \cdots P(A_{i_k})$$

**Independence of two events**
$$P(A \cap B) = P(A)P(B)$$

# 3. Random variables and their distributions
**Random variable** 

Given an experiment with sample space $S$, a random variable is a function $X: S \rightarrow \mathbb{R}$.

**Discrete random variable** 

A random variable $X$ is discrete if it takes on a finite or countably infinite number of values.

**Distributions**

Distributions are used to describe the probability of a random variable taking on a value in a given interval.

The distribution function of a random variable $X$ is the function $F_X: \mathbb{R} \rightarrow [0,1]$ defined by $F_X(x) = P(X \le x)$ for all $x \in \mathbb{R}$.

**Probability mass function (pmf)**

Probability mass functions are used to describe the probability of a discrete random variable taking on a specific value.

The probability mass function of a discrete random variable $X$ is the function $p_X: \mathbb{R} \rightarrow [0,1]$ defined by $p_X(x) = P(X=x)$ for all $x \in \mathbb{R}$.

**Binomial distribution**

The binomial distribution with parameters $n$ and $p$ is the discrete probability distribution of the number of successes in a sequence of $n$ independent experiments, each asking a yes-no question, and each with its own boolean-valued outcome: success/yes/true/one (with probability $p$) or failure/no/false/zero (with probability $q = 1 - p$). A single success/failure experiment is also called a Bernoulli trial or Bernoulli experiment and a sequence of outcomes is called a Bernoulli process; for a single trial, i.e., $n = 1$, the binomial distribution is a Bernoulli distribution. The binomial distribution is the basis for the popular binomial test of statistical significance.

**Bernoulli distribution**

The Bernoulli distribution is the probability distribution of a random variable which takes the value 1 with probability $p$ and the value 0 with probability $q = 1 - p$. It can be used to represent a coin toss where 1 and 0 would represent "head" and "tail" (or vice versa), respectively, and $p$ would be the probability of the coin landing on heads or tails, respectively. In particular, unfair coins would have $p \neq 0.5$.

**Hypergeometric distribution** (p19)

In probability theory and statistics, the hypergeometric distribution is a discrete probability distribution that describes the probability of $k$ successes (random draws for which the object drawn has a specified feature) in $n$ draws, without replacement, from a finite population of size $N$ that contains exactly $K$ objects with that feature, wherein each draw is either a success or a failure. In contrast, the binomial distribution describes the probability of $k$ successes in $n$ draws with replacement.

**Discrete uniform distribution**

In probability theory and statistics, the discrete uniform distribution is a symmetric probability distribution whereby a finite number of values are equally likely to be observed; every one of $n$ values has equal probability $1/n$. Another way of saying "discrete uniform distribution" would be "a known, finite number of outcomes equally likely to happen". A simple example of the discrete uniform distribution is throwing a fair die. The possible values are 1, 2, 3, 4, 5, 6, and each time the dice is thrown the probability of a given score is 1/6. If two dice are thrown and their values added, the resulting distribution is no longer uniform since not all sums have equal probability.

**Cumulative distribution function (cdf)**

The cumulative distribution function of a random variable $X$ is the function $F_X: \mathbb{R} \rightarrow [0,1]$ defined by $F_X(x) = P(X \le x)$ for all $x \in \mathbb{R}$.

In other words the cdf of a random variable $X$ evaluated at $x$ is the probability that $X$ will take a value less than or equal to $x$.

**Functions of random variables**

If $X$ is a random variable and $g$ is a function, then $Y = g(X)$ is also a random variable. The distribution of $Y$ is given by:
$$F_Y(y) = P(Y \le y) = P(g(X) \le y) = P(X \le g^{-1}(y)) = F_X(g^{-1}(y))$$

**Independent random variables**

Two random variables $X$ and $Y$ are independent if for all $x,y \in \mathbb{R}$ we have:
$$P(X \le x, Y \le y) = P(X \le x)P(Y \le y)$$

# Expectations
**Definition of Expectation**

The expectation of a random variable $X$ is the function $E(X): \mathbb{R} \rightarrow \mathbb{R}$ defined by $E(X) = \sum_{x \in \mathbb{R}} x p_X(x)$.

In other words the expectation of a random variable $X$ is the sum of the product of each value of $X$ and the probability of that value occurring.

**Linearity of Expectation**

If $X$ and $Y$ are random variables and $a,b \in \mathbb{R}$ then $E(aX + bY) = aE(X) + bE(Y)$.

What this means is that the expectation of a linear combination of random variables is equal to the linear combination of the expectations of the random variables.

**Geometric and negative binomial distributions**
In probability theory and statistics, the negative binomial distribution is a discrete probability distribution that models the number of successes in a sequence of independent and identically distributed Bernoulli trials before a specified (non-random) number of failures (denoted $r$) occurs. For example, if we define a 1 as failure, all non-1 values as successes, and we throw a die repeatedly until the third time 1 appears (r = 3), then the probability distribution of the number of non-1 values that appear before the third 1 is a negative binomial distribution.

The geometric distribution is a special case of the negative binomial distribution where the number of successes $r$ is equal to 1.

The geometric distribution is either of two discrete probability distributions:

The probability distribution of the number $X$ of Bernoulli trials needed to get one success, supported on the set $\{ 1, 2, 3, \ldots \}$
The probability distribution of the number $Y = X − 1$ of failures before the first success, supported on the set $\{ 0, 1, 2, 3, \ldots \}$
Which of these one calls "the" geometric distribution is a matter of convention and convenience.

A random variable $X$ has a geometric distribution with parameter $p$ if $X$ takes on values in $\{1,2,\ldots\}$ with $P(X=k) = p(1-p)^{k-1}$ for $k=1,2,\ldots$.

**Indicator random variables**

An indicator random variable is a random variable that takes on the value 1 with probability $p$ and the value 0 with probability $q = 1 - p$. It can be used to represent a coin toss where 1 and 0 would represent "head" and "tail" (or vice versa), respectively, and $p$ would be the probability of the coin landing on heads or tails, respectively. In particular, unfair coins would have $p \neq 0.5$.

**Law of the unconcious statistician (LOTUS)**

The law of the unconscious statistician states that if $X$ is a random variable and $g$ is a function, then $E(g(X)) = \sum_{x \in \mathbb{R}} g(x) p_X(x)$. In other words, the expectation of a function of a random variable is the sum of the function of each value of the random variable multiplied by the probability of that value occurring.

It is called the law of the unconscious statistician because it is often used to calculate the expectation of a random variable without knowing the distribution of the random variable.

**Variance**

The variance of a random variable $X$ is the expectation of the squared difference between $X$ and its expectation.

This is defined by the function $\text{Var}(X): \mathbb{R} \rightarrow \mathbb{R}$ defined by $\text{Var}(X) = E((X-E(X))^2)$.

**Poisson**

The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event. The Poisson distribution can also be used for the number of events in other specified intervals such as distance, area or volume.

A random variable $X$ has a Poisson distribution with parameter $\lambda$ if $X$ takes on values in $\{0,1,\ldots\}$ with $P(X=k) = \frac{\lambda^k}{k!}e^{-\lambda}$ for $k=0,1,\ldots$.

**Connection between Poisson and Binomial distributions**

If $X$ is a binomial random variable with parameters $n$ and $p$, then $X$ is approximately Poisson with parameter $\lambda = np$ when $n$ is large and $p$ is small.

# Continuous random variables
**Probability density function**

The probability density function of a continuous random variable $X$ is the function $f_X: \mathbb{R} \rightarrow [0,1]$ defined by $f_X(x) = \frac{d}{dx} F_X(x)$ for all $x \in \mathbb{R}$.

**Uniform distribution**

The uniform distribution is a continuous probability distribution whereby a finite number of values are equally likely to be observed; every one of n values has equal probability $1/n$. The support is defined by the two parameters, $a$ and $b$, which are its minimum and maximum values. The distribution is often abbreviated U(a,b). It is the maximum entropy probability distribution for a random variable X under no constraint other than that it is contained in the distribution's support.

A random variable $X$ has a uniform distribution on the interval $[a,b]$ if $X$ takes on values in $[a,b]$ with $f_X(x) = \frac{1}{b-a}$ for $x \in [a,b]$ and $f_X(x) = 0$ for $x \notin [a,b]$.

**Universality of the uniform**

If $U$ is a uniform random variable on $[0,1]$ and $F$ is a cumulative distribution function, then $F^{-1}(U)$ is a random variable with cumulative distribution function $F$. In other words, if $U$ is a uniform random variable on $[0,1]$ and $F$ is a cumulative distribution function, then $F^{-1}(U)$ has the same distribution as $F$.

**Normal distribution**

In probability theory, the normal (or Gaussian) distribution is a very common continuous probability distribution. Normal distributions are important in statistics and are often used in the natural and social sciences to represent real-valued random variables whose distributions are not known. A random variable with a Gaussian distribution is said to be normally distributed and is called a normal deviate.

A random variable $X$ has a normal distribution with parameters $\mu$ and $\sigma$ if $X$ takes on values in $(-\infty,\infty)$ with $P(X=x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ for $x \in (-\infty,\infty)$.

**Exponential**

The exponential distribution (also known as negative exponential distribution) is the probability distribution that describes the time between events in a Poisson point process, i.e., a process in which events occur continuously and independently at a constant average rate. It is a particular case of the gamma distribution. It is the continuous analogue of the geometric distribution, and it has the key property of being memoryless. In addition to being used for the analysis of Poisson point processes it is found in various other contexts.

A random variable $X$ has an exponential distribution with parameter $\lambda$ if $X$ takes on values in $[0,\infty)$ with $P(X=x) = \lambda e^{-\lambda x}$ for $x \in [0,\infty)$.

**Poisson process**

A Poisson process is a stochastic process that counts the number of events and the time that these events occur in a given time interval. The time between each pair of consecutive events has an exponential distribution with parameter $\lambda$. The number of events in a given time interval has a Poisson distribution with parameter $\lambda t$. The Poisson process is the continuous analogue of the binomial distribution.

# 6. Moments
Moments are a set of quantities used to describe the shape of a probability distribution. The $k\text{th}$ moment of a random variable $X$ is defined by $E(X^k)$. The first moment is the mean, the second moment is the variance, and the third moment is the skewness.

The $n$th moment of a random variable $X$ is the function $E(X^n): \mathbb{R} \rightarrow \mathbb{R}$ defined by $E(X^n) = \sum_{x \in \mathbb{R}} x^n p_X(x)$.

**Central moment**
The $n$th central moment of a random variable $X$ is the function $\text{Cov}(X^n): \mathbb{R} \rightarrow \mathbb{R}$ defined by $\text{Cov}(X^n) = E((X-E(X))^n)$.

**Cumulant**
The $n$th cumulant of a random variable $X$ is the function $\kappa_n: \mathbb{R} \rightarrow \mathbb{R}$ defined by $\kappa_n = \frac{d^n}{dt^n} \log M_X(t) \big|_{t=0}$.


**Summaries of a distribution**

The mean, variance, skewness, and kurtosis are all summaries of a distribution.

**Mean**

The mean of a random variable $X$ is the expectation of $X$. It is defined by $E(X)$.


**Variance**

The variance of a random variable $X$ is the expectation of the squared difference between $X$ and its expectation.

This is defined by the function $\text{Var}(X): \mathbb{R} \rightarrow \mathbb{R}$ defined by $\text{Var}(X) = E((X-E(X))^2)$.

**Skewness**

The skewness of a random variable $X$ is the third standardized moment of $X$. It is defined by $\text{Skew}(X) = \frac{E((X-E(X))^3)}{\text{Var}(X)^{3/2}}$.

**Kurtosis**

The kurtosis of a random variable $X$ is the fourth standardized moment of $X$. It is defined by $\text{Kurt}(X) = \frac{E((X-E(X))^4)}{\text{Var}(X)^2}$.

**Interpretation of moments**

The mean is the center of mass of the distribution. The variance is the average squared distance from the mean. The skewness is a measure of the asymmetry of the distribution. The kurtosis is a measure of the heaviness of the tails of the distribution.

**Moment generating function**

The moment generating function of a random variable $X$ is the function $M_X: \mathbb{R} \rightarrow [0,1]$ defined by $M_X(t) = E(e^{tX})$.


# 7. Joint distributions
**Joint, marginal and conditional distributions**

The joint distribution of random variables $X$ and $Y$ is the distribution of the pair $(X,Y)$. The marginal distribution of $X$ is the distribution of $X$ ignoring $Y$. The conditional distribution of $X$ given $Y$ is the distribution of $X$ given that $Y$ is known.

**Joint probability mass function**

The joint probability mass function of discrete random variables $X$ and $Y$ is the function $p_{X,Y}: \mathbb{R}^2 \rightarrow [0,1]$ defined by $p_{X,Y}(x,y) = P(X=x,Y=y)$ for all $(x,y) \in \mathbb{R}^2$.

**Joint cumulative distribution function**

The joint cumulative distribution function of random variables $X$ and $Y$ is the function $F_{X,Y}: \mathbb{R}^2 \rightarrow [0,1]$ defined by $F_{X,Y}(x,y) = P(X \le x, Y \le y)$ for all $(x,y) \in \mathbb{R}^2$.

**Joint probability density function**

The joint probability density function of continuous random variables $X$ and $Y$ is the function $f_{X,Y}: \mathbb{R}^2 \rightarrow [0,1]$ defined by $F_{X,Y}(x,y) = \frac{\partial^2}{\partial x \partial y} F_{X,Y}(x,y)$ for all $(x,y) \in \mathbb{R}^2$.

**Marginal probability mass function**

The marginal probability mass function of a discrete random variable $X$ is the function $p_X: \mathbb{R} \rightarrow [0,1]$ defined by $p_X(x) = P(X=x) = \sum_{y \in \mathbb{R}} p_{X,Y}(x,y)$ for all $x \in \mathbb{R}$.

**Marginal cumulative distribution function**

The marginal cumulative distribution function of a random variable $X$ is the function $F_X: \mathbb{R} \rightarrow [0,1]$ defined by $F_X(x) = P(X \le x) = F_{X,Y}(x,\infty)$ for all $x \in \mathbb{R}$.

**Marginal probability density function**

The marginal probability density function of a continuous random variable $X$ is the function $f_X: \mathbb{R} \rightarrow [0,1]$ defined by $f_X(x) = \int_{-\infty}^\infty f_{X,Y}(x,y) dy$ for all $x \in \mathbb{R}$.

**Conditional probability mass function**

The conditional probability mass function of a discrete random variable $X$ given $Y=y$ is the function $p_{X|Y=y}: \mathbb{R} \rightarrow [0,1]$ defined by $p_{X|Y=y}(x) = P(X=x|Y=y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}$ for all $x \in \mathbb{R}$.

**Conditional cumulative distribution function**

The conditional cumulative distribution function of a random variable $X$ given $Y=y$ is the function $F_{X|Y=y}: \mathbb{R} \rightarrow [0,1]$ defined by $F_{X|Y=y}(x) = P(X \le x|Y=y) = \frac{F_{X,Y}(x,y)}{F_Y(y)}$ for all $x \in \mathbb{R}$.

**Conditional probability density function**

The conditional probability density function of a continuous random variable $X$ given $Y=y$ is the function $f_{X|Y=y}: \mathbb{R} \rightarrow [0,1]$ defined by $f_{X|Y=y}(x) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$ for all $x \in \mathbb{R}$.

**Independence**

Random variables $X$ and $Y$ are independent if $F_{X,Y}(x,y) = F_X(x) F_Y(y)$ for all $(x,y) \in \mathbb{R}^2$.

**Covariance**

The covariance of random variables $X$ and $Y$ is the function $\text{Cov}(X,Y): \mathbb{R}^2 \rightarrow \mathbb{R}$ defined by $\text{Cov}(X,Y) = E((X-\mu_X)(Y-\mu_Y))$.

**Correlation**

The correlation of random variables $X$ and $Y$ is the function $\text{Corr}(X,Y): \mathbb{R}^2 \rightarrow [-1,1]$ defined by $\text{Corr}(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$.


# 8. Transformations

**Transformation of a random variable**

Let $X$ be a random variable and let $g: \mathbb{R} \rightarrow \mathbb{R}$ be a function. The random variable $Y = g(X)$ is called a transformation of $X$.

**Expectation of a transformation**

Let $X$ be a random variable and let $g: \mathbb{R} \rightarrow \mathbb{R}$ be a function. The expectation of $Y = g(X)$ is $E(Y) = E(g(X)) = \sum_{x \in \mathbb{R}} g(x) p_X(x)$.

**Convolution**

Let $X$ and $Y$ be random variables. The convolution of $X$ and $Y$ is the random variable $Z = X + Y$.

**Expectation of a convolution**

Let $X$ and $Y$ be random variables. The expectation of $Z = X + Y$ is $E(Z) = E(X + Y) = E(X) + E(Y)$

**Beta distribution** 

In probability theory and statistics, the beta distribution is a family of continuous probability distributions defined on the interval $[0, 1]$ parametrized by two positive shape parameters, denoted by $\alpha$ and $\beta$, that appear as exponents of the random variable and control the shape of the distribution. It is a special case of the Dirichlet distribution.

Mathematically the beta distribution has the form
$$
f(x;\alpha,\beta) = \frac{1}{B(\alpha,\beta)} x^{\alpha-1}(1-x)^{\beta-1}
$$
where the normalisation, $B(\alpha,\beta)$, is the beta function, defined by
$$
B(\alpha,\beta) = \int_0^1 x^{\alpha-1}(1-x)^{\beta-1} \, dx
$$
for $\alpha, \beta > 0$.

**Gamma distribution**

In probability theory and statistics, the gamma distribution is a two-parameter family of continuous probability distributions. The exponential distribution, Erlang distribution, and chi-squared distribution are special cases of the gamma distribution. There are three different parametrizations in common use:

With a shape parameter $k$ and a scale parameter $\theta$.
With a shape parameter $\alpha = k$ and an inverse scale parameter $\beta = 1/\theta$, called a rate parameter.
With a shape parameter $\alpha = k$ and a mean parameter $\mu = k\theta$, sometimes called a scale parameter.
In each of these three forms, both parameters are positive real numbers.

# 9. Conditional expectations

**Conditional expectation given an event**

The conditional expectation of a random variable $X$ given an event $A$ is the function $E(X|A): \mathbb{R} \rightarrow \mathbb{R}$ defined by $E(X|A) = \sum_{x \in \mathbb{R}} x p_{X|A}(x)$.

**Conditional expectation given a random variable**

The conditional expectation of a random variable $X$ given a random variable $Y$ is the function $E(X|Y): \mathbb{R} \rightarrow \mathbb{R}$ defined by $E(X|Y) = \sum_{x \in \mathbb{R}} x p_{X|Y}(x)$.

**Properties of conditional expectations**

**Conditional independence**

Random variables $X$ and $Y$ are conditionally independent given $Z=z$ if $F_{X,Y|Z=z}(x,y) = F_{X|Z=z}(x) F_{Y|Z=z}(y)$ for all $(x,y) \in \mathbb{R}^2$.

**Conditional covariance**

The conditional covariance of random variables $X$ and $Y$ given $Z=z$ is the function $\text{Cov}(X,Y|Z=z): \mathbb{R}^3 \rightarrow \mathbb{R}$ defined by $\text{Cov}(X,Y|Z=z) = E((X-\mu_{X|Z=z})(Y-\mu_{Y|Z=z})|Z=z)$.

**Conditional correlation**

The conditional correlation of random variables $X$ and $Y$ given $Z=z$ is the function $\text{Corr}(X,Y|Z=z): \mathbb{R}^3 \rightarrow [-1,1]$ defined by $\text{Corr}(X,Y|Z=z) = \frac{\text{Cov}(X,Y|Z=z)}{\sigma_{X|Z=z} \sigma_{Y|Z=z}}$.

**Conditional expectation**

The conditional expectation of a random variable $X$ given $Y=y$ is the function $E(X|Y=y): \mathbb{R} \rightarrow \mathbb{R}$ defined by $E(X|Y=y) = \sum_{x \in \mathbb{R}} x p_{X|Y=y}(x)$ for all $y \in \mathbb{R}$.

**Conditional variance**

The conditional variance of a random variable $X$ given $Y=y$ is the function $\text{Var}(X|Y=y): \mathbb{R} \rightarrow \mathbb{R}$ defined by $\text{Var}(X|Y=y) = E((X-E(X|Y=y))^2|Y=y)$ for all $y \in \mathbb{R}$.

# 10. Inequalities and limit theorems

**Inequalities**

**Markov's inequality**

Let $X$ be a non-negative random variable. Then for all $a > 0$, $P(X \geq a) \leq \frac{E(X)}{a}$.

**Chebyshev's inequality**

Let $X$ be a random variable with finite mean $\mu$ and finite variance $\sigma^2$. Then for all $a > 0$, $P(|X-\mu| \geq a) \leq \frac{\sigma^2}{a^2}$.

**Jensen's inequality**

Let $X$ be a random variable and let $g: \mathbb{R} \rightarrow \mathbb{R}$ be a convex function. Then $E(g(X)) \geq g(E(X))$.

**Cauchy-Schwarz inequality**

Let $X$ and $Y$ be random variables. Then $E(|XY|) \leq \sqrt{E(X^2) E(Y^2)}$.

**Hölder's inequality**

Let $X$ and $Y$ be random variables and let $p,q > 1$ be such that $\frac{1}{p} + \frac{1}{q} = 1$. Then $E(|XY|) \leq (E(|X|^p))^{1/p} (E(|Y|^q))^{1/q}$.

**Minkowski's inequality**

Let $X$ and $Y$ be random variables. Then $E(|X+Y|^p)^{1/p} \leq (E(|X|^p))^{1/p} + (E(|Y|^p))^{1/p}$ for all $p \geq 1$.

**Limit theorems**

**Weak law of large numbers**

Let $X_1, X_2, \ldots$ be a sequence of independent and identically distributed random variables with finite mean $\mu$ and finite variance $\sigma^2$. Then $\frac{1}{n} \sum_{i=1}^n X_i \rightarrow \mu$ in probability as $n \rightarrow \infty$.

The weak law of large numbers is also known as Bernoulli's theorem. It is a special case of the strong law of large numbers. It says that the sample mean converges in probability to the population mean.

**Strong law of large numbers**

Let $X_1, X_2, \ldots$ be a sequence of independent and identically distributed random variables with finite mean $\mu$ and finite variance $\sigma^2$. Then $\frac{1}{n} \sum_{i=1}^n X_i \rightarrow \mu$ almost surely as $n \rightarrow \infty$.

The strong law of large numbers is also known as Kolmogorov's theorem. It says that the sample mean converges almost surely to the population mean.

**Central limit theorem**

Let $X_1, X_2, \ldots$ be a sequence of independent and identically distributed random variables with finite mean $\mu$ and finite variance $\sigma^2$. Then $\frac{\sum_{i=1}^n X_i - n \mu}{\sqrt{n \sigma^2}} \rightarrow N(0,1)$ in distribution as $n \rightarrow \infty$.

The central limit theorem says that the sample mean converges in distribution to the standard normal distribution.

**Chi-squared distribution**

In probability theory and statistics, the chi-squared distribution (also chi-square or $\chi^2$-distribution) with $k$ degrees of freedom is the distribution of a sum of the squares of $k$ independent standard normal random variables. It is one of the most widely used probability distributions in inferential statistics, e.g., in hypothesis testing or in construction of confidence intervals. When it is being distinguished from the more general noncentral chi-squared distribution, this distribution is sometimes called the central chi-squared distribution.

**Student's t-distribution**

In probability and statistics, Student's t-distribution (or simply the t-distribution) is any member of a family of continuous probability distributions that arises when estimating the mean of a normally distributed population in situations where the sample size is small and population standard deviation is unknown. It was developed by William Sealy Gosset under the pseudonym Student.

# 11. Markov chains
Markov chains are a special type of stochastic process. They are used to model random processes that evolve over time and have the Markov property, which states that the conditional probability distribution of future states of the process depends only upon the present state, not on the sequence of events that preceded it. In other words, the future is independent of the past given the present.

**Markov chain**

A Markov chain is a stochastic process with the Markov property. It is a sequence of random variables $X_1, X_2, \ldots$ with the property that $P(X_{n+1} = x_{n+1} | X_1 = x_1, \ldots, X_n = x_n) = P(X_{n+1} = x_{n+1} | X_n = x_n)$ for all $n \geq 1$ and all $x_1, \ldots, x_n, x_{n+1} \in \mathbb{R}$.

**Transition matrix**

Let $X_1, X_2, \ldots$ be a Markov chain with state space $\mathcal{S}$. The transition matrix $P$ is defined as $P_{ij} = P(X_{n+1} = j | X_n = i)$ for all $i,j \in \mathcal{S}$. The rows of the transition matrix sum to 1. The transition matrix is also called the stochastic matrix.

**Stationary distribution**

Let $X_1, X_2, \ldots$ be a Markov chain with transition matrix $P$. A probability distribution $\pi$ is called a stationary distribution if $\pi = \pi P$. A Markov chain is called irreducible if there is a single communicating class. A Markov chain is called aperiodic if the greatest common divisor of the lengths of the cycles is 1. If a Markov chain is both irreducible and aperiodic, then it has a unique stationary distribution $\pi$ and $\pi_i = \frac{1}{\mathbb{E}(\tau_i)}$ where $\tau_i$ is the first return time to state $i$. The stationary distribution is also called the equilibrium distribution.

# 12. Havent found room yet 

**F-distribution** (p28)

In probability theory and statistics, the F-distribution, also known as Snedecor's F distribution or the Fisher–Snedecor distribution (after Ronald Fisher and George W. Snedecor) is a continuous probability distribution that arises frequently as the null distribution of a test statistic, most notably in the analysis of variance (ANOVA), e.g., F-test.

The F-distribution is a right-skewed distribution used most commonly in Analysis of Variance. When the F-distribution is used to test the hypothesis that two normal populations have the same mean, it is called the F-test.

It has the form 
$$
f(x;d_1,d_2) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1 + d_2}}}}{x B\left(\frac{d_1}{2},\frac{d_2}{2}\right)}
$$
where $d_1$ and $d_2$ are the degrees of freedom parameters and $B$ is the beta function.


**Conditional variance**

The conditional variance of a random variable $X$ given $Y=y$ is the function $\text{Var}(X|Y=y): \mathbb{R} \rightarrow \mathbb{R}$ defined by $\text{Var}(X|Y=y) = E((X-E(X|Y=y))^2|Y=y)$ for all $y \in \mathbb{R}$.

**Conditional moment generating function**

The conditional moment generating function of a random variable $X$ given $Y=y$ is the function $M_{X|Y=y}: \mathbb{R} \rightarrow [0,1]$ defined by $M_{X|Y=y}(t) = E(e^{tX}|Y=y)$ for all $t \in \mathbb{R}$.

**Conditional characteristic function**

The conditional characteristic function of a random variable $X$ given $Y=y$ is the function $\phi_{X|Y=y}: \mathbb{R} \rightarrow [0,1]$ defined by $\phi_{X|Y=y}(t) = E(e^{itX}|Y=y)$ for all $t \in \mathbb{R}$.

**Conditional joint probability mass function**

The conditional joint probability mass function of discrete random variables $X$ and $Y$ given $Y=y$ is the function $p_{X,Y|Y=y}: \mathbb{R}^2 \rightarrow [0,1]$ defined by $p_{X,Y|Y=y}(x,y) = P(X=x,Y=y|Y=y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}$ for all $(x,y) \in \mathbb{R}^2$.

**Conditional joint cumulative distribution function**

The conditional joint cumulative distribution function of random variables $X$ and $Y$ given $Y=y$ is the function $F_{X,Y|Y=y}: \mathbb{R}^2 \rightarrow [0,1]$ defined by $F_{X,Y|Y=y}(x,y) = P(X \le x,Y \le y|Y=y) = \frac{F_{X,Y}(x,y)}{F_Y(y)}$ for all $(x,y) \in \mathbb{R}^2$.

**Conditional joint probability density function**

The conditional joint probability density function of continuous random variables $X$ and $Y$ given $Y=y$ is the function $f_{X,Y|Y=y}: \mathbb{R}^2 \rightarrow [0,1]$ defined by $f_{X,Y|Y=y}(x,y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$ for all $(x,y) \in \mathbb{R}^2$.
