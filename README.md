# Probabilities
One of the major cornerstones of statistics, machine learning and inference. As such I felt it relevant to revisit my knowledge on this subject. I have gone through most of the book and then revisited the parts I felt were most relevant for myself - and for later use. 

## Probabilities and counting
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

# Conditional probabilities 
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

# Random variables and their distributions
**Random variable** 
Given an experiment with sample space $S$, a random variable is a function $X: S \rightarrow \mathbb{R}$.

**Discrete random variable** 
A random variable $X$ is discrete if it takes on a finite or countably infinite number of values.

**Probability mass function (pmf)**
The probability mass function of a discrete random variable $X$ is the function $p_X: \mathbb{R} \rightarrow [0,1]$ defined by $p_X(x) = P(X=x)$ for all $x \in \mathbb{R}$.

**Cumulative distribution function (cdf)**
The cumulative distribution function of a random variable $X$ is the function $F_X: \mathbb{R} \rightarrow [0,1]$ defined by $F_X(x) = P(X \le x)$ for all $x \in \mathbb{R}$.

**Expectation**
The expectation of a discrete random variable $X$ with pmf $p_X$ is the value $\mu = E(X)$ defined by:
$$E(X) = \sum_{x \in \mathbb{R}} x \cdot p_X(x)$$

**Variance**
The variance of a discrete random variable $X$ with pmf $p_X$ and expectation $\mu = E(X)$ is the value $\sigma^2 = V(X)$ defined by:
$$V(X) = \sum_{x \in \mathbb{R}} (x-\mu)^2 \cdot p_X(x)$$

**Standard deviation**
The standard deviation of a discrete random variable $X$ with pmf $p_X$ and expectation $\mu = E(X)$ is the value $\sigma = \sqrt{V(X)}$.


**Binomial distribution** (p16)

The binomial distribution with parameters $n$ and $p$ is the discrete probability distribution of the number of successes in a sequence of $n$ independent experiments, each asking a yes-no question, and each with its own boolean-valued outcome: success/yes/true/one (with probability $p$) or failure/no/false/zero (with probability $q = 1 - p$). A single success/failure experiment is also called a Bernoulli trial or Bernoulli experiment and a sequence of outcomes is called a Bernoulli process; for a single trial, i.e., $n = 1$, the binomial distribution is a Bernoulli distribution. The binomial distribution is the basis for the popular binomial test of statistical significance.

**Bernoulli distribution** (p16)

The Bernoulli distribution is the probability distribution of a random variable which takes the value 1 with probability $p$ and the value 0 with probability $q = 1 - p$. It can be used to represent a coin toss where 1 and 0 would represent "head" and "tail" (or vice versa), respectively, and $p$ would be the probability of the coin landing on heads or tails, respectively. In particular, unfair coins would have $p \neq 0.5$.

**Geometric distribution** (p17)

The geometric distribution is either of two discrete probability distributions:

The probability distribution of the number $X$ of Bernoulli trials needed to get one success, supported on the set $\{ 1, 2, 3, \ldots \}$
The probability distribution of the number $Y = X − 1$ of failures before the first success, supported on the set $\{ 0, 1, 2, 3, \ldots \}$
Which of these one calls "the" geometric distribution is a matter of convention and convenience.


**Hypergeometric distribution** (p19)

In probability theory and statistics, the hypergeometric distribution is a discrete probability distribution that describes the probability of $k$ successes (random draws for which the object drawn has a specified feature) in $n$ draws, without replacement, from a finite population of size $N$ that contains exactly $K$ objects with that feature, wherein each draw is either a success or a failure. In contrast, the binomial distribution describes the probability of $k$ successes in $n$ draws with replacement.

**Negative binomial distribution** (p20)

In probability theory and statistics, the negative binomial distribution is a discrete probability distribution that models the number of successes in a sequence of independent and identically distributed Bernoulli trials before a specified (non-random) number of failures (denoted $r$) occurs. For example, if we define a 1 as failure, all non-1 values as successes, and we throw a die repeatedly until the third time 1 appears (r = 3), then the probability distribution of the number of non-1 values that appear before the third 1 is a negative binomial distribution.

**Gamma distribution** (p24)

In probability theory and statistics, the gamma distribution is a two-parameter family of continuous probability distributions. The exponential distribution, Erlang distribution, and chi-squared distribution are special cases of the gamma distribution. There are three different parametrizations in common use:

With a shape parameter $k$ and a scale parameter $\theta$.
With a shape parameter $\alpha = k$ and an inverse scale parameter $\beta = 1/\theta$, called a rate parameter.
With a shape parameter $\alpha = k$ and a mean parameter $\mu = k\theta$, sometimes called a scale parameter.
In each of these three forms, both parameters are positive real numbers.

**Beta distribution** (p25)

In probability theory and statistics, the beta distribution is a family of continuous probability distributions defined on the interval $[0, 1]$ parametrized by two positive shape parameters, denoted by $\alpha$ and $\beta$, that appear as exponents of the random variable and control the shape of the distribution. It is a special case of the Dirichlet distribution.

**Chi-squared distribution** (p26)

In probability theory and statistics, the chi-squared distribution (also chi-square or $\chi^2$-distribution) with $k$ degrees of freedom is the distribution of a sum of the squares of $k$ independent standard normal random variables. It is one of the most widely used probability distributions in inferential statistics, e.g., in hypothesis testing or in construction of confidence intervals. When it is being distinguished from the more general noncentral chi-squared distribution, this distribution is sometimes called the central chi-squared distribution.

**Student's t-distribution** (p27)

In probability and statistics, Student's t-distribution (or simply the t-distribution) is any member of a family of continuous probability distributions that arises when estimating the mean of a normally distributed population in situations where the sample size is small and population standard deviation is unknown. It was developed by William Sealy Gosset under the pseudonym Student.

**F-distribution** (p28)

In probability theory and statistics, the F-distribution, also known as Snedecor's F distribution or the Fisher–Snedecor distribution (after Ronald Fisher and George W. Snedecor) is a continuous probability distribution that arises frequently as the null distribution of a test statistic, most notably in the analysis of variance (ANOVA), e.g., F-test.


**Bernoulli distribution**
A random variable $X$ has a Bernoulli distribution with parameter $p$ if $X$ takes on values in $\{0,1\}$ with $P(X=1) = p$ and $P(X=0) = 1-p$.

**Binomial distribution**
In probability theory and statistics, the binomial distribution with parameters $n$ and $p$ is the discrete probability distribution of the number of successes in a sequence of $n$ independent yes/no experiments, each of which yields success with probability $p$. A success/failure experiment is also called a Bernoulli experiment or Bernoulli trial; when $n = 1$, the binomial distribution is a Bernoulli distribution. The binomial distribution is the basis for the popular binomial test of statistical significance.

A random variable $X$ has a binomial distribution with parameters $n$ and $p$ if $X$ takes on values in $\{0,1,\ldots,n\}$ with $P(X=k) = {n \choose k} p^k (1-p)^{n-k}$ for $k=0,1,\ldots,n$.

**Geometric distribution**
A random variable $X$ has a geometric distribution with parameter $p$ if $X$ takes on values in $\{1,2,\ldots\}$ with $P(X=k) = p(1-p)^{k-1}$ for $k=1,2,\ldots$.

**Poisson distribution**
The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event. The Poisson distribution can also be used for the number of events in other specified intervals such as distance, area or volume.

A random variable $X$ has a Poisson distribution with parameter $\lambda$ if $X$ takes on values in $\{0,1,\ldots\}$ with $P(X=k) = \frac{\lambda^k}{k!}e^{-\lambda}$ for $k=0,1,\ldots$.

**Uniform distribution**

In probability theory and statistics, the continuous uniform distribution or rectangular distribution is a family of symmetric probability distributions such that for each member of the family, all intervals of the same length on the distribution's support are equally probable. The support is defined by the two parameters, a and b, which are its minimum and maximum values. The distribution is often abbreviated U(a,b). It is the maximum entropy probability distribution for a random variate X under no constraint other than that it is contained in the distribution's support.

A random variable $X$ has a uniform distribution on the interval $[a,b]$ if $X$ takes on values in $[a,b]$ with $P(X=x) = \frac{1}{b-a}$ for $x \in [a,b]$.

**Exponential distribution**
The exponential distribution (also known as negative exponential distribution) is the probability distribution that describes the time between events in a Poisson point process, i.e., a process in which events occur continuously and independently at a constant average rate. It is a particular case of the gamma distribution. It is the continuous analogue of the geometric distribution, and it has the key property of being memoryless. In addition to being used for the analysis of Poisson point processes it is found in various other contexts.

A random variable $X$ has an exponential distribution with parameter $\lambda$ if $X$ takes on values in $[0,\infty)$ with $P(X=x) = \lambda e^{-\lambda x}$ for $x \in [0,\infty)$.

**Normal distribution**
In probability theory, the normal (or Gaussian) distribution is a very common continuous probability distribution. Normal distributions are important in statistics and are often used in the natural and social sciences to represent real-valued random variables whose distributions are not known. A random variable with a Gaussian distribution is said to be normally distributed and is called a normal deviate.

A random variable $X$ has a normal distribution with parameters $\mu$ and $\sigma$ if $X$ takes on values in $(-\infty,\infty)$ with $P(X=x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ for $x \in (-\infty,\infty)$.

**Moment generating function**
The moment generating function of a random variable $X$ is the function $M_X: \mathbb{R} \rightarrow [0,1]$ defined by $M_X(t) = E(e^{tX})$ for all $t \in \mathbb{R}$.

**Central limit theorem**
The central limit theorem states that the distribution of the average of a large number of independent, identically distributed variables will be approximately normal, regardless of the underlying distribution.



Let $X_1,X_2,\ldots$ be a sequence of independent and identically distributed random variables with mean $\mu$ and variance $\sigma^2$. Then the random variable $Z_n = \frac{X_1 + \cdots + X_n - n\mu}{\sigma \sqrt{n}}$ converges in distribution to a standard normal random variable as $n \rightarrow \infty$.

# Joint distributions
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

**Conditional expectation**
The conditional expectation of a random variable $X$ given $Y=y$ is the function $E(X|Y=y): \mathbb{R} \rightarrow \mathbb{R}$ defined by $E(X|Y=y) = \sum_{x \in \mathbb{R}} x p_{X|Y=y}(x)$ for all $y \in \mathbb{R}$.

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

**Conditional moment generating function**
The conditional moment generating function of a random variable $X$ given $Y=y$ is the function $M_{X|Y=y}: \mathbb{R} \rightarrow [0,1]$ defined by $M_{X|Y=y}(t) = E(e^{tX}|Y=y)$ for all $t \in \mathbb{R}$.

**Conditional characteristic function**
The conditional characteristic function of a random variable $X$ given $Y=y$ is the function $\phi_{X|Y=y}: \mathbb{R} \rightarrow [0,1]$ defined by $\phi_{X|Y=y}(t) = E(e^{itX}|Y=y)$ for all $t \in \mathbb{R}$.

# Expectations

