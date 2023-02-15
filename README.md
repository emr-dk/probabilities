# probabilities
Various assignments from "Introduction to Probability" af Joseph Blitzstein og Jessica Hwang

# Probabilities and counting
### Naive definition of probability (p6)
$$
P_{naive}(A) = \frac{\lvert A \rvert}{\lvert S \rvert} = \frac{\text{No. of outcomes in favor of} A}{\text{total number of outcomes in }S}
$$

### Multiplication rule (P8)
If we have a compound experiment consisting of Experiment A with $a$ outcomes and Experiment B with $b$ outcomes, the possible outcomes will be $a \cdot b = b \cdot a$ 

### Sampling with replacement (p11)
Making $k$ choices from $n$ objects has $n^k$ outcomes

### Sampling without replacement (p12)
Making $k$ choices from $n$ has 
$n(n-1) \cdots (n-k+1), 1 \le k \le n$ outcomes

### Binomials
For any integer where $k \ge 0$ and $n \ge 0$ the binomial coefficient is the number of subsets of size $k$ for $S$ with size $n$. Written as ${n\choose k}$ and read as $n$ choose $k$.

Where $k \le n$ the binomial coefficient formula is written as 
$${n\choose k} = \frac{n(n-1)\cdots(n-k+1)}{k!}=\frac{n!}{(n-k)!k!}$$

# Conditional probabilities 
### Conditional probability (p46)
If $A$ and $B$ are events and $P(B) > 0$ then the conditional probability of $A$ given $B$ is:
$$P(A|B) = \frac{P(A\cap B)}{P(B)}$$

### Probability of the intersection of two events (p52)
For any events A and B with positive probabilities:
$$P(A \cap B) = P(B)P(A|B) = P(A)P(B|A)$$

### Bayes rule (p53)
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

### Odds (p53)
The odds of A are:
$$\odds(A) = \frac{P(A)}{P(A)^C}$$

### Law of total probability (p54)
Let $A_1,\ldots,A_n$ be a partition of sample space $S$ - the events $A_i$ are disjoint and their union is $S$), then:
$$P(B) = \sum_{i=1}^n P(B|A_i)P(A_i)$$

### Inclusion-exclusion (p59)
$$P(A \cup B |E) = P(A|E) + P(B|E) - P(A \cap B|E)$$

### Independence of two events
$$P(A \cap B) = P(A)P(B)$$

# Random variables and their distributions

# Expectations