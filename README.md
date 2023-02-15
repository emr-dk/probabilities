# probabilities
Various assignments from "Introduction to Probability" af Joseph Blitzstein og Jessica Hwang

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

# Binomials
For any integer where $k \ge 0$ and $n \ge 0$ the binomial coefficient is the number of subsets of size $k$ for $S$ with size $n$. Written as $begin{pmatrix}n \\ k \end{pmatrix}$ and read as $n$ choose $k$.

Where $k \le n$ the binomial coefficient formula is written as 
$
\begin{pmatrix}n \\ k \end{pmatrix} = \frac{n(n-1)\cdots(n-k+1)}{k!}=\frac{n!}{(n-k)!k!}
$
