## Find Nearest Neighbor
We compute the $\textbf{Nearest Neighbor}$ using the following formula for a ray $R \in \mathbb{R}^{rays}$.  

$$
\begin{align*}
    Cloud_O &= (-3.75,-3.75,-3.75) \\
    d(R_{ij}, Cloud_O) &= \lfloor R_{ij} - Cloud_O \rceil_3 
\end{align*}
$$  

In the code below, we observe the values of the closest $\textit{cloud point}$ for ray $R_1$ at first sample.  

![Alt text](image.png)