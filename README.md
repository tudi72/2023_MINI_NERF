## 1. Find Nearest Neighbor
We compute the $\textbf{Nearest Neighbor}$ using the following formula for a ray $R \in \mathbb{R}^{rays}$.  

$$
\begin{align*}
    Cloud_O &= (-3.75,-3.75,-3.75) \\
    d(R_{ij}, Cloud_O) &= \lfloor R_{ij} - Cloud_O \rceil_3 \\ 
    index_{ijk} &= \frac{d(R_{ij}, Cloud_O)}{STEP} \\
    index &= index_i \cdot 200 + index_j \cdot 200 \cdot 200 + index_k 
\end{align*}
$$  

In the code below, we observe the values of the closest $\textit{cloud point}$ for ray $R_1$ at first sample.  

![Alt text](image.png)


## 2. Raw to output 
It's the rendering function that outputs the color for a predicted density $\sigma_i$ as well as RGB values $c_i$. For context, $z_n$ is the sampling and $\overrightarrow{ray_i}$ is the direction of the ray.  

$$
\begin{align*}
    z_{vals} &\in \mathbb{R}^1 \quad \text{and} \quad |z_{vals}| = 200\\
    ray &\in \mathbb{R}^3 \quad \text{and} \quad |ray| = 1024 \\ 
    t_i &= z_n * \|\|\overrightarrow{ray_i}\|\| \\
    \delta_i &= t_{i+1} - t_i \quad \text{and} \quad \delta_n = t_n 
\end{align*}
$$

In the code below, we can observe how the step function $t_n$ is created and then multiplied with the ray's direction as a magnitude.

![Alt text](image-1.png)

## 3. Regularization 

We compute the regularization of the color and the depth sigma by substracting a neighbor. The neighbor chosen si the immediate $k+1$ next element from the **point cloud indices**.   

$$
\begin{align*}
    L_{reg colour} &= \sum_{i,j,k}{|c[i,j,k]-c[i,j,k+1]|^2} \\    
    L_{reg \sigma} &= \sum_{i,j,k}{|\sigma[i,j,k]-\sigma[i,j,k+1]|^2}
\end{align*}
$$

![Alt text](image-2.png)