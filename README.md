**Auto-encoders** serve as versatile tools in **deep learning** due to their ability to learn efficient representations of data. Their primary motivation lies in **dimensionality reduction** and **feature learning**, allowing for the extraction of meaningful features from raw data. By encoding input data into a compressed representation and subsequently reconstructing it, auto-encoders aid in tasks like **denoising**, **anomaly detection**, and **generative modeling**. Their self-supervised nature, where they learn from unlabeled data, makes them valuable in scenarios where labeled data is scarce or costly. Moreover, their adaptability across various domains, from image and text data to more complex sequential data, underscores their significance in modern deep learning architectures.

## Auto-Encoder Fundamentals

An auto-encoder comprises two primary components: the encoder, represented as  $f(\phi)$ , and the decoder, denoted as $g(\theta)$ . The encoder function  $f(\phi)$  compresses the input data  $x$  into a lower-dimensional latent space representation called the bottleneck, denoted as  $z$ . This encoding process aims to capture the most essential features of the input. Subsequently, the decoder function  $g(\theta)$  reconstructs the original input from this bottleneck representation. The auto-encoder's training involves minimizing a loss function that measures the discrepancy between the input and the reconstructed output.

The **reconstruction loss** can be computed using various metrics, commonly employing the **mean squared error (MSE)** for continuous data or **binary cross-entropy** for binary data. For instance, the MSE loss function for a set of $N$ training samples can be expressed as:

$$
\begin{equation}
L(\phi, \theta) = \frac{1}{N} \sum_{i=1}^{N} || x^{(i)} - g(\theta)(f(\phi)(x^{(i)})) ||^2
\end{equation}
$$

Here,  $x^{(i)}$  represents the  $i$ th input sample,  $f(\phi)(x^{(i)})$  is the encoding of  $x^{(i)}$  to the bottleneck  $z^{(i)}$ , and  $g(\theta)(z^{(i)})$  is the reconstruction of  $x^{(i)}$  from the bottleneck representation. The goal during training is to adjust the encoder and decoder parameters  $\phi$  and  $\theta$  to minimize this loss, thereby improving the accuracy of the reconstructed output compared to the original input.

![Illustration of autoencoder model architecture. Source: [https://lilianweng.github.io/posts/2018-08-12-vae/](https://lilianweng.github.io/posts/2018-08-12-vae/)](images/Untitled.png)

Illustration of autoencoder model architecture. Source: [https://lilianweng.github.io/posts/2018-08-12-vae/](https://lilianweng.github.io/posts/2018-08-12-vae/)

The **encoder network** serves a purpose akin to dimensionality reduction methods like Principal Component Analysis (PCA) or Matrix Factorization (MF). It condenses the input data into a compressed or encoded representation, reducing its dimensionality. This reduction helps in capturing the most relevant and **essential features of the data while discarding less critical information**. This process is crucial as it simplifies the data representation, making it more manageable and efficient for subsequent analysis or processing.

Moreover, the **autoencoder** is specifically designed to optimize the reconstruction of data from this compressed representation. It learns to reconstruct the original input from the encoded representation. This emphasis on accurate reconstruction encourages the encoder to create a meaningful and informative intermediate representation. This representation not only captures latent variables or underlying structures within the data but also facilitates a more effective and accurate process of decompression or decoding. In essence, a well-trained encoder within an auto-encoder framework results in an intermediate representation that not only efficiently captures important features but also greatly aids in the faithful reconstruction of the original data from this compressed form.

### **Denoising Autoencoder**

In a **denoising autoencoder**, the primary objective is to reconstruct clean data from corrupted or noisy input. This approach aids in learning robust and meaningful representations by forcing the model to capture essential features while filtering out noise.

The **need for denoising** arises from the desire to make the learned representation more robust against noise present in real-world data. By training on corrupted input and expecting the model to recover the original, clean data, the denoising autoencoder becomes adept at extracting salient features while filtering out irrelevant or noisy information.

![Illustration of denoising autoencoder model architecture. Source: [https://lilianweng.github.io/posts/2018-08-12-vae/](https://lilianweng.github.io/posts/2018-08-12-vae/)](images/Untitled%201.png)

Illustration of denoising autoencoder model architecture. Source: [https://lilianweng.github.io/posts/2018-08-12-vae/](https://lilianweng.github.io/posts/2018-08-12-vae/)

The modification in the loss function involves comparing the reconstructed output to the original, uncorrupted input. Assuming an additive random noise  $\epsilon$  is applied to the input  $x$  to generate the corrupted input  $x_{\text{corrupted}}$ , the denoising autoencoder aims to minimize the reconstruction error between the reconstructed output and the original, uncorrupted input.

The adjusted loss function using mean squared error (MSE) for a set of $N$ training samples can be expressed as:

$$
\begin{equation}
L(\phi, \theta) = \frac{1}{N} \sum_{i=1}^{N} || x^{(i)} - g(\theta)(f(\phi)(x_{\text{corrupted}}^{(i)})) ||^2
\end{equation}
$$

To illustrate how noise is added to the input, assuming Gaussian noise with mean  $\mu$  and standard deviation  $\sigma$  is added to each element of the input  $x$  to generate the corrupted input  $x_{\text{corrupted}}$ , it can be represented as:

$$
\begin{equation}
x_{\text{corrupted}} = x + \epsilon, \quad \text{where} \quad \epsilon \sim \mathcal{N}(\mu, \sigma^2)
\end{equation}
$$

Here,  $x^{(i)}$  denotes the  $i$ th input sample,  $f(\phi)(x_{\text{corrupted}}^{(i)})$  represents the encoding of the corrupted input to the bottleneck  $z^{(i)}$ , and  $g(\theta)(z^{(i)})$  signifies the reconstruction of  $x^{(i)}$  from the bottleneck representation. The denoising autoencoder, by learning to reconstruct clean data from noisy inputs, encourages the model to capture robust and meaningful features while enhancing its ability to filter out unwanted noise in the data.

### Sparse Auto-Encoder

Sparse auto-encoders aim to introduce **sparsity in the learned representations**, meaning that only a **few units in the network are activated at a time**. This sparsity constraint forces the model to learn more efficient and selective representations, focusing on the most important features in the data. The sparsity constraint encourages the autoencoder to use only a limited number of neurons in the encoding process, which can lead to better generalization, reduced overfitting, and improved interpretability of the learned features.

In a sparse autoencoder, the loss function is modified to include a **sparsity term**. The overall loss function now comprises two components: the reconstruction loss and the sparsity regularization term. The sparsity term encourages the activation of only a small fraction $\rho$ of units in the hidden layer.

The loss function for a sparse autoencoder with $N$ training samples can be expressed as follows:

$$
\begin{equation}
L(\phi, \theta) = \frac{1}{N} \sum_{i=1}^{N} || x^{(i)} - g(\theta)(f(\phi)(x^{(i)})) ||^2 + \lambda \sum_{j=1}^{K} \text{KL}(\rho || \hat{\rho}_j)
\end{equation}
$$

Where:

- $x^{(i)}$  represents the  $i$ th input sample.
- $f(\phi)(x^{(i)})$  denotes the encoding of  $x^{(i)}$  to the bottleneck.
- $g(\theta)(z^{(i)})$  represents the reconstruction of  $x^{(i)}$  from the bottleneck representation.
- $K$  is the number of units in the hidden layer.
- $\lambda$  is the regularization parameter controlling the impact of the sparsity term.
- $\text{KL}$  denotes the Kullback-Leibler divergence, measuring the difference between the desired sparsity  $\rho$  and the actual average activation  $\hat{\rho}_j$  of each neuron in the hidden layer.

The sparsity term  $\text{KL}(\rho || \hat{\rho}_j)$  can be defined as:

$$
\begin{equation}
\text{KL}(\rho || \hat{\rho}_j) = \rho \log \left( \frac{\rho}{\hat{\rho}_j} \right) + (1 - \rho) \log \left( \frac{1 - \rho}{1 - \hat{\rho}_j} \right)
\end{equation}
$$

Here,  $\rho$  represents the desired sparsity level, and  $\hat{\rho}_j$  is the average activation of neuron  $j$  in the hidden layer over the training set. Let say, there are $s_l$ neurons in the $l$-th layer and the activation function of $j$-th neuron in this layer is $a_j^{(l)}(.)$ where $j=1,..s_l$, $\hat{\rho}_j^{(l)}$ can be calculated as:

$$
\begin{equation}
\hat{\rho}_j^{(l)} = \frac{1}{n} \sum_{i=1}^n [a_j^{(l)}(\mathbf{x}^{(i)})] \approx \rho 
\end{equation}
$$

By incorporating the sparsity term in the loss function, the sparse autoencoder encourages the network to learn sparse and informative representations, promoting more efficient and selective encoding of input data.

## **VAE: Variational Autoencoder**

Variational Autoencoders (VAEs) address the limitations of traditional autoencoders by enabling the generation of new data samples. They aim to learn a latent space representation that not only captures meaningful features but also allows for the generation of new, realistic data points by sampling from the learned distribution. VAEs achieve this by imposing a specific structure on the latent space, making it follow a probabilistic distribution.

In a VAE, the goal is to **learn a probability distribution** over the latent space, which is typically assumed to follow a Gaussian distribution. The model learns to **encode input data into a probability distribution**, and during the decoding process, it generates new samples by sampling from this distribution.

![Variational Autoencoder. Source: [https://lilianweng.github.io/posts/2018-08-12-vae/](https://lilianweng.github.io/posts/2018-08-12-vae/)](images/Untitled%202.png)

Variational Autoencoder. Source: [https://lilianweng.github.io/posts/2018-08-12-vae/](https://lilianweng.github.io/posts/2018-08-12-vae/)

The VAE's loss function comprises two components: a reconstruction loss similar to traditional autoencoders and a regularization term that enforces the latent space to follow a specific distribution (usually a standard Gaussian distribution).

The loss function for a VAE with $N$ training samples can be expressed as follows:

$$
\begin{equation}
L(\phi, \theta) = -\frac{1}{N} \sum_{i=1}^{N} \mathbb{E}{z \sim q\phi(z|x^{(i)})} \left[ \log p_\theta(x^{(i)}|z) \right] + \text{KL}(q_\phi(z|x^{(i)}) || p(z))
\end{equation}
$$

Where

- $x^{(i)}$  represents the  $i$ th input sample.
- $q_\phi(z|x^{(i)})$  denotes the approximate posterior distribution (encoder) that maps input  $x^{(i)}$  to a distribution over the latent space  $z$ .
- $p_\theta(x^{(i)}|z)$  represents the likelihood of generating  $x^{(i)}$  from the latent variable  $z$  (decoder).
- $p(z)$  is the prior distribution assumed over the latent space, often chosen as a standard Gaussian distribution.
- $\text{KL}$  denotes the Kullback-Leibler divergence, measuring the difference between the approximate posterior  $q_\phi(z|x^{(i)})$  and the prior  $p(z)$ .

The first term in the loss function is the reconstruction loss, ensuring the fidelity of the generated output to the original input. The second term is the KL divergence, encouraging the distribution of latent variables to approximate the chosen prior distribution. This regularization term helps in shaping the latent space to be continuous and smooth, facilitating the generation of new data samples by sampling from this learned distribution.

> VAEs enable both effective representation learning and generative capabilities, allowing for the creation of novel data points from the learned latent space distribution.

### **VAE vs Generative Arversarial Networks (GAN)?**

Imagine you are an artist who wants to draw a picture of a cat. The VAE (Variational Autoencoder) would help you draw a cat by giving you a set of rules to follow. It tells you what a cat looks like in general, and you try to draw a cat based on those rules. Sometimes, the cat you draw may not look exactly like a real cat, but it's close.

On the other hand, the GAN (Generative Adversarial Network) works differently. It's like having two artists competing against each other. One artist, called the generator, tries to draw a cat, and the other artist, called the discriminator, looks at the drawing and tries to tell if it's a real cat or not. The generator keeps improving its drawings to fool the discriminator, and the discriminator keeps getting better at spotting fake cats. This competition makes the generator really good at drawing cats that look very realistic.

![Simple Architecture of a GAN. Source: [https://www.clickworker.com/ai-glossary/generative-adversarial-networks/](https://www.clickworker.com/ai-glossary/generative-adversarial-networks/)](images/Untitled%203.png)

Simple Architecture of a GAN. Source: [https://www.clickworker.com/ai-glossary/generative-adversarial-networks/](https://www.clickworker.com/ai-glossary/generative-adversarial-networks/)

So, the main difference is that the VAE gives you rules to follow to draw something, while the GAN learns by competition to create things that look very real. Both methods are used for creating new, realistic-looking images, but they work in different ways.