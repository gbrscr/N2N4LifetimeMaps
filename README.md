# Noise2Noise Image Reconstruction of Lifetime Maps in Halide Perovskite Thin Films

Code for the paper "Noise2Noise Image Reconstruction of Lifetime Maps in Halide Perovskite Thin Films" published at EUSIPCO 2025.

**Authors:** [Gabriele Scrivanti](https://gbrscr.github.io/)¹, [Luca Calatroni](https://sites.google.com/view/lucacalatroni/home)², [Stefania Cacovich]¹

¹ IPVF, UMR 9006, CNRS, École Polytechnique, IP Paris, Chimie ParisTech, PSL, Palaiseau, France  
² MaLGa Center, DIBRIS, Università di Genova & MMS, Istituto Italiano di Tecnologia, Genoa, Italy
[Paper](https://eusipco2025.org/wp-content/uploads/pdfs/0001707.pdf)
**Abstract:**

We present an unsupervised deep-learning approach for lifetime map reconstruction from noisy time-resolved fluorescence imaging (TR-FLIM) datasets. In the context of semiconductor and photovoltaic device characterisation, this method is critical for accurately predicting solar cell performance and detecting early signs of degradation. More precisely, we consider an unsupervised Noise2Noise (N2N) training framework combined with physics-driven modelling for the quantitative reconstruction of lifetime maps. The proposed approach incorporates a loglinear fit in the N2N loss function and parameterises the unknown maps as outputs of a shallow neural network with a multibranch architecture. By learning from multiple noisy acquisitions of the same scene, our method effectively allows an accurate estimation with shorter acquisition protocols, which translates into a lower risk of damage for the sample under consideration. Tests on simulated data and comparisons with available modelbased approaches show that the proposed approach improves robustness w.r.t. noise levels with limited tuning of the regularisation/algorithmic parameters.
**Index Terms** Quantitative image reconstruction, Noise2Noise, perovskite cell characterisation.

![image info](N2N.png)

Details coming soon!

