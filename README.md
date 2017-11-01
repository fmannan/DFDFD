# Discriminative Filters for Depth from Defocus

Depth from defocus (DFD) requires estimating the depth dependent defocus blur at every pixel. Several approaches for accomplishing this have been proposed over the years. For a pair of images this is done by modeling the defocus relationship between the two differently defocused images and for single defocused images by relying on the the properties of the point spread function and the characteristics of the latent sharp image. We propose depth discriminative filters for DFD that can represent many of the widely used models such as the relative blur, Blur Equalization Technique, deconvolution based depth estimation, and subspace projection methods. We show that by optimizing the parameters of this general model we can obtain state-of-the-art result on synthetic and real defocused images with single or multiple defocused images with different apertures.


This is a version of the code used in the following publication:

```
F Mannan, M S Langer, Discriminative Filters for Depth from Defocus, 3D Vision (3DV), 2016.
```

If you use the code in your work then please cite:

```
@INPROCEEDINGS{MannanLanger3DV16, 
author={F. Mannan and M. S. Langer}, 
booktitle={2016 Fourth International Conference on 3D Vision (3DV)},
title={Discriminative Filters for Depth from Defocus}, 
year={2016}, 
pages={592-600}, 
keywords={deconvolution;image filtering;image restoration;optical transfer function;DFD;blur equalization;deconvolution based depth estimation;defocused images;depth dependent defocus blur estimation;depth discriminative filters;depth from defocus;latent sharp image;point spread function;relative blur;subspace projection;Apertures;Cameras;Convolution;Deconvolution;Estimation;Image reconstruction;Optics;Blur Equalization Technique;Deconvolution;Depth from Defocus;Discriminative Filters;Relative Blur;Subspace Projection}, 
doi={10.1109/3DV.2016.67}, 
month={Oct},}
```

