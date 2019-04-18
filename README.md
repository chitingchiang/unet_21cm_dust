# Predicting CMB dust foreground using galactic 21-cm data with U-Net

## Introduction

The cosmic microwave background (CMB) is one of the most powerful astronomical probes to study the content and evolution of our universe. To extract the cosmological information from the measurement, it is crucial to remove the foreground contamination so that the analysis can be free of bias. The two most important foregrounds of CMB are synchrotron and dust.

In this [project](https://arxiv.or/abs/1904.xxxxx), we adopt the deep neural network to predict the microwave dust foreground in our galaxy using the galactic 21-cm data. Since the galactic 21-cm emission traces the neutral hydrogen in our galaxy and various components of the Milky Way (stars, dust, neutral and ionized hydrogen, etc.) trace each other, it is reasonable to expect that the galactic 21-cm data is correlated with the dust foreground.

## Data

For our target dust maps, we use the publicly available all-sky dust map produced by the [Planck Satellite](https://www.esa.int/Our_Activities/Space_Science/Planck) team using component separation. For our 21-cm data we take the full-sky measurements of the 21-cm emission from our galaxy as measured by [HI4PI survey](https://arxiv.org/abs/1610.06175). Since both data sets are measured in spherical coordinates (intensity at different angular positions on the sky), we first use [HEALPix](https://healpix.sourceforge.net) to cut out square images of random centers and rotations with side length of 24.96 degrees and resolution of 64x64 pixels. For the target dust map, there is only one channel; for the 21-cm data, we use 50 velocity slices ranging from -32.25 km/s to 32.25 km/s.

After the data processing, for each set we have the target dust map of shape (64,64,1) and the 21-cm galactic data of shape (64,64,50). All maps in one set have the same cutout and [mask](https://arxiv.org/abs/1801.04945) (to remove bright point sources). We generate 50000 sets of maps from the northern galactic hemisphere for training the neural network and 1000 sets of maps from the southern galactic hemisphere for measuring the performance of the model. To ensure no overlap between the training and test data sets, we do not use projection centers with declination below 17.65 degrees.

## Model

![unet_structure](assets/unet_structure.png)

Our neural network structure is based on the [U-Net](https://arxiv.org/abs/1505.04597) model. U-Net is an image-to- image network, which takes images as input and also outputs images. U-Net was designed to perform image segmentation tasks. The task in this project is similar to image segmentation, i.e. both generating pixel-to-pixel maps. Since this model has no fully connected layer, it can fit different input image sizes using the same model.

## Result

![map animation](assets/maps_animation.gif)
