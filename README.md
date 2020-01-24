# physics-freeform

This is a freeform optics simulation code used for one of my physics projects. 

The actual refraction calculation is based on the paper: Wojtanowski, J. (2018). Efficient numerical method of freeform lens design for arbitrary irradiance shaping. Journal of Modern Optics, 65(9), 1019-1032.

The idea is to take an arbitrary surface (the example here is in the data/z_profile, a pytorch tensor), add some gaussian noise with a pre-defined variance (data/noise_screen*) and calculate the beam refraction through it. The refractive index is taken to be 1.5, the formulas are closely following the ones described in the paper.
