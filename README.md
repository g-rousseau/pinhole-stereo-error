# pinhole-stereo-error
Basic simulator of a pinhole stereo camera pair errors

The two cameras are supposed coplanar and aligned along the baseline (such that the epipolar curves are straight horizontal lines).

* Given the horizontal field of view, baseline and definition of the cameras, compute the depth measurement accuracy of the stereo pair

![alt text](https://github.com/g-rousseau/pinhole-stereo-error/blob/main/example_1.png)

* A stereo matching accuracy and a disparity uncertainty can be added into the model

![alt text](https://github.com/g-rousseau/pinhole-stereo-error/blob/main/example_2.png)
