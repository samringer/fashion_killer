## Pose Detector
    * Add hands, detailed face etc.
    * Investigate changing contrast etc of input at inference to
      boost accuracy.

## Asos
    * Ensure the correct pose image (in terms of RGB) is being used.
    * Download all the data.
    * Clean properly.
    * Check if adding more of a black border allows for better
        decoding of knees, feet and hands.
    * Fix issue of bad detection on white clothing on white background
    * Fix issue of not detecting arms when in same material as body and
        overlapping body.
    * Investigate use of attention layer.
    * Investigate discriminative loss.
    * Investigate using a full CGAN.

## V U Net

CHECK IF I CAN REMOVE INPUTTING THE INPUT POSE OF THE DESIRED APPEARANCE
IMAGE INTO THE ENCODER, would make things simpler

Laplace of output

(Also look at how sampling is done in a standard VAE)
Think about best croppings to feed into appearance encoder
THe wierd laplacian thing
Batch Norm
Bigger and cleaner data
Fine tuning of VGG
Inception Score monitoring
dropout
augmentation

Possibly formulate as a metalearning problem to encorporate multiple images of the same person

Better hands and face using: https://github.com/CMU-Perceptual-Computing-Lab/openpose
