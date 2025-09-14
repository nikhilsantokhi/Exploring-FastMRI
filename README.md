# Exploring-FastMRI
Exploring fastMRI and accelerated MRI techniques.

# Introduction
Magnetic Resonance Imaging (MRI) is one of the most important medical imaging tools, but long scan times remain a major limitation. Accelerating MRI by undersampling k-space can make scans faster, but it introduces aliasing and blurring in the images that degrade quality. This tutorial walks through how deep learning can combats these challenges, starting with analysis of U-Net as a baseline model and moving to physics-based methods like the End-to-End Variational Network (E2E VarNet) and the more recent Feature-Image VarNet. 

Other key ideas such as coil sensitivity estimation, cascaded refinement, feature-space processing, and block-wise attention are explained. The aim is to give readers an accessible but technically detailed understanding of how modern reconstruction models work, why they are effective, and how they are being validated for clinical use.

# Structure Overview
## Section 1: Introduction
Introduces the motivation for accelerated MRI, explains why long scan times are a problem, and outlines how the MRI scanners work and how MR images are acquired. Also gives background on classical reconstruction methods like parallel imaging and compressed sensing.

## Section 2: fastMRI 
Describes the fastMRI knee dataset, and sets the stage for understanding how reconstruction models process this data. Details of utility functions in the original fastMRI GitHub repo, and explains the design pattern for the upcoming deep learning models using PyTorch and PyTorch Lightning.

## Section 3: K-Space and Masking
Explains important details about k-space, how they are undersampled and the role of acceleration factors in undersamlping. Shows how missing lines in k-space cause predictable aliasing patterns in image space and how different masks are applied to simulate accelerated acquisitions.

## Section 4: Image Space
Breaks down the image construction process starting from raw k-space data to the final image, covering Fourier transforms and the root-sum-of-squares (RSS) transform, alongside the corresponding code implementation of this process.

## Section 5: Models
In depth analysis of the progression of three deep learning models for MRI reconstruction, providing detailed step-by-step explanations of the neural networks. U-Net as a simple supervised baseline. E2E VarNet with cascaded refinement, sensitivity estimation, and physics-based data consistency. Feature-Image VarNet extending E2E-VarNet with feature-space cascades and block-wise attention.

## Section 6: Conclusions 
Summary of the strengths and trade-offs of U-Net, E2E VarNet, and FI VarNet, while highlighting leaderboard performances on the fastMRI leaderboard with the mention of private models. Also mentions emerging methods such as diffusion models and GANs as promising future directions.

## Appendix
Contains information on the required Python packages.
