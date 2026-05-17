# Deep Generative Augmentations

Paper-facing group for generator-based baselines.

Included arms:

- `timevae_classwise_optional`
- `timegan_classwise`
- `timevqvae_classwise`
- `diffusionts_classwise`

Status notes:

- `timevae_classwise_optional` is a PyTorch clean-room / translation-style
  adapter, not an official Keras-pipeline parity claim.
- `timegan_classwise` is a compact PyTorch TimeGAN-style adapter, not an
  official TensorFlow-pipeline parity claim.
- `timevqvae_classwise` uses the TimeVQVAE adapter and vendored code tree.
- `diffusionts_classwise` uses the Diffusion-TS adapter and vendored code tree.

TimeCAE is not currently implemented in this group.
