# SGD_Boost

This repository contains the official PyTorch implementation of the paper <em>"Why Transformers Don’t Need Adam: Scale Is All You Need"</em> at [arxiv](https://arxiv.org/abs/), providing a runtime and memory efficient optimizer for training large language models and diffusion models, named **"SGD-Boost"**.

Adaptive gradient methods like Adam and AdamW are popular for training transformer-based models due to their strong performance, but they have significant memory overhead and require extensive hyperparameter tuning. In this work, we argue that these adaptive methods are not necessary for effective training.

SGD-Boost introduces a learning rate rescaling strategy based on initial gradient patterns, applied to stochastic gradient descent with momentum (SGDM). This simple modification allows SGDM to achieve performance levels comparable to AdamW while reducing memory consumption and execution time. By removing the need to store second-order momentum terms, our approach reduces optimizer state memory by half, providing a “free lunch” in training efficiency.

Our method also enhances robustness to variations in learning rate and weight decay during ViT training on the Imagenet-1K task. Experimental results show that it outperforms existing optimizers in LoRA training for both large language models (LLMs) and diffusion models (DMs). Specifically, it enables full precision (FP32) training of GPT-2 (1.5B parameters) on a single RTX3090 and Llama2-7B on an A100-80G GPU. Code is now available at [GitHub](https://github.com/AnonymousAlethiometer/SGD_Boost).


<!-- [Overview](./figures/overview.svg) -->
<p align='center'>
    <img src="./figures/overview.svg" height='300px'/>
    <br/>
    <em><b>Figure1:</b> We analyze four key parameters: the weights of the Query, Key, and Value (QKV) in the first Transformer block; the normalization layer; the fully connected layers within that block; and the final MLP head layer. The gradient signal-to-noise ratio (g-SNR) differs across various parameter groups but remains stable throughout the training process. We utilize this signal to create a scaling strategy that adjusts the fixed learning rates in Stochastic Gradient Descent (SGD).</em>
</p>
<p align='center'>
    <img src="./figures/optimizer_memory_comparison.svg" height='300px'/>
    <img src="./figures/3d_scatter.svg" height='300px'/>
    <br/>
    <em>
    <b>Figure2:</b>
    <b>Left:</b>The figure shows the significant memory overhead for optimizer states with increasing model sizes. SGD-Boost maintains a much lower memory usage compared to other optimizers.
    <b>Right:</b>This figure displays the results from a grid search conducted on the classic ResNet18 model using the CIFAR10 dataset. The maximum top-1 test accuracy is highlighted in red text. Our method surpasses other popular optimizers, achieving the highest test accuracy.
    </em>
</p>
<p align='center'>
    <img src="./figures/vit_3d_scatter.svg" height='450px'/>
    <img src="./figures/algorithm_pseudocode.png" height='450px'/>
    <br/>
    <em>
    <b>Figure3:</b> 
    <b>Left:</b> The figure presents the results of a grid search conducted on the ViT-S/16 model using the ImageNet-1K dataset. For better visualization and comparison, the axis for the learning rate has been inverted for SGD-like optimizers. The maximum top-1 test accuracy is highlighted in red text. Additionally, our method demonstrates superior performance compared to other popular optimizers, particularly in terms of stability in response to changes in hyperparameters.
    </em>
    <b>Right:</b>The pseudocode of the SGD-Boost optimizer.
</p>

<!-- [Memory Comparison](./figures/optimizer_memory_comparison.svg) -->
<!-- <img src="./figures/optimizer_memory_comparison.svg"/> -->
<!-- [Stability & Perfomance with Hyperparameters Changes](./figures/3d_scatter.svg) -->
<!-- <img src="./figures/3d_scatter.svg" /> -->

<!-- <img src='./figures/algorithm_pseudocode.png'/> -->

## How To Use

### Installation
Prerequisites:
- Python >= 3.6
- PyTorch >= 1.7.0

Since most of this optimizer uses the native PyTorch APIs, it should have a wider compatibility with different versions of PyTorch. However, we recommend using the Pytorch 2.X version for better performance and compatibility.

Install from PyPI:
```bash
pip install sgd-boost
```

Install from the source code:
```bash
git clone https://github.com/AnonymousAlethiometer/SGD_Boost.git

cd SGD_Boost

# you can use the flag "--use-feature=in-tree-build" to avoid the warning of "FutureWarning: The 'build' command is deprecated"
pip install . --use-feature=in-tree-build

# [Optional] Or you can use '-e' flag to install in editable mode
pip install -e . --use-feature=in-tree-build
```


### Usgae of the optimizer:

The optimizer is normally used in the following way:

```python
from sgd_boost import SGDBoost

# initialize the optimizer
optimizer = SGD_boost(model.parameters(), lr=lr, momentum=0.9, eps=1e-08, weight_decay=weight_decay)

for _ in range(steps):
    pred = model(input_ids)
    loss = loss_fn(pred, labels)
    # calculate the gradient
    loss.backward()
    # process the warmup step, only need once after the gradient is calculated
    if not hasattr(optimizer, 'has_warmup') and hasattr(optimizer, 'warmup_step'):
        optimizer.warmup_step()
        optimizer.has_warmup = True
    # update the parameters
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

### Distributed Training

For distributed training, you need to ensure to perform this g-SNR calculation (refer as the `.warmup step()`) on each worker. Even if you accidentally perform it multiple times, it will not affect the final result thanks to the stability of the g-SNR values. Feel free to use the optimizer in your own training scripts. 

In most circumstances, you only need to replace the original optimizer with our optimizer, perform the `.warmup step()` after first gradient calculation (aka. the first effective invoke of `loss.backwards()`) and keep the rest of the code unchanged.


## Example:

The CNN examples lie in the `examples` directory. It contains the training code for CNN models, as well as the profiling code for the optimizer perfomance evaluation.

Please follow the README in that directory will guide you to restore the environment. Due to the procedure of anonymization, although the main part has been leaved unchanged, some of the components may not be available, try to delete the unavailable resources as needed. 

The ViT example will be released soon.


## Acknowledgement
1. The codebase is based on the [timm:pytorch-image-models](https://github.com/huggingface/pytorch-image-models)(ViT training example, release soon), [NanoGPT](https://github.com/karpathy/nanoGPT) and [Adam-mini](https://github.com/zyushun/Adam-mini)(GPT2 training) repository.

2. We thanks for [Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) for an accurate and efficient way to profile the memory usage of the optimizer.


## Citation
If you find this work helpful, please consider citing our paper:
```
@article{XXXXXXXXXX,
  title={Why Transformers Don’t Need Adam: Scale Is All You Need},
  author={Anonymous},
  journal={arXiv preprint arXiv:24XX.XXXXX},
  year={2024}
}
```
