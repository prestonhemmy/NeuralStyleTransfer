# Neural Style Transfer with PyTorch

A PyTorch implementation of the Neural Style Transfer algorithm, allowing you to apply the artistic style of one image to the content of another.

[//]: <> (Todo: Add image)

## Overview

This project implements Neural Style Transfer (NST), an algorithm that manipulates images by applying the artistic style of one image to the content of another. This technique was introduced by Gatys et al. in their paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) (2015).

The implementation uses PyTorch and a pre-trained VGG19 network to:
1. Extract content features from a content image
2. Extract style features from a style image
3. Generate a new image that combines the content of the first with the style of the second

## Features

- Apply artistic styles to any content image
- Visualize the optimization process
- Adjust content/style weights to control the balance
- Track loss changes throughout the iterations
- Save intermediate results to see the transformation

[//]: <> (TODO:## Examples)

<!--
| Content Image | Style Image | Result |
|---------------|-------------|--------|
| <img src="images/content/dancing.jpg" width="200"> | <img src="images/style/starry_night.jpg" width="200"> | <img src="images/output/result.png" width="200"> |
-->

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- Pillow
- matplotlib
- numpy

[//]: <> (Update with remaining information as development continues)
<!--## Usage

### Quick Start

```python
python main.py --content images/content/your_content.jpg --style images/style/your_style.jpg
```

### Advanced Options

```python
python main.py --content images/content/your_content.jpg \
               --style images/style/your_style.jpg \
               --output images/output/result.jpg \
               --steps 300 \
               --content-weight 1 \
               --style-weight 1000000 \
               --image-size 512
```

### Using as a Module

```python
from nst import run_style_transfer, load_image, plot_images

# Load images
content_img = load_image('images/content/your_content.jpg')
style_img = load_image('images/style/your_style.jpg')

# Run style transfer
output_img, _, _, _ = run_style_transfer(
    content_img, style_img, 
    num_steps=300, 
    content_weight=1, 
    style_weight=1000000
)

# Display results
plot_images(content_img, style_img, output_img)
```

## How It Works

Neural Style Transfer leverages the capabilities of Convolutional Neural Networks (CNNs) to separate and recombine content and style from images:

1. **Content Representation**: 
   - Higher layers in the CNN capture the high-level content (objects, arrangement)
   - We use these layers to ensure the generated image maintains the same content as the content image

2. **Style Representation**: 
   - Style is captured by the correlations between different filter responses in multiple layers
   - These correlations are calculated using the Gram matrix
   - Multiple style layers are used to capture both fine and coarse style elements

3. **Optimization Process**:
   - Start with the content image
   - Iteratively update the image to minimize both content and style loss
   - The L-BFGS optimizer is used for faster convergence

## Project Structure

```
neural-style-transfer/
├── images/
│   ├── content/          # Content images
│   ├── style/            # Style images 
│   └── output/           # Generated outputs
├── src/
│   ├── model.py          # Model definitions and loss functions
│   ├── utils.py          # Utility functions for image processing
│   └── visualization.py  # Functions to visualize results
├── main.py               # Main script to run style transfer
├── README.md
└── requirements.txt
```

## Future Enhancements

- [ ] Add a web interface for easier experimentation
- [ ] Implement multiple style layers with different weights
- [ ] Add total variation loss for smoother results
- [ ] Support video style transfer
- [ ] Experiment with different pre-trained models
- [ ] Improve performance with GPU acceleration

## References

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)
2. PyTorch Tutorial: [Neural Transfer Using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
3. Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual Losses for Real-Time Style Transfer and Super-Resolution. [arXiv:1603.08155](https://arxiv.org/abs/1603.08155)
-->

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- Preston Hemmy - [GitHub](https://github.com/prestonhemmy) - [LinkedIn](https://linkedin.com/in/preston-hemmy)

---
