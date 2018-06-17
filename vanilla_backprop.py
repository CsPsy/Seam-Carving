"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
import torchvision
from misc_functions import get_params, convert_to_grayscale, \
    save_gradient_images,preprocess_image
import sys
import cv2
from PIL import Image
import numpy as np

class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class=None):
        # Forward
        model_output = self.model(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    img_path = sys.argv[1]
    out_path = sys.argv[2]
    original_image = Image.open(img_path).convert('RGB')
    prep_img = preprocess_image(original_image)
    # Get params
    pretrained_model = torchvision.models.vgg16(pretrained=True)
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img)
    # Save colored gradients
    save_gradient_images(vanilla_grads, out_path + '_Vanilla_BP_color.png')
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads, out_path + '_Vanilla_BP_gray.png')
    print('Vanilla backprop completed')
