"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np
import sys
import cv2
from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images, preprocess_image)
import torchvision
from grad_cam import GradCam
from guided_backprop import GuidedBackprop

pretrained_model = torchvision.models.vgg16(pretrained=True)
# print(pretrained_model)
# Grad cam
grad_cam = GradCam(pretrained_model, target_layer=30)
guided_backprop = GuidedBackprop(pretrained_model)


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


if __name__ == '__main__':
    # Get params
    img_path = sys.argv[1]
    out_path = sys.argv[2]
    target_layer = sys.argv[3]
    original_image = cv2.imread(img_path)



    prep_img = preprocess_image(original_image)

    pretrained_model = torchvision.models.vgg16(pretrained=True)
    #print(pretrained_model)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=int(target_layer))
    # Generate cam mask
    cam, target_class = grad_cam.generate_cam(prep_img)
    print('grad cam completed')

    # guided backprop
    guided_backprop = GuidedBackprop(pretrained_model)
    guided_grads = guided_backprop.generate_gradients(prep_img, target_class)
    print('Guided backpropagation completed')

    cam_gb = guided_grad_cam(cam, guided_grads)

    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    save_gradient_images(grayscale_cam_gb, out_path+'gray.png',
                         shape=(original_image.shape[1], original_image.shape[0]))

    '''
    save_class_activation_on_image(original_image, cam, out_path)
    print('Grad cam completed')

    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_params(target_example)

    # Grad cam
    gcv2 = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = gcv2.generate_cam(prep_img, target_class)
    print('Grad cam completed')

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    print('Guided backpropagation completed')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
    print('Guided grad cam completed')
    '''