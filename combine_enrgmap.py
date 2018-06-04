from seam_operates import visualize_energy_map
import numpy as np
import sys
import cv2
from misc_functions import (get_params,
                            convert_to_grayscale,
                            save_gradient_images, preprocess_image)
import torchvision
from grad_cam import GradCam
from guided_backprop import GuidedBackprop
from guided_grad_cam import guided_grad_cam

if __name__ == "__main__":
    '''
    img_path = sys.argv[1]
    out_path = sys.argv[2]
    original_image = cv2.imread(img_path)
    visualize_energy_map(None, original_image, out_path + 'enegy_map.png', mode=0, opt=True)

    '''
    gradient_map_path = sys.argv[1]
    saliency_map_path = sys.argv[2]
    combine_map_path = sys.argv[3]
    gradient_map = cv2.imread(gradient_map_path)
    saliency_map = cv2.imread(saliency_map_path)
    cv2.imwrite(combine_map_path, 0.5*gradient_map+0.5*saliency_map)