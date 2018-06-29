# seam_operates.py - the functions for operating

import numpy as np
from skimage.filters.rank import entropy
import cv2 as cv
import torchvision
from tqdm import tqdm
from guided_grad_cam import *

# Global Variables
ENTROPY_WEIGHT = 5
FORWARD_WEIGHT = 1
CAM_WEIGHT = 1.5
ENTROPY_SELEM = np.ones((9, 9), dtype=np.uint8)
KERNEL = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
pretrained_model = torchvision.models.vgg16(pretrained=True)
grad_cam = GradCam(pretrained_model, target_layer=30)
guided_backprop = GuidedBackprop(pretrained_model)


def valid(i, j, h, w):
    """
    :param i: the row coordinate of the pixel
    :param j: the column coordinate of the pixel
    :param h: the height of the image
    :param w: the width of the image
    :return: whether (i,j) is in the image
    """

    if i < 0 or i >= h or j < 0 or j >= w:
        return False
    return True


def min_from_three(mode, img, array, i, j, h, w):
    """
    :param mode: the mode of energy
    :param img: the original image
    :param array: the dynamic map
    :param i: the row coordinate of the pixel
    :param j: the column coordinate of the pixel
    :param h: the height of the image
    :param w: the width of the image
    :return: the min energy among (i-1,j-1), (i-1,j) and (i-1,j+1)
    """

    a = float('inf')
    if valid(i-1, j-1, h, w):
        a = array[i-1][j-1]
    b = float('inf')
    if valid(i-1, j, h, w):
        b = array[i-1][j]
    c = float('inf')
    if valid(i-1, j+1, h, w):
        c = array[i-1][j+1]

    # Add forward energy
    # caution: if we want to add forward energy in other mode, modify here
    if mode == 2:
        if valid(i, j+1, h, w) and valid(i, j-1, h, w):
            tmp = np.sum(np.abs(img[i][j+1] - img[i][j-1])) / 3
            a += FORWARD_WEIGHT * tmp
            b += FORWARD_WEIGHT * tmp
            c += FORWARD_WEIGHT * tmp
        if valid(i-1, j, h, w) and valid(i, j-1, h, w):
            a += FORWARD_WEIGHT * np.sum(np.abs(img[i-1][j] - img[i][j-1])) / 3
        if valid(i-1, j, h, w) and valid(i, j+1, h, w):
            c += FORWARD_WEIGHT * np.sum(np.abs(img[i-1][j] - img[i][j+1])) / 3

    ls = np.array([a, b, c])
    idx = np.argmin(ls)
    return j + idx - 1, ls[idx]


def get_range(i, j, h, w):
    """
    :param i: the row coordinate of the pixel
    :param j: the column coordinate of the pixel
    :param h: the height of the image
    :param w: the width of the image
    :return: the left-top, right-bottom and nr_pixels of patch of pixel (i,j)
    """

    if 0 < i < h-1 and 0 < j < w-1:
        return (-1, -1), (1, 1), 8
    if i == 0 and j == 0:
        return (0, 0), (1, 1), 3
    if i == 0 and j == w-1:
        return (0, -1), (1, 0), 3
    if i == h-1 and j == 0:
        return (-1, 0), (0, 1), 3
    if i == h-1 and j == w-1:
        return (-1, -1), (0, 0), 3
    if i == 0:
        return (0, -1), (1, 1), 5
    if i == h-1:
        return (-1, -1), (0, 1), 5
    if j == 0:
        return (-1, 0), (1, 1), 5
    if j == w-1:
        return (-1, -1), (1, 0), 5


def gray_scale_image(img):
    """
    :param img: the original RGB image
    :return: the gray scale image of img
    """

    grayimg = 0.30*img[:, :, 0] + 0.59*img[:, :, 1] + 0.11*img[:, :, 2]
    return grayimg.astype(np.uint8)


def energy_of_element_with_abs(imgarr, i, j, height, width):
    """
    :param imgarr: the original image
    :param i: the row coordinate of the pixel
    :param j: the column coordinate of the pixel
    :param height: the height of the image
    :param width: the width of the image
    :return: the basic energy of pixel (i,j)
    """

    left_top, right_bottom, nr_pixels = get_range(i, j, height, width)
    patch = imgarr[i+left_top[0]:i+right_bottom[0] + 1, j+left_top[1]:j+right_bottom[1] + 1, :]
    energy = np.sum(np.abs(patch - imgarr[i][j])) / (3 * nr_pixels)
    return energy


def compute_energy_function_with_abs(imgarr, mode=0):
    """
    :param imgarr: the original image
    :param mode: the mode of energy
    :return: the energy map over whole image, with abs according to the instruction
    """

    imgarr = imgarr.astype(float)
    height, width, channels = imgarr.shape

    enrg = np.zeros((height, width), dtype=np.double)
    for i in range(height):
        for j in range(width):
            enrg[i][j] = energy_of_element_with_abs(imgarr, i, j, height, width)

    if mode == 1:
        ent = entropy(gray_scale_image(imgarr), ENTROPY_SELEM)
        enrg = enrg + ENTROPY_WEIGHT * ent

    return enrg


def get_vertical_seam(mode, img, enrg):
    """
    :param mode: the mode of energy
    :param img: the original image
    :param enrg: the energy map of the image
    :return: a vertical seam with the lowest energy by dynamic programming
    """

    height, width = enrg.shape
    dp = np.zeros_like(enrg)
    b = np.zeros_like(enrg, dtype=np.int64)
    for i in range(height):
        for j in range(width):
            dp[i][j] = enrg[i][j]
            if i != 0:
                b[i][j], m = min_from_three(mode, img, dp, i, j, height, width)
                dp[i][j] += m

    # find the minimum vertical seam
    seam_b = np.argmin(dp[height - 1])
    # trace back to find the seam route
    seam = [seam_b]

    for i in range(height-1, 0, -1):
        seam_b = b[i][seam_b]
        seam += [seam_b]

    seam.reverse()

    return seam


def delete_seam(seam, imgarr):
    """
    :param seam: a seam on the image
    :param imgarr: the original image
    :return: the image after deleting the seam
    """

    height, width, colors = imgarr.shape
    img = np.zeros((height, width - 1, colors), dtype=np.int64)
    for i, j in enumerate(seam):
        img[i][:j] = imgarr[i][:j]
        if j + 1 < width:
            img[i][j:] = imgarr[i][j + 1:]
    return img


def delete_map_seam(seam, maparr):
    """
    :param seam: a seam on the image
    :param maparr: the energy map
    :return: the energy map after deleting the seam
    """

    height, width = maparr.shape
    img = np.zeros((height, width - 1), dtype=np.int64)
    for i, j in enumerate(seam):
        img[i][:j] = maparr[i][:j]
        if j + 1 < width:
            img[i][j:] = maparr[i][j + 1:]
    return img


def add_seam(seam, imgarr):
    """
    :param seam: a seam on the image
    :param imgarr: the original image
    :return: the image after adding the seam
    """

    height, width, colors = imgarr.shape
    img = np.zeros((height, width + 1, colors), dtype=np.int64)
    for i, j in enumerate(seam):
        img[i][:j] = imgarr[i][:j]
        if j == 0:
            img[i][j] = np.mean(imgarr[i, j:j + 3], axis=0)
        elif j == width - 1:
            img[i][j] = np.mean(imgarr[i, j - 2:j + 1], axis=0)
        else:
            img[i][j] = np.mean(imgarr[i, j - 1:j + 2], axis=0)
        img[i, j + 1:] = imgarr[i, j:]
    return img


def verti_op_pic(img, newwidth, mode):
    """
    :param img: the original image
    :param newwidth: the new width expected for the output
    :param mode: the mode of energy
    :return: the image after vertical operations
    """

    height, width, colors = img.shape
    verti_seams = abs(width - newwidth)
    smap = None
    if width >= newwidth:
        if mode == 3:
            smap = compute_saliency_map(img)
        for i in tqdm(range(verti_seams)):
            enrg = compute_energy_function_by_con(img, mode)
            enrg = enrg / (np.max(enrg) - np.min(enrg)) * 255
            if mode == 3:
                smap = compute_saliency_map(img)  # recompute or not
                enrg += CAM_WEIGHT * smap
            seam = get_vertical_seam(mode, img, enrg)
            img = delete_seam(seam, img)
            if mode == 3:
                smap = delete_map_seam(seam, smap)
    else:
        tmp_img = np.copy(img)
        seam_stack = []
        if mode == 3:
            smap = compute_saliency_map(tmp_img)
        for i in tqdm(range(verti_seams)):
            enrg = compute_energy_function_by_con(tmp_img, mode)
            enrg = enrg / (np.max(enrg) - np.min(enrg)) * 255
            if mode == 3:
                smap = compute_saliency_map(tmp_img)  # recompute or not
                enrg += CAM_WEIGHT * smap
            seam = get_vertical_seam(mode, tmp_img, enrg)
            tmp_img = delete_seam(seam, tmp_img)
            if mode == 3:
                smap = delete_map_seam(seam, smap)

            # reset the position of current seam
            if i > 0:
                pre_seam = seam_stack[-1]
                for k, j in enumerate(seam):
                    if j > pre_seam[k]:
                        seam[k] += 1
            seam_stack.append(seam)

        seam_mat = np.array(seam_stack)
        for i in range(seam_mat.shape[0]):
            img = add_seam(list(seam_mat[i]), img)
            seam_mat[i + 1:][np.where(seam_mat[i + 1:] > seam_mat[i])] += 1

    return img


def hori_op_pic(img, newheight, mode):
    """
    :param img: the original image
    :param newheight: the new height expected for the output
    :param mode: the mode of energy
    :return: the image after horizontal operations
    """

    # Do horizontal operations on a image
    img = np.array(img, dtype=np.double)
    img = img.transpose((1, 0, 2))
    img = verti_op_pic(img, newheight, mode)
    img = img.transpose((1, 0, 2))
    return img


def visualize_energy_map(imgarr, filepath, mode=0, opt=True):
    """
    :param imgarr: the original image
    :param filepath: the file path where the energy map will be
    :param mode: the mode of energy
    :param opt: whether using optimization
    :return: none
    """

    if opt:
        enrg = compute_energy_function_with_abs(imgarr, mode)
    else:
        enrg = compute_energy_function_by_con(imgarr, mode)
    if mode == 3:
        enrg = enrg / (np.max(enrg) - np.min(enrg)) * 255
        smap = compute_saliency_map(imgarr)
        enrg += CAM_WEIGHT*smap
    enrg = enrg / (np.max(enrg) - np.min(enrg))
    enrg = enrg * 255
    enrg_heatmap = cv.applyColorMap(enrg, cv.COLORMAP_JET)
    cv.imwrite(filepath, enrg_heatmap)


def delete_seam_with_opt(mode, seam, imgarr, enrg):
    """
    :param mode: the mode of energy
    :param seam: a seam on the image
    :param imgarr: the original image
    :param enrg: the energy map of the image
    :return: the image after deleting the seam and the updated energy map
    """

    # Optimization: Delete a seam and update the energy map instead of computing the whole energy map every time
    height, width, colors = imgarr.shape
    img = np.zeros((height, width - 1, colors), dtype=np.int64)
    newenrg = np.zeros((height, width - 1), dtype=np.double)

    # reset the energy of pixels along and near the seam
    for i, j in enumerate(seam):
        lt, rb, nr_pixels = get_range(i, j, height, width)
        enrg[i+lt[0]:i+rb[0]+1, j+lt[1]:j+rb[1]+1] = -1

    for i, j in enumerate(seam):
        img[i][:j] = imgarr[i][:j]
        newenrg[i][:j] = enrg[i][:j]  # maybe can use np.delete
        if j + 1 < width:
            img[i][j:] = imgarr[i][j + 1:]
            newenrg[i][j:] = enrg[i][j + 1:]

    height, width = newenrg.shape
    ent = np.zeros(newenrg.shape)
    if mode == 1:
        ent = ENTROPY_WEIGHT * entropy(gray_scale_image(img), ENTROPY_SELEM)

    # deal with newenrg which shall be changed
    for i in range(height):
        for j in range(width):
            if newenrg[i][j] == -1:
                newenrg[i][j] = energy_of_element_with_abs(img, i, j, height, width) + (mode == 1) * ent[i][j]

    return img, newenrg


def verti_op_pic_with_opt(img, newwidth, mode):
    """
    :param img: the original image
    :param newwidth: the expected width of the output
    :param mode: the mode of energy
    :return:
    """

    # Optimization: Delete a seam and update the energy map instead of computing the whole energy map every time
    height, width, colors = img.shape
    verti_seams = abs(width - newwidth)
    if width >= newwidth:
        enrg = compute_energy_function_with_abs(img, mode)
        for times in tqdm(range(verti_seams)):
            seam = get_vertical_seam(mode, img, enrg)
            img, enrg = delete_seam_with_opt(mode, seam, img, enrg)
    else:
        tmp_img = np.copy(img)
        seam_stack = []
        enrg = compute_energy_function_with_abs(tmp_img, mode)
        for i in tqdm(range(verti_seams)):
            seam = get_vertical_seam(mode, tmp_img, enrg)
            tmp_img, enrg = delete_seam_with_opt(mode, seam, tmp_img, enrg)
            # reset the position of current seam
            if i > 0:
                pre_seam = seam_stack[-1]
                for k, j in enumerate(seam):
                    if j >= pre_seam[k]:
                        seam[k] += 1
            seam_stack.append(seam)
        # add to the image
        seam_mat = np.array(seam_stack)
        for i in range(seam_mat.shape[0]):
            img = add_seam(list(seam_mat[i]), img)
            seam_mat[i + 1:][np.where(seam_mat[i + 1:] > seam_mat[i])] += 1

    return img


def hori_op_pic_with_opt(img, newheight, mode):
    """
    :param img: the original image
    :param newheight: the new height expected of the output
    :param mode: the mode of energy
    :return: the image after horizontal operations with optimization
    """

    # Revised hori_op_pic function for the optimization
    img = np.array(img, dtype=np.double)
    img = img.transpose((1, 0, 2))
    img = verti_op_pic_with_opt(img, newheight, mode)
    img = img.transpose((1, 0, 2))
    return img


def compute_saliency_map(imgarr):
    """
    :param imgarr: the original image
    :return: the guided grad-CAM energy map
    """

    prep_img = preprocess_image(imgarr)
    cam, target_class = grad_cam.generate_cam(prep_img)
    guided_grads = guided_backprop.generate_gradients(prep_img, target_class)
    cam_gb = guided_grad_cam(cam, guided_grads)
    cam_gb = convert_to_grayscale(cam_gb)
    cam_gb = cam_gb / (np.max(cam_gb) - np.min(cam_gb)) * 255
    cam_gb = cv2.resize(cam_gb.reshape(224, 224), (imgarr.shape[1], imgarr.shape[0]))
    return cam_gb


def compute_energy_function_by_con(imgarr, mode=0):
    """
    :param imgarr: the original image
    :param mode: the mode of energy
    :return: the energy map computed by laplace operator
    """

    imgarr = imgarr.astype(float)
    height, width, channels = imgarr.shape

    r, g, b = cv.split(imgarr)
    enrg_r = np.abs(cv.filter2D(r, -1, KERNEL))
    enrg_g = np.abs(cv.filter2D(g, -1, KERNEL))
    enrg_b = np.abs(cv.filter2D(b, -1, KERNEL))
    enrg = enrg_r + enrg_g + enrg_b

    enrg[0][0] = enrg[0][0] * 8 / 3
    enrg[0][width-1] = enrg[0][width-1] * 8 / 3
    enrg[height-1][0] = enrg[height-1][0] * 8 / 3
    enrg[height-1][width-1] = enrg[height-1][width-1] * 8 / 3
    for i in range(1, height):
        enrg[i][0] = enrg[i][0] * 8 / 5
        enrg[i][width-1] = enrg[i][width-1] * 8 / 5
    for j in range(1, width):
        enrg[0][j] = enrg[0][j] * 8 / 5
        enrg[height-1][j] = enrg[height-1][j] * 8 / 5

    enrg /= 3 * 8
    if mode == 1:
        ent = entropy(gray_scale_image(imgarr), ENTROPY_SELEM)
        enrg = enrg + ENTROPY_WEIGHT*ent
    return enrg
