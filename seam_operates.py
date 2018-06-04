from PIL import Image
import numpy as np
from skimage.filters.rank import entropy
import cv2 as cv
from tqdm import tqdm
#from grad_cam import *
from guided_grad_cam import (grad_cam, guided_backprop, preprocess_image,
                             guided_grad_cam, convert_to_grayscale)

ENTROPY_SELEM = np.ones((9,9), dtype=np.uint8)
KERNEL = np.array([[1,1,1], [1,-8,1], [1,1,1]])
ENTROPY_WEIGHT = 2
FORWARD_WEIGHT = 1
CAM_WEIGHT = 1

def valid(i,j,h,w):
    #given the height and width of a picture, verify if (i,j) is valid
    if i < 0 or i >= h or j < 0 or j >= w:
        return False
    return True

def min_from_three(mode, img, array, i, j, h, w):
    #get the min energy among (i-1,j-1), (i-1,j) and (i-1,j+1)
    a = float('inf')
    if valid(i-1,j-1,h,w):
        a = array[i-1][j-1]
    b = float('inf')
    if valid(i-1,j,h,w):
        b = array[i-1][j]
    c = float('inf')
    if valid(i-1,j+1,h,w):
        c = array[i-1][j+1]

    #add forward energy
    #caution: if we want to add forward energy in other mode, modify here
    if mode == 2:
        if valid(i,j+1,h,w) and valid(i,j-1,h,w):
            tmp = np.sum(np.abs(img[i][j+1] - img[i][j-1])) / 3
            a += FORWARD_WEIGHT * tmp
            b += FORWARD_WEIGHT * tmp
            c += FORWARD_WEIGHT * tmp
        if valid(i-1,j,h,w) and valid(i,j-1,h,w):
            a += FORWARD_WEIGHT * np.sum(np.abs(img[i-1][j] - img[i][j-1])) / 3
        if valid(i-1,j,h,w) and valid(i,j+1,h,w):
            c += FORWARD_WEIGHT * np.sum(np.abs(img[i-1][j] - img[i][j+1])) / 3

    ls = np.array([a, b, c])
    idx = np.argmin(ls)
    return j + idx - 1, ls[idx]

def get_range(i, j, h, w):
    # return left-top, right-bottom and nr_pixels of patch of pixel (i,j)
    if i > 0 and i < h-1 and j > 0 and j < w-1:
        return (-1,-1), (1,1), 8
    if i == 0 and j == 0:
        return (0,0), (1,1), 3
    if i == 0 and j == w-1:
        return (0,-1), (1,0), 3
    if i == h-1 and j == 0:
        return (-1,0), (0,1), 3
    if i == h-1 and j == w-1:
        return (-1,-1), (0,0), 3
    if i == 0:
        return (0,-1), (1,1), 5
    if i == h-1:
        return (-1,-1), (0,1), 5
    if j == 0:
        return (-1,0), (1,1), 5
    if j == w-1:
        return (-1,-1), (1,0), 5

def gray_scale_image(img):
    #turn a rgb img to gray
    grayimg = 0.30*img[:, :, 0] + 0.59*img[:, :, 1] + 0.11*img[:, :, 2]
    return grayimg.astype(np.uint8)

def energy_of_element_with_abs(imgarr,i,j,height,width):
    # return basic energy of pixel (i,j)
    left_top, right_bottom, nr_pixels = get_range(i, j, height, width)
    patch = imgarr[i+left_top[0]:i+right_bottom[0] + 1, j+left_top[1]:j+right_bottom[1] + 1, :]
    energy = np.sum(np.abs(patch - imgarr[i][j])) / (3 * nr_pixels)
    return energy

def compute_energy_function_with_abs(imgarr, mode=0):
    # return energy over whole img, with abs according to the instruction
    imgarr = imgarr.astype(float)
    height, width, channels = imgarr.shape

    enrg  = np.zeros((height,width),dtype=np.double)
    for i in range(height):
        for j in range(width):
            enrg[i][j] = energy_of_element_with_abs(imgarr,i,j,height,width)

    if mode == 1:
        ent = entropy(gray_scale_image(imgarr), ENTROPY_SELEM)
        enrg = enrg + ENTROPY_WEIGHT * ent

    return enrg

def compute_saliency_map(imgarr):
    #return the enrg for part 3

    prep_img = preprocess_image(imgarr)
    cam, target_class = grad_cam.generate_cam(prep_img)
    guided_grads = guided_backprop.generate_gradients(prep_img, target_class)
    cam_gb = guided_grad_cam(cam, guided_grads)
    cam_gb = convert_to_grayscale(cam_gb)
    cam_gb = cam_gb / (np.max(cam_gb) - np.min(cam_gb)) * 255
    cam_gb = cv.resize(cam_gb.reshape(224, 224), (imgarr.shape[1], imgarr.shape[0]))
    '''
    prep_img = preprocess_image(imgarr)
    enrg = grad_cam.generate_cam(prep_img)
    enrg = cv.resize(enrg, (imgarr.shape[1], imgarr.shape[0]))
    enrg = (enrg / (np.max(enrg) - np.min(enrg)))
    '''
    return cam_gb

def get_vertical_seam(mode,img,enrg):
    # get a vertical seam by dynamic programming
    height,width = enrg.shape
    dp = np.zeros_like(enrg)
    B = np.zeros_like(enrg,dtype = np.int64)
    for i in range(height):
        for j in range(width):
            dp[i][j] = enrg[i][j]
            if i!=0:
                B[i][j],M = min_from_three(mode,img,dp,i,j,height,width)
                dp[i][j] += M

    # find the minimum vertical seam
    seam_b = np.argmin(dp[height - 1])
    # trace back to find the seam route
    seam = [seam_b]

    for i in range(height-1,0,-1):
        seam_b = B[i][seam_b]
        seam += [seam_b]

    seam.reverse()

    return seam

def delete_seam(seam, imgarr):
    # delete a seam on a img
    height, width, colors = imgarr.shape
    img = np.zeros((height, width - 1, colors), dtype=np.int64)
    for i, j in enumerate(seam):
        img[i][:j] = imgarr[i][:j]
        if j + 1 < width:
            img[i][j:] = imgarr[i][j + 1:]
    return img

def add_seam(seam, imgarr):
    #add a seam on a img
    height, width, colors = imgarr.shape
    img = np.zeros((height, width + 1, colors), dtype=np.int64)
    for i, j in enumerate(seam):
        img[i][:j] = imgarr[i][:j]
        # add new column
        if j == 0:
            img[i][j] = np.mean(imgarr[i, j:j + 3], axis=0)
        elif j == width - 1:
            img[i][j] = np.mean(imgarr[i, j - 2:j + 1], axis=0)
        else:
            img[i][j] = np.mean(imgarr[i, j - 1:j + 2], axis=0)
        img[i, j + 1:] = imgarr[i, j:]
    return img

def verti_op_pic(img,newwidth,mode,grad_cam=None):
    #do vertical operations on a image
    height,width,colors = img.shape
    verti_seams = abs(width - newwidth)
    if width >= newwidth:
        for i in tqdm(range(verti_seams)):
            enrg = compute_energy_function_by_con(img, mode)
            enrg = enrg / (np.max(enrg) - np.min(enrg)) * 255
            if mode == 3:
                assert grad_cam is not None
                enrg += CAM_WEIGHT * compute_saliency_map(img)
            seam = get_vertical_seam(mode, img, enrg)
            img = delete_seam(seam, img)
    else:
        tmp_img = np.copy(img)
        seam_stack = []
        for i in tqdm(range(verti_seams)):
            enrg = compute_energy_function_by_con(tmp_img, mode)
            enrg = enrg / (np.max(enrg) - np.min(enrg)) * 255
            if mode == 3:
                assert grad_cam is not None
                enrg += CAM_WEIGHT * compute_saliency_map(tmp_img)
            seam = get_vertical_seam(mode, tmp_img, enrg)
            tmp_img = delete_seam(seam, tmp_img)
            # reset the position of current seam
            if i > 0:
                pre_seam = seam_stack[-1]
                for i, j in enumerate(seam):
                    if j > pre_seam[i]:
                        seam[i] += 1
            seam_stack.append(seam)

        seam_mat = np.array(seam_stack)
        for i in range(seam_mat.shape[0]):
            img = add_seam(list(seam_mat[i]), img)
            seam_mat[i + 1:][np.where(seam_mat[i + 1:] > seam_mat[i])] += 1

    return img

def hori_op_pic(img,newheight,mode,grad_cam=None):
    #do horizontal operations on a image
    height,width,colors = img.shape
    img = np.array(img,dtype = np.double)
    img = img.transpose((1,0,2))
    img = verti_op_pic(img, newheight, mode, grad_cam)
    img = img.transpose((1,0,2))
    return img

def visualize_energy_map(grad_cam,imgarr, filepath, mode=0, opt=True):
    #visualize the energy map
    if opt == True:
        enrg = compute_energy_function_with_abs(imgarr, mode)
    else:
        enrg = compute_energy_function_by_con(imgarr, mode)
    if mode == 3:
        assert grad_cam is not None
        enrg = enrg / (np.max(enrg) - np.min(enrg)) * 255
        enrg += CAM_WEIGHT*compute_saliency_map(imgarr)
    enrg = enrg / (np.max(enrg) - np.min(enrg))
    enrg = enrg * 255
    enrg_heatmap = cv.applyColorMap(enrg, cv.COLORMAP_JET)
    cv.imwrite(filepath, enrg_heatmap)


def show_image(imgarr):
    #show a img or a mask from array
    img = Image.fromarray(np.uint8(imgarr))
    img.show()
    #plt.imshow(imgarr,cmap ='gray')
    #plt.show()
    return


def delete_seam_with_opt(mode,seam,imgarr,enrg):
    # delete or add a seam and update the energy map instead of computing the whole energy map every time
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


def verti_op_pic_with_opt(img,newwidth,mode):
    #the verti_op_pic function for the optimization of delete_seam
    height,width,colors = img.shape
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
                for i, j in enumerate(seam):
                    if j >= pre_seam[i]:
                        seam[i] += 1
            seam_stack.append(seam)
        # add to the image
        seam_mat = np.array(seam_stack)
        for i in range(seam_mat.shape[0]):
            img = add_seam(list(seam_mat[i]), img)
            seam_mat[i + 1:][np.where(seam_mat[i + 1:] > seam_mat[i])] += 1

    return img


def hori_op_pic_with_opt(img,newheight,mode):
    img = np.array(img,dtype = np.double)
    img = img.transpose((1,0,2))
    img = verti_op_pic_with_opt(img, newheight, mode)
    img = img.transpose((1,0,2))
    return img

def compute_energy_function_by_con(imgarr, mode=0):
    # return energy over whole img
    imgarr = imgarr.astype(float)
    height, width, channels = imgarr.shape

    r, g, b = cv.split(imgarr)
    enrg_r = np.abs(cv.filter2D(r, -1, KERNEL))
    enrg_g = np.abs(cv.filter2D(g, -1, KERNEL))
    enrg_b = np.abs(cv.filter2D(b, -1, KERNEL))
    enrg = enrg_r + enrg_g + enrg_b

    enrg[0][0] = enrg[0][0] * 8 / 3
    enrg[0][width-1] = enrg[0][width-1] * 8 / 3
    enrg[height-1][0] =enrg[height-1][0] * 8 / 3
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