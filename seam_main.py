# seam_main.py - the main code file to run

from PIL import Image
from seam_operates import *
from guided_grad_cam import *
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help='input image path')
    parser.add_argument('out_width', help="output image's width", type=int)
    parser.add_argument('out_height', help="output image's height", type=int)
    parser.add_argument('energy_type', help="the energy_type you want to use, range = [0,1,2,3]",
                        type=int)
    parser.add_argument('out_path', help='output image path')
    parser.add_argument('--speedup', default=True, help='whether to speedup mode 3')
    args = parser.parse_args()

    if args.energy_type > 3:
        args.energy_type = 3

    since = time.time()
    img = Image.open(args.img_path).convert('RGB')
    img = np.asarray(img, dtype=np.double)

    assert img.shape[2] == 3

    if args.energy_type == 3:
        # visualize_energy_map(img, out_path + 'enegy_map.png', mode=energy_type, opt = False)
        set_ggcam()
        img = verti_op_pic(img, args.out_width, args.energy_type, args.speedup)
        img = hori_op_pic(img, args.out_height, args.energy_type, args.speedup)
    else:
        # visualize_energy_map(img, out_path + 'enegy_map.png', mode=energy_type, opt = True)
        img = verti_op_pic_with_opt(img, args.out_width, args.energy_type)
        img = hori_op_pic_with_opt(img, args.out_height, args.energy_type)

    img = Image.fromarray(np.uint8(img))
    img.save(args.out_path)
    total_time = time.time() - since
    print('Whole process takes {} m {} s'.format(total_time // 60, total_time % 60))
    return


if __name__ == '__main__':
    main()
