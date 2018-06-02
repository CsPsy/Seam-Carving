from seam_operates import *
from grad_cam import *
import time
import sys


def main():
    img_path = sys.argv[1]
    out_width = int(sys.argv[2])
    out_height = int(sys.argv[3])
    energy_type = int(sys.argv[4])
    if energy_type > 3:
        energy_type = 3
    out_path = sys.argv[5]
    since = time.time()
    img = Image.open(img_path).convert('RGB')
    img = np.asarray(img, dtype=np.double)

    assert img.shape[2] == 3
    if energy_type == 3:
        pretrained_model = torchvision.models.vgg16_bn(pretrained=True)
        grad_cam = GradCam(pretrained_model, target_layer=42)
    else:
        grad_cam = None

    if energy_type == 3:
        visualize_energy_map(grad_cam, img, out_path + 'enegy_map.png', mode=energy_type, opt = False)
        img = verti_op_pic(img, out_width, energy_type, grad_cam)
        img = hori_op_pic(img, out_height, energy_type, grad_cam)
    else:
        visualize_energy_map(grad_cam, img, out_path + 'enegy_map.png', mode=energy_type, opt = True)
        img = verti_op_pic_with_opt(img, out_width,energy_type)
        img = hori_op_pic_with_opt(img, out_height,energy_type)

    img = Image.fromarray(np.uint8(img))
    img.save(out_path)
    total_time = time.time() - since
    print('Whole process takes {} m {} s'.format(total_time // 60, total_time % 60))
    return

if __name__ == '__main__':
    main()

