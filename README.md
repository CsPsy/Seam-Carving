# Seam Carving
This is a project for a visual course with Chris Wu.

In this project, we implemented the system of seam carving, a algorithm for content-aware image re-sizing. In Task one, We computed the gradient energy map over the image based on the gradients and local entropy of pixels. In Task two, we added energy for seam (forward energy) in the process of dynamic programming to improve the performance. In Task three, we use several deep learning techniques such as grad-CAM, guided back-propagation and guided grad-CAM, to design more sophisticated, semantically motivated energy functions and compare their effects. During the tasks, we also tried to apply some techniques to optimize time consumption.

# Run
- You can run `sh run.sh` directly a simple test
- Or you can use command `python3 seam_main.py <input image path> <new width> <new height> <mode> <output image path>`
  - mode 0: basic energy
  - mode 1: +local entropy
  - mode 2: +forward energy
  - mode 3: DL based energy(guided Grad-CAM)

# Requirement
- python 3.6.2
- numpy 1.14.3
- pytorch 0.4.0
- pillow 3.4.2
- opencv 3.1.0

# Reference
http://www.faculty.idc.ac.il/arik/papers/imret.pdf

http://www.faculty.idc.ac.il/arik/SCWeb/vidret/vidretLowRes.pdf
