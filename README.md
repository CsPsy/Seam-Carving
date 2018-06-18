# Seam Carving
This is a project for a visual course with Chris Wu.

In this project, we implemented the system of seam carving, a algorithm for content-aware image re-sizing. In Task one, We computed the gradient energy map over the image based on the gradients and local entropy of pixels. In Task two, we added energy for seam (forward energy) in the process of dynamic programming to improve the performance. In Task three, we use several deep learning techniques such as grad-CAM, guided back-propagation and guided grad-CAM, to design more sophisticated, semantically motivated energy functions and compare their effects. During the tasks, we also tried to apply some techniques to optimize time consumption.

# Demo
python3 seam_main.py <your_image>.png <newwidth> <newheight> <mode> <output_img>.png

mode 0: basic energy
mode 1: local entropy
mode 2: forward energy
mode 3: guided Grad-CAM

# Reference
http://www.faculty.idc.ac.il/arik/papers/imret.pdf

http://www.faculty.idc.ac.il/arik/SCWeb/vidret/vidretLowRes.pdf
