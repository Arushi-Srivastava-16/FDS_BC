# breast_cancer_detection/utils/visualization.py

import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_proposals(image, boxes):
    """
    Visualizes image with bounding boxes.

    Args:
        image (Tensor): [C, H, W]
        boxes (Tensor): [N, 5]
    """
    image = TF.to_pil_image(image.cpu())
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in boxes:
        x, y, w, h, conf = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()
