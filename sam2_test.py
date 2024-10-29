from sam2 import load_model
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from concave_hull import concave_hull
from sklearn.cluster import DBSCAN
import requests
from io import BytesIO

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_polygon(mask, ax):
    # Get the mask coordinates
    mask_coords = np.argwhere(mask)
    # Use DBSCAN to cluster the points and choose the largest cluster
    clustering = DBSCAN(eps=5, min_samples=10).fit(mask_coords)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster = unique_labels[np.argmax(counts)]
    mask_coords = mask_coords[labels == largest_cluster]

    # Get the polygon from concave hull
    polygon = concave_hull(mask_coords, concavity=1, length_threshold=10)
    print(f"Polygon shape: {polygon.shape}")
    # Plot the polygon
    ax.plot(polygon[:, 1], polygon[:, 0], color='yellow', lw=2)

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    # Find the mask with the highest score.
    highest_index = np.argmax(scores)
    mask = masks[highest_index]
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(mask, plt.gca(), borders=borders)
    show_polygon(mask, plt.gca())
    if point_coords is not None:
        assert input_labels is not None
        show_points(point_coords, input_labels, plt.gca())
    if box_coords is not None:
        # boxes
        show_box(box_coords, plt.gca())
    if len(scores) > 1:
        plt.title(f"Mask {highest_index+1}, Score: {scores[highest_index]:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

def segment_sam2():
    model = load_model(
        variant="tiny",
        ckpt_path="models/sam2/sam2_hiera_tiny.pt",
        device="cpu"
    )
    image_base = 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HM54MGYC39W9MZ32YD4GJ59D'
    image_url = f'{image_base}/full/1024,/0/default.jpg'
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = np.array(image.convert("RGB"))
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)
    input_box = np.array([459, 345, 559, 487])
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=True,
    )
    show_masks(image, masks, scores, box_coords=input_box)

segment_sam2()
