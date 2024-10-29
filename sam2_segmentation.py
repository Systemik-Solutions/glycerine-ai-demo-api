from sam2 import load_model
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image
from concave_hull import concave_hull
from sklearn.cluster import DBSCAN
import requests
from io import BytesIO

def segment_sam2(image_url, bounding_box):
    model = load_model(
        variant="tiny",
        ckpt_path="models/sam2/sam2_hiera_tiny.pt",
        device="cpu"
    )
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = np.array(image.convert("RGB"))
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)
    input_box = np.array(bounding_box)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=True,
    )
    # Find the mask with the highest score.
    highest_index = np.argmax(scores)
    mask = masks[highest_index]
    score = scores[highest_index]
    polygon = get_mask_polygon(mask)
    return {"image": {"url": image_url, "width": image.shape[1], "height": image.shape[0]}, "box": bounding_box, "segment": polygon.tolist(), "score": score.item()}

def get_mask_polygon(mask):
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

    # Convert the polygon in x,y format
    polygon = polygon[:, [1, 0]]
    return polygon
