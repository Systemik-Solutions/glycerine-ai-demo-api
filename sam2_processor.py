from contextlib import ExitStack
import numpy as np
import torch
from concave_hull import concave_hull
from sklearn.cluster import DBSCAN

class SAM2Processor:
    def __init__(self, model, processor, image):
        self.model = model
        self.image = image
        image_data = np.array(self.image.convert("RGB"))
        self.predictor = processor
        with ExitStack() as stack:
            if torch.cuda.is_available():
                stack.enter_context(torch.inference_mode())
                stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
            self.predictor.set_image(image_data)

    def segment(self, bounding_box):
        input_box = np.array(bounding_box)
        with ExitStack() as stack:
            if torch.cuda.is_available():
                stack.enter_context(torch.inference_mode())
                stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=True,
            )
        # Find the mask with the highest score.
        highest_index = np.argmax(scores)
        mask = masks[highest_index]
        score = scores[highest_index]
        polygon = self.__get_mask_polygon(mask)
        return {"image": {"width": self.image.width, "height": self.image.height}, "box": bounding_box, "segment": polygon.tolist(), "score": score.item()}
    
    def __get_mask_polygon(self, mask):
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
