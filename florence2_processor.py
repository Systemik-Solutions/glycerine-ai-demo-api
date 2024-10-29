import re
import torch

class Florence2Processor:
    def __init__(self, model, processor, image):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = model
        self.processor = processor
        self.image = image

    def __run_task(self, task_prompt, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=self.image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(self.image.width, self.image.height)
        )

        return parsed_answer
    
    def __box_to_text(self, box):
        image_width, image_height = self.image.size
        region = [int(box[0] / image_width * 999), int(box[1] / image_height * 999), int(box[2] / image_width * 999), int(box[3] / image_height * 999)]
        return f"<loc_{region[0]}><loc_{region[1]}><loc_{region[2]}><loc_{region[3]}>"
    
    def __parse_bbox_pairs(self, data):
        items = []
        for bbox, label in zip(data['bboxes'], data['labels']):
            items.append({
                'bbox': bbox,
                'label': label
            })
        return items
    
    def __parse_polygon_paris(self, data):
        items = []
        for polygon, label in zip(data['polygons'], data['labels']):
            items.append({
                'polygons': polygon,
                'label': label
            })
        return items
    
    def __parse_quad_box_pairs(self, data):
        items = []
        for quad_box, label in zip(data['quad_boxes'], data['labels']):
            items.append({
                'bbox': [quad_box[0], quad_box[1], quad_box[4], quad_box[5]],
                'label': label
            })
        return items
    
    def __remove_tags(self, text):
        # Remove all xml tags.
        return re.sub('<[^<]+?>', '', text)
    
    def get_image_size(self):
        return {"width": self.image.width, "height": self.image.height}
    
    def caption(self):
        task_prompt = '<CAPTION>'
        result = self.__run_task(task_prompt)
        return result[task_prompt]
    
    def detailed_caption(self):
        task_prompt = '<DETAILED_CAPTION>'
        result = self.__run_task(task_prompt)
        return result[task_prompt]
    
    def more_detailed_caption(self):
        task_prompt = '<MORE_DETAILED_CAPTION>'
        result = self.__run_task(task_prompt)
        return result[task_prompt]
    
    def object_detection(self):
        task_prompt = '<OD>'
        result = self.__run_task(task_prompt)
        return self.__parse_bbox_pairs(result[task_prompt])
    
    def dense_region_caption(self):
        task_prompt = '<DENSE_REGION_CAPTION>'
        result = self.__run_task(task_prompt)
        return self.__parse_bbox_pairs(result[task_prompt])
    
    def region_proposal(self):
        task_prompt = '<REGION_PROPOSAL>'
        result = self.__run_task(task_prompt)
        return self.__parse_bbox_pairs(result[task_prompt])
    
    def phrase_grounding(self, text_input):
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        result = self.__run_task(task_prompt, text_input=text_input)
        return self.__parse_bbox_pairs(result[task_prompt])
    
    def referring_expression_segmentation(self, text_input):
        task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
        result = self.__run_task(task_prompt, text_input=text_input)
        return self.__parse_polygon_paris(result[task_prompt])
    
    def region_to_segmentation(self, box):
        task_prompt = '<REGION_TO_SEGMENTATION>'
        image_width, image_height = self.image.size
        region = [int(box[0] / image_width * 999), int(box[1] / image_height * 999), int(box[2] / image_width * 999), int(box[3] / image_height * 999)]
        text_input = f"<loc_{region[0]}><loc_{region[1]}><loc_{region[2]}><loc_{region[3]}>"
        result = self.__run_task(task_prompt, text_input=text_input)
        return self.__parse_polygon_paris(result[task_prompt])
    
    def region_to_category(self, box):
        task_prompt = '<REGION_TO_CATEGORY>'
        text_input = self.__box_to_text(box)
        result = self.__run_task(task_prompt, text_input=text_input)
        return self.__remove_tags(result[task_prompt])
    
    def region_to_description(self, box):
        task_prompt = '<REGION_TO_DESCRIPTION>'
        text_input = self.__box_to_text(box)
        result = self.__run_task(task_prompt, text_input=text_input)
        return self.__remove_tags(result[task_prompt])
    
    def ocr_with_region(self):
        task_prompt = '<OCR_WITH_REGION>'
        result = self.__run_task(task_prompt)
        return self.__parse_quad_box_pairs(result[task_prompt])
    
    def segmentation_description(self, box):
        return {
            "segmentation": self.region_to_segmentation(box), 
            "description": self.region_to_description(box),
            "category": self.region_to_category(box)
        }
    
    def sgementation_description_all(self):
        results = self.dense_region_caption()
        for result in results:
            result['segment'] = self.region_to_segmentation(result['bbox'])
        return results
