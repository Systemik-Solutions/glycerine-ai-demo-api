from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image
import requests
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches  

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer

def plot_bbox(image, data=None, title=None):
   # Create a figure and axes  
    fig, ax = plt.subplots(figsize=(15, 15))  
      
    # Display the image  
    ax.imshow(image)  

    if data is not None:
        # Plot each bounding box  
        for bbox, label in zip(data['bboxes'], data['labels']):  
            # Unpack the bounding box coordinates  
            x1, y1, x2, y2 = bbox  
            # Create a Rectangle patch  
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
            # Add the rectangle to the Axes  
            ax.add_patch(rect)  
            # Annotate the label  
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  

    if title is not None:
        plt.title(title, wrap=True)  
    
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.show()  

def plot_polygons(image, data=None, bounding_box=None, title=None):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)
    # Plot the bounding box
    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    if data is not None:
        for polygons, label in zip(data['polygons'], data['labels']):
            for polygon in polygons:
                polygon = np.array(polygon).reshape(-1, 2)
                if len(polygon) < 3:  
                    print('Invalid polygon:', polygon)  
                    continue
                ax.add_patch(patches.Polygon(polygon, fill=False, edgecolor='yellow', linewidth=1))

    if title is not None:
        plt.title(title, wrap=True)

    ax.axis('off')
    plt.show()

def convert_ocr_data(ocr_data):
    quad_boxes = ocr_data['quad_boxes']
    # Convert the quad boxes to bounding boxes
    bboxes = []
    for quad_box in quad_boxes:
        bboxes.append([quad_box[0], quad_box[1], quad_box[4], quad_box[5]])
    return {'bboxes': bboxes, 'labels': ocr_data['labels']}

url = "https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HM54MGYC39W9MZ32YD4GJ59D/full/1024,/0/default.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image_width, image_height = image.size

# Caption.
task_prompt = '<CAPTION>'
results = run_example(task_prompt)
print(results)
plot_bbox(image, title=f"Caption: {results['<CAPTION>']}")

# Detailed Cpation.
task_prompt = '<DETAILED_CAPTION>'
results = run_example(task_prompt)
print(results)
plot_bbox(image, title=f"Detailed Caption: {results['<DETAILED_CAPTION>']}")

# More detailed caption.
task_prompt = '<MORE_DETAILED_CAPTION>'
results = run_example(task_prompt)
print(results)
plot_bbox(image, title=f"More Detailed Caption: {results['<MORE_DETAILED_CAPTION>']}")

# Object detection.
task_prompt = '<OD>'
results = run_example(task_prompt)
print(results)
plot_bbox(image, results['<OD>'], "Object Detection")

# Dense region caption.
task_prompt = '<DENSE_REGION_CAPTION>'
results = run_example(task_prompt)
print(results)
plot_bbox(image, results['<DENSE_REGION_CAPTION>'], "Dense Region Caption")

# Region proposal.
task_prompt = '<REGION_PROPOSAL>'
results = run_example(task_prompt)
print(results)
plot_bbox(image, results['<REGION_PROPOSAL>'], "Region Proposal")

# Phrase grounding.
task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
prompt = 'Buddha under a flowering tree crown'
results = run_example(task_prompt, text_input=prompt)
print(results)
plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'], f"Phrase Grounding. Prompt: {prompt}")

# Referring expression segmentation
task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
prompt = 'robe'
results = run_example(task_prompt, text_input=prompt)
print(results)
plot_polygons(image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], title=f"Referring Expression Segmentation. Prompt: {prompt}")

# Region to segmentation
task_prompt = '<REGION_TO_SEGMENTATION>'
box = [173, 389, 383, 845]
# Scale the region to range 0-999
region = [int(box[0] / image_width * 999), int(box[1] / image_height * 999), int(box[2] / image_width * 999), int(box[3] / image_height * 999)]
text_input = f"<loc_{region[0]}><loc_{region[1]}><loc_{region[2]}><loc_{region[3]}>"
results = run_example(task_prompt, text_input=text_input)
print(results)
plot_polygons(image, results['<REGION_TO_SEGMENTATION>'], bounding_box=box, title="Region to Segmentation")

# Region to category
task_prompt = '<REGION_TO_CATEGORY>'
results = run_example(task_prompt, text_input=text_input)
print(results)
plot_polygons(image, bounding_box=box, title=f"Region to Category: {results['<REGION_TO_CATEGORY>']}")

# Region to description
task_prompt = '<REGION_TO_DESCRIPTION>'
results = run_example(task_prompt, text_input=text_input)
print(results)
plot_polygons(image, bounding_box=box, title=f"Region to Description: {results['<REGION_TO_DESCRIPTION>']}")

# OCR
url = 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01J8NTQYRHSB098G2ZBQ8QF3XH/full/1024,/0/default.jpg';
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
task_prompt = '<OCR_WITH_REGION>'
results = run_example(task_prompt)
print(results)
ocr_data = convert_ocr_data(results['<OCR_WITH_REGION>'])
plot_bbox(image, ocr_data, title="OCR with Region")
