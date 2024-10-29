from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import requests
from sam2 import load_model
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from florence2_processor import Florence2Processor
from sam2_processor import SAM2Processor

processors = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processors
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # Load SAM2 model.
    processors['sam2'] = {}
    processors['sam2']['model'] = load_model(
        variant="large",
        ckpt_path="models/sam2/sam2_hiera_large.pt",
        device=device
    )
    processors['sam2']['processor'] = SAM2ImagePredictor(processors['sam2']['model'])

    # Load Florence-2 model.
    processors['florence'] = {}
    processors['florence']['model'] = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processors['florence']['processor'] = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    yield

    # Clean up
    processors.clear()



app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "Worlds"}

class SegmentRequest(BaseModel):
    image_url: str
    box: list

@app.post("/segment")
def segment(request: SegmentRequest):
    image = Image.open(requests.get(request.image_url, stream=True).raw)
    processor = SAM2Processor(processors["sam2"]["model"], processors["sam2"]["processor"], image)
    return processor.segment(request.box)

class FlorenceRequest(BaseModel):
    image_url: str
    text_input: str | None = None
    box: list | None = None

@app.post("/florence/{task}")
def florence(task: str, request: FlorenceRequest):
    result = None
    image = Image.open(requests.get(request.image_url, stream=True).raw)
    processor = Florence2Processor(processors["florence"]["model"], processors["florence"]["processor"], image)
    response = {"image": processor.get_image_size()}
    match task:
        case "caption":
            result = processor.caption()
        case "detailed_caption":
            result = processor.detailed_caption()
        case "more_detailed_caption":
            result = processor.more_detailed_caption()
        case "object-detection":
            result = processor.object_detection()
        case "dense-region-caption":
            result = processor.dense_region_caption()
        case "region-proposal":
            result = processor.region_proposal()
        case "phrase-grounding":
            result = processor.phrase_grounding(request.text_input)
        case "referring-expression-segmentation":
            result = processor.referring_expression_segmentation(request.text_input)
        case "region-to-segmentation":
            result = processor.region_to_segmentation(request.box)
        case "region-to-category":
            result = processor.region_to_category(request.box)
        case "region-to-description":
            result = processor.region_to_description(request.box)
        case "ocr-with-region":
            result = processor.ocr_with_region()
        case "segmentation-description":
            result = processor.segmentation_description(request.box)
        case "segmentation-description-all":
            result = processor.sgementation_description_all()
        case _:
            # return 404
            raise HTTPException(status_code=404, detail="Task not found")
    response['result'] = result
    return response

@app.post("/flosam/seg-cap-all")
def flosam(request: FlorenceRequest):
    image = Image.open(requests.get(request.image_url, stream=True).raw)
    flo_processor = Florence2Processor(processors["florence"]["model"], processors["florence"]["processor"], image)
    sam_processor = SAM2Processor(processors["sam2"]["model"], processors["sam2"]["processor"], image)
    response = {"image": flo_processor.get_image_size()}
    region_results = flo_processor.dense_region_caption()
    for region_result in region_results:
        box = region_result['bbox']
        seg_result = sam_processor.segment(box)
        region_result['segment'] = seg_result['segment']
    response['result'] = region_results
    return response

@app.post("/flosam/seg-cap")
def flosam(request: FlorenceRequest):
    image = Image.open(requests.get(request.image_url, stream=True).raw)
    flo_processor = Florence2Processor(processors["florence"]["model"], processors["florence"]["processor"], image)
    sam_processor = SAM2Processor(processors["sam2"]["model"], processors["sam2"]["processor"], image)
    response = {"image": flo_processor.get_image_size()}
    description = flo_processor.region_to_description(request.box)
    segmentation = sam_processor.segment(request.box)
    response['result'] = {"description": description, "segmentation": segmentation}
    return response
