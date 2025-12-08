"""
Image-Text Dataset classes for MicroVLM-E.
"""

import os
import json
import random
import logging
from typing import Dict, Any, Optional, List, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset class with common functionality."""

    def __init__(
        self,
        data_path: str,
        vis_processor: Optional[Callable] = None,
        text_processor: Optional[Callable] = None,
    ):
        self.data_path = data_path
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.annotation = []

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def load_image(self, image_path: str) -> Image.Image:
        """Load and process an image."""
        full_path = os.path.join(self.data_path, image_path)
        image = Image.open(full_path).convert("RGB")
        return image

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Apply visual processor to image."""
        if self.vis_processor is not None:
            image = self.vis_processor(image)
        return image

    def process_text(self, text: str) -> str:
        """Apply text processor to text."""
        if self.text_processor is not None:
            text = self.text_processor(text)
        return text


class ImageTextPairDataset(BaseDataset):
    """
    Dataset for image-text pairs (e.g., LAION, CC3M, SBU).

    Expects data in one of the following formats:
    1. JSON file with list of {"image": path, "caption": text}
    2. Directory structure with images and corresponding .txt files
    """

    def __init__(
        self,
        data_path: str,
        vis_processor: Optional[Callable] = None,
        text_processor: Optional[Callable] = None,
        ann_file: Optional[str] = None,
    ):
        super().__init__(data_path, vis_processor, text_processor)

        self.annotation = []

        # Try to load annotation file
        if ann_file is not None:
            ann_path = os.path.join(data_path, ann_file)
        else:
            ann_path = os.path.join(data_path, "annotations.json")

        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                self.annotation = json.load(f)
        else:
            # Try to build from directory structure
            self._build_from_directory()

        logging.info(f"Loaded {len(self.annotation)} image-text pairs from {data_path}")

    def _build_from_directory(self):
        """Build annotation from directory structure."""
        image_dir = os.path.join(self.data_path, "images")
        if not os.path.exists(image_dir):
            image_dir = self.data_path

        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_path = os.path.join("images", filename) if os.path.exists(os.path.join(self.data_path, "images")) else filename

                # Look for caption file
                base_name = os.path.splitext(filename)[0]
                caption_file = os.path.join(self.data_path, "captions", f"{base_name}.txt")

                if os.path.exists(caption_file):
                    with open(caption_file, "r") as f:
                        caption = f.read().strip()
                    self.annotation.append({
                        "image": image_path,
                        "caption": caption,
                    })

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ann = self.annotation[index]

        # Load and process image
        image = self.load_image(ann["image"])
        image = self.process_image(image)

        # Process caption
        caption = self.process_text(ann.get("caption", ""))

        return {
            "image": image,
            "text_input": caption,
            "text_output": caption,
        }


class CaptionDataset(BaseDataset):
    """
    Dataset for image captioning (e.g., COCO, Flickr30k).

    Expects COCO-style annotation format.
    """

    def __init__(
        self,
        data_path: str,
        ann_file: str,
        vis_processor: Optional[Callable] = None,
        text_processor: Optional[Callable] = None,
        split: str = "train",
    ):
        super().__init__(data_path, vis_processor, text_processor)

        self.split = split

        # Load annotations
        ann_path = os.path.join(data_path, ann_file)
        with open(ann_path, "r") as f:
            ann_data = json.load(f)

        # Build image id to filename mapping
        self.image_id_to_file = {}
        if "images" in ann_data:
            for img_info in ann_data["images"]:
                self.image_id_to_file[img_info["id"]] = img_info["file_name"]

        # Load annotations
        self.annotation = []
        if "annotations" in ann_data:
            for ann in ann_data["annotations"]:
                image_id = ann["image_id"]
                if image_id in self.image_id_to_file:
                    self.annotation.append({
                        "image": self.image_id_to_file[image_id],
                        "caption": ann["caption"],
                        "image_id": image_id,
                    })

        logging.info(f"Loaded {len(self.annotation)} captions from {ann_path}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ann = self.annotation[index]

        # Determine image directory based on split
        if "train" in self.split:
            image_dir = "train2017"
        elif "val" in self.split:
            image_dir = "val2017"
        else:
            image_dir = "images"

        # Load image
        image_path = os.path.join(image_dir, ann["image"])
        image = self.load_image(image_path)
        image = self.process_image(image)

        # Process caption
        caption = self.process_text(ann["caption"])

        return {
            "image": image,
            "text_input": "Describe this image.",
            "text_output": caption,
            "image_id": ann.get("image_id", index),
        }


class VQADataset(BaseDataset):
    """
    Dataset for Visual Question Answering (e.g., VQAv2, OK-VQA, GQA).

    Expects annotation with questions and answers.
    """

    def __init__(
        self,
        data_path: str,
        vis_processor: Optional[Callable] = None,
        text_processor: Optional[Callable] = None,
        split: str = "train",
    ):
        super().__init__(data_path, vis_processor, text_processor)

        self.split = split

        # Load annotations
        questions_file = os.path.join(data_path, f"v2_OpenEnded_mscoco_{split}2014_questions.json")
        annotations_file = os.path.join(data_path, f"v2_mscoco_{split}2014_annotations.json")

        # Alternative file patterns
        if not os.path.exists(questions_file):
            questions_file = os.path.join(data_path, "questions.json")
        if not os.path.exists(annotations_file):
            annotations_file = os.path.join(data_path, "annotations.json")

        # Load questions
        questions = {}
        if os.path.exists(questions_file):
            with open(questions_file, "r") as f:
                q_data = json.load(f)
                for q in q_data.get("questions", []):
                    questions[q["question_id"]] = q

        # Load annotations
        self.annotation = []
        if os.path.exists(annotations_file):
            with open(annotations_file, "r") as f:
                ann_data = json.load(f)
                for ann in ann_data.get("annotations", []):
                    q_id = ann["question_id"]
                    if q_id in questions:
                        self.annotation.append({
                            "image_id": ann["image_id"],
                            "question": questions[q_id]["question"],
                            "answer": ann["multiple_choice_answer"] if "multiple_choice_answer" in ann else ann.get("answers", [{}])[0].get("answer", ""),
                        })

        logging.info(f"Loaded {len(self.annotation)} VQA samples from {data_path}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ann = self.annotation[index]

        # Build image path
        image_id = ann["image_id"]
        image_name = f"COCO_{self.split}2014_{image_id:012d}.jpg"

        if self.split == "train":
            image_path = os.path.join("train2014", image_name)
        else:
            image_path = os.path.join("val2014", image_name)

        # Load image
        try:
            image = self.load_image(image_path)
        except FileNotFoundError:
            # Try alternative path
            image_path = f"{image_id:012d}.jpg"
            image = self.load_image(image_path)

        image = self.process_image(image)

        # Process question and answer
        question = self.process_text(ann["question"])
        answer = self.process_text(ann["answer"])

        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
            "question_id": ann.get("question_id", index),
        }


class InstructionDataset(BaseDataset):
    """
    Dataset for instruction tuning (e.g., LLaVA-Instruct, multi-task).

    Expects annotation with conversations or instruction-response pairs.
    """

    def __init__(
        self,
        data_path: str,
        vis_processor: Optional[Callable] = None,
        text_processor: Optional[Callable] = None,
        ann_file: Optional[str] = None,
    ):
        super().__init__(data_path, vis_processor, text_processor)

        # Load annotations
        if ann_file is not None:
            ann_path = os.path.join(data_path, ann_file)
        else:
            ann_path = os.path.join(data_path, "instructions.json")

        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                self.annotation = json.load(f)
        else:
            self.annotation = []

        logging.info(f"Loaded {len(self.annotation)} instruction samples from {data_path}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ann = self.annotation[index]

        # Load image
        image_path = ann.get("image", ann.get("image_path", ""))
        if image_path:
            image = self.load_image(image_path)
            image = self.process_image(image)
        else:
            image = None

        # Handle conversation format
        if "conversations" in ann:
            convs = ann["conversations"]
            # Extract first human/assistant exchange
            instruction = ""
            response = ""
            for conv in convs:
                role = conv.get("from", conv.get("role", ""))
                content = conv.get("value", conv.get("content", ""))
                if role in ["human", "user"]:
                    instruction = content
                elif role in ["gpt", "assistant"]:
                    response = content
                    break
        else:
            instruction = ann.get("instruction", ann.get("question", ""))
            response = ann.get("response", ann.get("answer", ""))

        # Process text
        instruction = self.process_text(instruction)
        response = self.process_text(response)

        result = {
            "text_input": instruction,
            "text_output": response,
        }

        if image is not None:
            result["image"] = image

        return result


class RefCOCODataset(BaseDataset):
    """
    Dataset for referring expression comprehension (RefCOCO, RefCOCO+, RefCOCOg).
    """

    def __init__(
        self,
        data_path: str,
        vis_processor: Optional[Callable] = None,
        text_processor: Optional[Callable] = None,
        split: str = "train",
    ):
        super().__init__(data_path, vis_processor, text_processor)

        self.split = split

        # Load annotations
        ann_file = os.path.join(data_path, f"refs({split}).json")
        if os.path.exists(ann_file):
            with open(ann_file, "r") as f:
                self.annotation = json.load(f)
        else:
            # Try alternative format
            ann_file = os.path.join(data_path, "annotations.json")
            if os.path.exists(ann_file):
                with open(ann_file, "r") as f:
                    all_ann = json.load(f)
                    self.annotation = [a for a in all_ann if a.get("split", "") == split]
            else:
                self.annotation = []

        logging.info(f"Loaded {len(self.annotation)} RefCOCO samples from {data_path}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ann = self.annotation[index]

        # Load image
        image_id = ann.get("image_id", ann.get("id", 0))
        image_path = f"COCO_train2014_{image_id:012d}.jpg"

        try:
            image = self.load_image(os.path.join("train2014", image_path))
        except FileNotFoundError:
            image = self.load_image(image_path)

        image = self.process_image(image)

        # Get referring expression and bounding box
        expression = ann.get("sentences", [{}])[0].get("raw", ann.get("expression", ""))
        bbox = ann.get("bbox", [0, 0, 0, 0])

        # Format output as coordinate string
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        bbox_str = f"[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]"

        return {
            "image": image,
            "text_input": f"Find: {expression}",
            "text_output": bbox_str,
        }

