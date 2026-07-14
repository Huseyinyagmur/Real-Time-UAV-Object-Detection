
from  __future__ import annotations
from pathlib import Path
from typing import Any
from ultralytics import YOLO
from core.errors import InferenceError

class YOLOInference:
    def __init__(
        self,
        model_path:str|Path,
        confidence:float,
        image_size:int,
        class_ids:list[int]|tuple[int,...])->None:
        
        self.model_path=Path(model_path)
        self.confidence=confidence
        self.image_size=image_size
        self.class_ids=tuple(class_ids)
        try:
            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            raise InferenceError(
                f"Model could not be loaded: {self.model_path}"
            ) from exc
    def predict(self,frame:object)->list[Any]:
        results=self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=list(self.class_ids),#tuple ı geçici olarak liste çeviriyoruz
            verbose=False
        )
        return results