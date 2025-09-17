import os
import numpy as np
import cv2
import torch
import onnxruntime as ort
from huggingface_hub import snapshot_download
from cog import BasePredictor, Input, Path
from segment_anything import sam_model_registry, SamPredictor

HF_REPO   = "alexa1610/vit-h-onnx"
PTH_FILE  = "sam_vit_h_4b8939.pth"
ONNX_FILE = "sam_onnx_example.onnx"
MODEL_TYPE = "vit_h"

def _providers():
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]

class Predictor(BasePredictor):
    def setup(self):
        # Télécharge tous les assets depuis HF (lit HF_TOKEN si repo privé/gated)
        repo_dir  = snapshot_download(repo_id=HF_REPO)
        ckpt_path = os.path.join(repo_dir, PTH_FILE)
        onnx_path = os.path.join(repo_dir, ONNX_FILE)

        # SAM (torch) pour embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=ckpt_path)
        sam.to(device=device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.device = device

        # ONNX Runtime (tête de prédiction)
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=_providers()
        )
        self.input_names  = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

        # Noms attendus par l’export SAM ONNX “classique”
        self.expected_inputs = [
            "image_embeddings", "point_coords", "point_labels",
            "mask_input", "has_mask_input", "orig_im_size"
        ]

    def predict(
        self,
        image: Path = Input(description="Image d'entrée (PNG/JPG)"),
        box: str = Input(description="Bounding box: x0,y0,x1,y1"),
    ) -> Path:
        # Lecture image
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise ValueError("Impossible de lire l'image.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Embeddings
        self.predictor.set_image(img_rgb)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()  # (1,256,64,64)

        # Parse la box
        try:
            x0, y0, x1, y1 = map(int, box.split(","))
        except Exception:
            raise ValueError("Format attendu pour 'box' : x0,y0,x1,y1")
        input_box = np.array([x0, y0, x1, y1], dtype=np.float32)
        coords = input_box.reshape(2, 2)[None, :, :]                   # (1,2,2)
        coords = self.predictor.transform.apply_coords(coords, img_rgb.shape[:2]).astype(np.float32)
        labels = np.array([[2, 3]], dtype=np.float32)

        # Prépare les entrées ONNX
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": coords,
            "point_labels": labels,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
            "orig_im_size": np.array(img_rgb.shape[:2], dtype=np.float32),  # (H,W)
        }

        # Exécution (None = tous les outputs dans l'ordre du graph)
        masks, iou_scores, low_res = self.sess.run(None, ort_inputs)

        # Binarisation
        mask = (masks[0, 0] > self.predictor.model.mask_threshold).astype(np.uint8) * 255

        out_path = "/tmp/mask.png"
        cv2.imwrite(out_path, mask)
        return Path(out_path)
