import onnxruntime as ort
import numpy as np
import cv2
import torch
from cog import BasePredictor, Input, Path
from segment_anything import sam_model_registry, SamPredictor


MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"   # ton checkpoint PyTorch
ONNX_PATH = "sam_onnx_example.onnx"        # ton modèle ONNX exporté


class Predictor(BasePredictor):
    def setup(self):
        """
        Chargé une seule fois au démarrage du conteneur
        """
        # Charger SAM en PyTorch juste pour faire les embeddings
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device="cpu")
        self.predictor = SamPredictor(sam)

        # Charger la session ONNX pour la prédiction
        self.ort_session = ort.InferenceSession(ONNX_PATH)

    def predict(
        self,
        image: Path = Input(description="Image d'entrée (PNG ou JPG)"),
        box: str = Input(description="Bounding box format: x0,y0,x1,y1"),
    ) -> Path:
        """
        Fait une prédiction avec SAM (ONNX + embeddings calculés en PyTorch)
        """
        # Lire image
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError("Impossible de lire l'image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Préparer SAM predictor pour embeddings
        self.predictor.set_image(img_rgb)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()

        # Parse la box
        x0, y0, x1, y1 = map(int, box.split(","))
        input_box = np.array([x0, y0, x1, y1])
        coords = input_box.reshape(2, 2)[None, :, :]
        coords = self.predictor.transform.apply_coords(coords, img_rgb.shape[:2]).astype(np.float32)

        labels = np.array([[2, 3]], dtype=np.float32)

        # Inputs ONNX
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": coords,
            "point_labels": labels,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.array(img_rgb.shape[:2], dtype=np.float32),
        }

        # Exécuter modèle
        masks, _, _ = self.ort_session.run(None, ort_inputs)

        # Binariser le masque
        mask = masks[0][0] > self.predictor.model.mask_threshold

        # Sauvegarde résultat
        out_path = "/tmp/mask.png"
        cv2.imwrite(out_path, (mask * 255).astype(np.uint8))

        return Path(out_path)

