import os, cv2, numpy as np, yaml
import torch, onnxruntime as ort
from huggingface_hub import snapshot_download, hf_hub_download
from cog import BasePredictor, Input, Path
from segment_anything import sam_model_registry, SamPredictor

HF_REPO    = "alexa1610/vit-h-onnx"
SAM_PTH    = "sam_vit_h_4b8939.pth"
SAM_ONNX   = "sam_onnx_example.onnx"
YOLO_ONNX  = "yolov11_kneeCT.onnx"
YOLO_DATA  = "yolo_classes.yaml"   
MODEL_TYPE = "vit_h"

def _providers():
    return ["CUDAExecutionProvider"]  # forcé GPU


class Predictor(BasePredictor):
    def setup(self):
        repo_dir  = snapshot_download(repo_id=HF_REPO)
        sam_ckpt  = os.path.join(repo_dir, SAM_PTH)
        sam_onnx  = os.path.join(repo_dir, SAM_ONNX)
        yolo_onnx = os.path.join(repo_dir, YOLO_ONNX)
        data_yaml = os.path.join(repo_dir, YOLO_DATA)

        # SAM Torch (embeddings)
        sam = sam_model_registry[MODEL_TYPE](checkpoint=sam_ckpt)
        sam.to(device="cuda")
        sam.eval()
        self.predictor = SamPredictor(sam)

        # SAM ONNX
        self.sam_sess = ort.InferenceSession(sam_onnx, providers=_providers())

        # YOLO ONNX
        self.yolo_sess = ort.InferenceSession(yolo_onnx, providers=_providers())
        self.yolo_input = self.yolo_sess.get_inputs()[0].name
        self.yolo_output = [o.name for o in self.yolo_sess.get_outputs()]

        # Charger les classes depuis data.yaml
        with open(data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        self.class_names = data_cfg["names"]

        self.class_colors = {
            0: (0, 255, 0),   # tibia = vert
            1: (255, 0, 0),   # femur = bleu
            2: (0, 0, 255),   # prothèse = rouge
            3: (255, 255, 0), # fibula = jaune
            4: (255, 0, 255)  # patella = magenta
        }

        # Prothèse > reste (si existe)
        self.class_priority = {
            i: (10 if "prosthesis" in name.lower() else 1)
            for i, name in self.class_names.items()
        }

    def predict(
        self,
        images: list[Path] = Input(description="Liste d'images CT (PNG/JPG)"),
    ) -> Path:
        images = sorted(images, key=lambda x: str(x))
        frames = []

        for image in images:
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # YOLO
            boxes = self.run_yolo(img_rgb)

            # SAM
            overlay = np.zeros_like(img_rgb, dtype=np.uint8)
            self.predictor.set_image(img_rgb)
            image_embedding = self.predictor.get_image_embedding().cpu().numpy()

            mask_layers = []
            for box, cls in boxes:
                sam_mask = self.run_sam(img_rgb, image_embedding, box)
                if sam_mask is not None:
                    mask_layers.append((
                        sam_mask,
                        self.class_colors.get(cls, (255,255,255)),
                        self.class_priority.get(cls, 1)
                    ))

            combined = self.fuse_masks(img_rgb, mask_layers)
            frames.append(combined)

        out_path = "/tmp/segmentation.mp4"
        self.save_video(frames, out_path)
        return Path(out_path)

    def run_yolo(self, img_rgb):
        h, w = img_rgb.shape[:2]
        img_resized = cv2.resize(img_rgb, (640, 640))
        inp = img_resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        preds = self.yolo_sess.run(self.yolo_output, {self.yolo_input: inp})[0]

        boxes = []
        for det in preds:  # (x,y,w,h,conf,cls)
            x, y, bw, bh, conf, cls = det
            if conf < 0.4:
                continue
            x0 = int((x - bw/2) * w / 640)
            y0 = int((y - bh/2) * h / 640)
            x1 = int((x + bw/2) * w / 640)
            y1 = int((y + bh/2) * h / 640)
            boxes.append(((x0, y0, x1, y1), int(cls)))
        return boxes

    def run_sam(self, img_rgb, image_embedding, box):
        x0, y0, x1, y1 = box
        coords = np.array([[ [x0,y0], [x1,y1] ]], dtype=np.float32)
        coords = self.predictor.transform.apply_coords(coords, img_rgb.shape[:2]).astype(np.float32)
        labels = np.array([[2, 3]], dtype=np.float32)
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": coords,
            "point_labels": labels,
            "mask_input": np.zeros((1,1,256,256), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
            "orig_im_size": np.array(img_rgb.shape[:2], dtype=np.float32),
        }
        try:
            masks, _, _ = self.sam_sess.run(None, ort_inputs)
            mask = (masks[0, 0] > self.predictor.model.mask_threshold).astype(np.uint8)
            return mask
        except Exception:
            return None

    def fuse_masks(self, img_rgb, mask_layers):
        h, w, _ = img_rgb.shape
        label_map = np.zeros((h, w), dtype=np.int32)

        # trier par priorité (les plus importants en dernier)
        mask_layers.sort(key=lambda x: x[2])

        for mask, color, priority, cls_id in mask_layers:
            label_map[mask > 0] = cls_id + 1  # +1 pour garder 0 = background

        out_img = np.zeros_like(img_rgb)
        for cid, name in self.class_names.items():
            col = self.class_colors.get(cid, (255,255,255))
            out_img[label_map == cid + 1] = col

        return out_img



    def save_video(self, frames, path):
        if not frames:
            return
        h, w, _ = frames[0].shape
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
