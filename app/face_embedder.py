from typing import List

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from facenet_pytorch.models.mtcnn import fixed_image_standardization

from .utils.logging_helper import LoggingHelper


class FaceEmbedder:
    LOGGER = LoggingHelper("FaceEmbedder").logger
    """
    From an image, obtain embedding vector for each face.
    """

    def __init__(self, image_size: int = 160, margin: int = 0, min_face_size: int = 20,
                 thresholds: List[float] = [0.6, 0.7, 0.7], factor: float = 0.709, post_process: bool = True,
                 keep_all: bool = False, selection_method: str = None, embedding_model_name: str = 'vggface2'):
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            post_process=post_process,
            keep_all=keep_all,
            selection_method=selection_method,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.post_process = post_process

        self.resnet = InceptionResnetV1(pretrained=embedding_model_name).eval() \
            .to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    @staticmethod
    def cast_type(face, return_type: str = None, type=None, permute=True):
        if return_type == "np":
            if permute:
                face = face.permute(1, 2, 0)

            face = face.detach().cpu().numpy()
            if type:
                face = face.astype(type)
            return face
        elif return_type == "list":
            if permute:
                face = face.permute(1, 2, 0)
            face = face.detach().cpu().numpy()
            if type:
                face = face.astype(type)
            return face.tolist()
        elif return_type == "pt":
            return face
        else:
            return face

    def crop_image(self, image, return_type: str = "np", type="uint8", permute=False):
        boxes, probs = self.mtcnn.detect(image)
        if boxes is not None:
            list_faces = []
            for box in boxes:
                if self.post_process:
                    list_faces.append(fixed_image_standardization(extract_face(image, box)))
                else:
                    list_faces.append(extract_face(image, box))

            results = [
                {"face": self.cast_type(face, return_type, type, permute), "prob": probs[i], "box": boxes[i].tolist()}
                for i, face in enumerate(list_faces)]
            return results

    def embed_tensor_faces(self, tensor_faces, return_type="np", type=None, normalize=True):
        embeddings = self.resnet(tensor_faces)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return self.cast_type(embeddings, return_type, type, permute=False)

    def embed_faces(self, list_faces, return_type="np", type="float32", normalize=True):
        tensor_faces = [list_faces[i]['face'] for i, face in enumerate(list_faces)]
        tensor_faces = torch.stack(tensor_faces).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        return self.embed_tensor_faces(tensor_faces, return_type, type, normalize)

    def crop_and_embed(self, image, return_type="np", type=None, normalize=True):
        crops = self.crop_image(image, return_type="pt", type=None, permute=False)
        return self.embed_faces(crops, return_type, type, normalize), crops
