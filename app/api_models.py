from typing import List

from pydantic import BaseModel


class Face(BaseModel):
    face: List[List[List[int]]]
    prob: float
    box: List[float]


class NearestNeighbour(BaseModel):
    nearest_neighbours_path: List[str]
    nearest_neighbours_distance: List[float]
    predicted_identity: str
