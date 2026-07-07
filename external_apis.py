"""
النداءات لخدمات الـ APIs الخارجية: PlantNet (step 1) و Roboflow (step 2).
"""

from typing import List, Tuple

import requests

from config import (
    PLANTNET_API_KEY,
    PLANTNET_PROJECT,
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_WORKFLOW_ID,
)


def check_is_tomato(image_bytes: bytes) -> Tuple[bool, str, float]:
    """Step 1: بيتأكد إن الصورة لنبات طماطم عن طريق PlantNet."""
    url = f"https://my-api.plantnet.org/v2/identify/{PLANTNET_PROJECT}?api-key={PLANTNET_API_KEY}"
    files = [("images", ("image.jpg", image_bytes, "image/jpeg"))]
    data = {"organs": ["leaf"]}

    response = requests.post(url, files=files, data=data, timeout=15)
    response.raise_for_status()

    result = response.json()
    if not result.get("results"):
        return False, "unknown", 0.0

    best_match = result["results"][0]
    sci_name = best_match["species"]["scientificNameWithoutAuthor"]
    common_names = best_match["species"].get("commonNames", [])
    score = best_match["score"]

    is_tomato = "solanum lycopersicum" in sci_name.lower() or any(
        "tomato" in n.lower() for n in common_names
    )
    return is_tomato, sci_name, score


def get_leaf_predictions(image_b64: str) -> List[dict]:
    """Step 2: بيبعت الصورة لـ Roboflow ويرجع قائمة كل الورق المكتشف (polygon points لكل ورقة)."""
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY,
    )
    result = client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_WORKFLOW_ID,
        images={"image": image_b64},
        parameters={"classes": "leaf_"},
        use_cache=True,
    )

    if isinstance(result, list):
        result = result[0]
    if "predictions" in result and isinstance(result["predictions"], dict):
        return result["predictions"].get("predictions", [])
    if "predictions" in result and isinstance(result["predictions"], list):
        return result["predictions"]
    return []
