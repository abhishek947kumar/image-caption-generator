"""
Pre-configured sample images and reference captions for the Image Caption Generator demo.
Mirrors the Flickr8k 5-caption reference format for evaluation.
"""
from typing import List, Dict, Optional

SAMPLE_IMAGES: List[Dict] = [
    {
        "id": "dog_frisbee",
        "title": "Dog Playing Frisbee",
        "filename": "dog_frisbee.jpg",
        "category": "Animals / Action",
        "tags": ["dog", "retriever", "frisbee", "grass", "park", "jumping"],
        "references": [
            "A golden retriever jumping high in the air to catch a red frisbee on green grass",
            "A playful golden dog leaping midair with its mouth open catching a toy disc in a sunny park",
            "A golden retriever catches a red frisbee in a lush grassy field on a bright day",
            "A dog jumps through the air trying to grab a flying red frisbee in a park",
            "A joyful dog in mid flight catching a red toy over the grass"
        ],
        "primary_concepts": ["dog", "golden", "retriever", "frisbee", "red", "jumping", "grass", "park"]
    },
    {
        "id": "mountain_hiker",
        "title": "Mountain Summit Hiker",
        "filename": "mountain_hiker.jpg",
        "category": "Adventure / Nature",
        "tags": ["hiker", "mountain", "summit", "sunset", "clouds", "backpack"],
        "references": [
            "A hiker with a backpack stands on a rocky mountain summit with arms outstretched at sunset",
            "A person with hiking gear looking out over mountain peaks during a dramatic golden sunset",
            "A hiker standing triumphantly on top of a rocky cliff watching the sunset over misty mountains",
            "A traveler with a large backpack admiring the sun setting behind mountain ranges",
            "A person standing on a high mountain rock with arms spread wide in front of the clouds"
        ],
        "primary_concepts": ["hiker", "person", "mountain", "summit", "rocky", "sunset", "backpack", "clouds", "standing"]
    },
    {
        "id": "cyclist_autumn",
        "title": "Cyclist in Autumn Park",
        "filename": "cyclist_autumn.jpg",
        "category": "City / Lifestyle",
        "tags": ["cyclist", "bicycle", "autumn", "park", "yellow jacket", "leaves"],
        "references": [
            "A woman in a yellow jacket riding a bicycle on a paved path through an autumn park",
            "A smiling young female cyclist wearing a helmet riding a bike covered with fallen leaves",
            "A woman riding a bicycle with a front basket along a curving road with golden trees",
            "A cyclist wearing a yellow raincoat and helmet pedaling through a park in the fall",
            "A cheerful person riding a bike on a wet asphalt pathway surrounded by yellow foliage"
        ],
        "primary_concepts": ["woman", "cyclist", "riding", "bicycle", "bike", "yellow", "jacket", "autumn", "park", "leaves"]
    },
    {
        "id": "child_blocks",
        "title": "Toddler Stacking Blocks",
        "filename": "child_blocks.jpg",
        "category": "Children / Indoor",
        "tags": ["child", "toddler", "blocks", "playroom", "toys", "carpet"],
        "references": [
            "A young toddler sitting on a colorful playroom rug enthusiastically stacking wooden blocks",
            "A smiling little boy in a striped shirt building a tall colorful tower of blocks",
            "A small child playing with colorful wooden building blocks on a carpeted floor",
            "A toddler happily balancing wooden toy blocks in a bright indoor playroom",
            "A young child seated on a patterned rug stacking colorful toy bricks"
        ],
        "primary_concepts": ["child", "toddler", "sitting", "playing", "blocks", "wooden", "colorful", "rug", "playroom", "stacking"]
    }
]

def get_sample_by_id(sample_id: str) -> Optional[Dict]:
    for sample in SAMPLE_IMAGES:
        if sample["id"] == sample_id or sample["filename"] == sample_id:
            return sample
    return None
