import csv
from typing import Union, List, Tuple, Dict

from fastapi import FastAPI, Body
import uvicorn
from model_utils import load_model
from nemo.collections.nlp.models.dialogue.dialogue_zero_shot_slot_filling_model import DialogueZeroShotSlotFillingModel
from pydantic import BaseModel
import logging

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


class Query(BaseModel):
    text: str = None
    entity_types: List[str] = None
    entity_descriptions: List[str] = None


class ZeroShotResponse(BaseModel):
    utterance_tokens: List[str] = None
    entities_dict: Dict[int, List[Tuple[int, int]]] = None
    slot_types: List[str] = None


model: DialogueZeroShotSlotFillingModel = load_model()
app = FastAPI()


@app.get("/")
def read_root():
    logging.info('root')
    return {"Hello": "World"}


@app.post("/label/")
def predict(query: Query):
    types_descriptions = [_type + '\t' + _description
                          for _type, _description in zip(query.entity_types, query.entity_descriptions)]

    slot_class, iob_slot_class = model.predict(query.text, types_descriptions)
    slot_class, iob_slot_class = slot_class[0][1:-1], iob_slot_class[0][1:-1]
    utterance_tokens, slot_class, iob_slot_class = model.merge_subword_tokens_and_slots(query.text, slot_class, iob_slot_class)
    entities_dict = model.get_continuous_slots(slot_class, utterance_tokens)

    logging.info(entities_dict)
    types = query.entity_types

    if not types_descriptions:
        reader = csv.reader(model.slot_descriptions, delimiter="\t")
        types, descriptions = zip(*reader)
    return ZeroShotResponse(utterance_tokens=utterance_tokens, entities_dict=entities_dict, slot_types=types)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level='debug')
