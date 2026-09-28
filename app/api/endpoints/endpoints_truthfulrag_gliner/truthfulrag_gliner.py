from fastapi import APIRouter, Depends, Response, HTTPException
from app.services.setup_service import get_selected_item_id
from app.services.truthfulrag_gliner_service import TruthfulRAGGlinerService, get_truthfulrag_gliner_service

import textwrap
import ast

router = APIRouter()


@router.get("/filtered-chunks")
def gliner_filtered_chunks(item_id: str = Depends(get_selected_item_id),
                           truthfulrag_service: TruthfulRAGGlinerService = Depends(
                               get_truthfulrag_gliner_service)):
    filtered_chunks = truthfulrag_service.get_filtered_chunks(item_id)
    if filtered_chunks is None:
        raise HTTPException(status_code=404, detail="Item not found")

    chunk_lines = ""
    for chunk in filtered_chunks:
        chunk_tokens, chunk_content, chunk_id = chunk["tokens"], chunk["context"], chunk["chunk_id"]
        chunk_content = textwrap.fill(chunk_content, width=100)

        chunk_lines += f"TOKENS:\n{chunk_tokens}\n\nCHUNK_ID:\n{chunk_id}\n\nCONTENT:\n{chunk_content}\n\n---\n\n"

    content = (f"## FILTERED CHUNKS\n\n"
               f"{chunk_lines}")

    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/extracted-entities")
def gliner_extracted_entities(item_id: str = Depends(get_selected_item_id),
                              truthfulrag_service: TruthfulRAGGlinerService = Depends(
                                  get_truthfulrag_gliner_service)):
    extracted_entities = truthfulrag_service.get_extracted_entities(item_id)
    if extracted_entities is None:
        raise HTTPException(status_code=404, detail="Extracted entities not found")

    entities_lines = ""
    for e_key in extracted_entities:
        e = extracted_entities[e_key]
        entities_lines += f"ENTITY_NAME:\n{e_key}\n\nENTITY_TYPE:\n{e['entity_type']}\n\nENTITY_DESCRIPTION:\n{e['description']}\n\nGLINER_CONFIDENCE:\n{e['confidence']}\n\n---\n\n"

    content = (f"## EXTRACTED ENTITIES\n\n"
               f"{entities_lines}")

    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/extracted-relations")
def gliner_extracted_relations(item_id: str = Depends(get_selected_item_id),
                               truthfulrag_service: TruthfulRAGGlinerService = Depends(
                                   get_truthfulrag_gliner_service)):
    extracted_relations = truthfulrag_service.get_extracted_relations(item_id)
    if extracted_relations is None:
        raise HTTPException(status_code=404, detail="Extracted relations not found")

    relations_lines = ""
    for relation_dict in extracted_relations:
        relation_dict_key = next(iter(relation_dict))

        relation_dict_content = relation_dict[relation_dict_key]
        subject, object_ = ast.literal_eval(relation_dict_key)
        keywords = ", ".join(relation_dict_content['keywords'].split("<SEP>"))

        relations_lines += f"SRC_ID:\n{subject}\n\nTGT_ID:\n{object_}\n\nRELATION_DESCRIPTION:\n{relation_dict_content['description']}\n\nRELATION_KEYWORDS:\n{keywords}\n\nGLINER_WEIGHT:\n{relation_dict_content['weight']}\n\n---\n\n"

    content = (f"## EXTRACTED RELATIONS\n\n"
               f"{relations_lines}")

    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/view-extracted-paths")
def gliner_view_extracted_paths(item_id: str = Depends(get_selected_item_id),
                                truthfulrag_service: TruthfulRAGGlinerService = Depends(
                                    get_truthfulrag_gliner_service)):
    retrieved_paths = truthfulrag_service.get_paths(item_id)
    if retrieved_paths is None:
        raise HTTPException(status_code=404, detail="Extracted relations not found")

    path_elements, path_filtered_elements = retrieved_paths["elements"], retrieved_paths["filtered_elements"]

    path_lines = ""
    for path in path_elements:
        path_lines += f"{path}\n\n---\n\n"

    path_lines += f"## FILTERED PATHS\n\n"

    for path in path_filtered_elements:
        path_lines += f"{path['element']}\n\nENTROPY_DELTA:\n\n{path['delta']}\n\n---\n\n"

    content = (f"## EXTRACTED PATHS\n\n"
               f"{path_lines}")

    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/response-wo-context")
def gliner_response_without_context(item_id: str = Depends(get_selected_item_id),
                                    truthfulrag_service: TruthfulRAGGlinerService = Depends(
                                        get_truthfulrag_gliner_service)):
    response = truthfulrag_service.get_response_without_context(item_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Item not found")

    content = (f"## PREDICTED ANSWER\n\n"
               f"{response['predicted_answer']}\n\n"
               f"---\n\n"
               f"## REASONING\n\n"
               f"{response['reasoning']}\n\n"
               f"---\n\n"
               f"## EVALUATION\n\n"
               f"* ACCURACY: {response['acc']}\n\n"
               f"* F1: {response['f1']}\n\n"
               f"---\n\n")

    return Response(content=content, status_code=200, media_type="text/markdown")


@router.get("/response-with-context")
def gliner_response_with_context(item_id: str = Depends(get_selected_item_id),
                                 truthfulrag_service: TruthfulRAGGlinerService = Depends(
                                     get_truthfulrag_gliner_service)):
    response = truthfulrag_service.get_response_with_context(item_id)
    if response is None:
        raise HTTPException(status_code=404, detail="Item not found")

    content = (f"## PREDICTED ANSWER\n\n"
               f"{response['predicted_answer']}\n\n"
               f"---\n\n"
               f"## REASONING\n\n"
               f"{response['reasoning']}\n\n"
               f"---\n\n"
               f"## EVALUATION\n\n"
               f"* ACCURACY: {response['acc']}\n\n"
               f"* F1: {response['f1']}\n\n"
               f"---\n\n")

    return Response(content=content, status_code=200, media_type="text/markdown")
