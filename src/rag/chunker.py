import uuid

import spacy

from rag.config import settings
from rag.schemas import ChunkMetadata, DocumentChunk

nlp = spacy.load(settings.spacy_model)


def chunker_fixed_size(
    document: str, filename: str, document_id: str, collection: str, size: int, overlap: int
) -> list[DocumentChunk]:

    result = []
    n_id = 0
    char_start = 0

    while len(document) > 0:
        if document[:size].strip() == "":
            document = document[size - overlap :]
            char_start = char_start + size - overlap
            continue
        result.append(
            DocumentChunk(
                chunk_id=uuid.uuid4(),
                document_id=document_id,
                collection=collection,
                text=document[:size],
                chunk_index=n_id,
                chunk_metadata=ChunkMetadata(
                    source_file=filename,
                    page_number=None,
                    char_start=char_start,
                    char_end=char_start + len(document[:size]) - 1,
                ),
            )
        )
        if len(document[size - overlap:]) <= overlap:
            break
        document = document[size - overlap :]
        n_id += 1
        char_start = char_start + size - overlap

    return result


def chunker_sentence_aware(
    document: str, filename: str, document_id: str, collection: str, size: int, overlap: int
) -> list[DocumentChunk]:

    individual_sentences = [sent for sent in nlp(document).sents]
    result = []
    n_id = 0

    while len(individual_sentences) > 0:
        chunk = individual_sentences[:size]

        result.append(
            DocumentChunk(
                chunk_id=uuid.uuid4(),
                document_id=document_id,
                collection=collection,
                text=" ".join(x.text for x in chunk),
                chunk_index=n_id,
                chunk_metadata=ChunkMetadata(
                    source_file=filename,
                    page_number=None,
                    char_start=chunk[0].start_char,
                    char_end=chunk[-1].end_char - 1,
                ),
            )
        )
        n_id += 1
        individual_sentences = individual_sentences[size - overlap :]

    return result


def chunker_recursive(
    document: str,
    filename: str,
    document_id: str,
    collection: str,
    size: int,
    overlap: int,
    recursive_order: list[str] | None = None,
    n_id: int = 0,
    char_start: int = 0,
) -> list[DocumentChunk]:

    if not recursive_order:
        return chunker_fixed_size(document, filename, document_id, collection, size, overlap)

    result = []
    splitter = recursive_order[0]
    blocks = document.split(splitter)
    cursor = 0

    for block in blocks:
        doc_find = document.find(block, cursor)
        block_start = char_start + doc_find
        cursor = doc_find + len(block)

        if len(block) > size:
            rec_result = chunker_recursive(
                block,
                filename,
                document_id,
                collection,
                size,
                overlap,
                recursive_order[1:],
                n_id,
                block_start,
            )
            result.extend(rec_result)
            n_id += len(rec_result)
        elif block.strip() != "":
            result.append(
                DocumentChunk(
                    chunk_id=uuid.uuid4(),
                    document_id=document_id,
                    collection=collection,
                    text=block,
                    chunk_index=n_id,
                    chunk_metadata=ChunkMetadata(
                        source_file=filename,
                        page_number=None,
                        char_start=block_start,
                        char_end=block_start + len(block) - 1,
                    ),
                )
            )
            n_id += 1

    return result


if __name__ == "__main__":
    print("/-=" * 20)

    for chunk in chunker_recursive(
        document="""Hypertrophy Training Frequency

Training each muscle group two to three times per week is often more effective than a traditional bro-split for natural lifters. When volume is matched, higher frequency allows you to distribute hard sets across sessions instead of cramming them into one long workout. That usually means better performance on each set and more consistent progressive overload over time.

Protein intake during a cut matters as much as training. Most research suggests roughly 1.6 to 2.2 grams of protein per kilogram of body weight daily when calories are restricted. Going higher than that may help some people preserve lean mass, but returns diminish quickly once you are already near the upper end of that range.

Recovery is the constraint most programs ignore. Sleep, stress, and total weekly volume all change how many hard sets you can recover from. If performance drops for two consecutive sessions on the same movement, reducing volume or frequency is usually smarter than adding more exercises.""",
        filename="Default Lorem Ipsum",
        document_id="random_id_123",
        collection="test",
        size=200,
        overlap=20,
        recursive_order=["\n\n", "\n", "."],
    ):
        print(chunk)
        print("&" * 20)
