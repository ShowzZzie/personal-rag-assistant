from rag.schemas import DocumentChunk, ChunkMetadata
import uuid
import spacy

nlp = spacy.load("en_core_web_sm")

def chunker_fixed_size(
    document: str,
    filename: str,
    document_id: str,
    collection: str,
    size: int,
    overlap: int) -> list[DocumentChunk]:
    
    result = []
    n_id = 0
    char_start = 0

    while len(document) > 0:
        result.append(DocumentChunk(
            chunk_id=uuid.uuid4(),
            document_id=document_id,
            collection=collection,
            text=document[:size],
            chunk_index=n_id,
            chunk_metadata=ChunkMetadata(
                source_file=filename,
                page_number=None,
                char_start=char_start,
                char_end=char_start+size-1
            )
            ))
        document = document[size-overlap:]
        n_id += 1
        char_start = char_start + size - overlap

    return result


def chunker_sentence_aware(
    document: str,
    filename: str,
    document_id: str,
    collection: str,
    size: int,
    overlap: int) -> list[DocumentChunk]:

    individual_sentences = [sent for sent in nlp(document).sents]
    result = []
    n_id = 0
    char_start = 0

    while len(individual_sentences) > 0:
        chunk = individual_sentences[:size]
        chunk_length = sum(len(x.text) for x in chunk)

        result.append(DocumentChunk(
            chunk_id=uuid.uuid4(),
            document_id=document_id,
            collection=collection,
            text=" ".join(x.text for x in chunk),
            chunk_index=n_id,
            chunk_metadata=ChunkMetadata(
                source_file=filename,
                page_number=None,
                char_start=char_start,
                char_end=char_start+chunk_length-1
            )
            ))
        n_id += 1
        char_start += chunk_length
        individual_sentences = individual_sentences[size-overlap:]

    return result




if __name__ == "__main__":
    print(chunker_fixed_size(
        document="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        filename="Default Lorem Ipsum",
        document_id="random_id_123",
        collection="test",
        size=10,
        overlap=3
    ))
    for chunk in chunker_sentence_aware(
        document="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        filename="Default Lorem Ipsum",
        document_id="random_id_123",
        collection="test",
        size=3,
        overlap=1
    ):
        print(chunk)
        print("="*20)