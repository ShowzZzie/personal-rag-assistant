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



def chunker_recursive(
    document: str,
    filename: str,
    document_id: str,
    collection: str,
    size: int,
    overlap: int,
    recursive_order: list[str] = None,
    n_id: int = 0,
    char_start: int = 0,) -> list[DocumentChunk]:

    if not recursive_order:
        return chunker_fixed_size(document, filename, document_id, collection, size, overlap)

    result = []
    splitter = recursive_order[0]
    blocks = document.split(splitter)

    for block in blocks:
        if len(block) > size:
            rec_result = chunker_recursive(block, filename, document_id, collection, size, overlap, recursive_order[1:], n_id, char_start)
            result.extend(rec_result)
            n_id+=len(rec_result)
        elif block != "":
            result.append(DocumentChunk(
                chunk_id=uuid.uuid4(),
                document_id=document_id,
                collection=collection,
                text=block,
                chunk_index=n_id,
                chunk_metadata=ChunkMetadata(
                    source_file=filename,
                    page_number=None,
                    char_start=char_start,
                    char_end=char_start+len(block)-1
                )
            ))
            n_id+=1
        char_start = char_start+len(block)+len(splitter) # char_start needs fixing

    return result




if __name__ == "__main__":
    """print(chunker_fixed_size(
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
    """

    print("/-="*20)

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
        recursive_order=["\n\n", "\n", "."]
    ):
        print(chunk)
        print("&"*20)