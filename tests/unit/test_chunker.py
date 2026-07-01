from rag.chunker import chunker_fixed_size, chunker_recursive, chunker_sentence_aware
import pytest


def test_chunker_fixed_size_good():
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    chunks = chunker_fixed_size(text, "test", "123", "tc", 100, 10)
    assert chunks[0].text == "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore "
    assert chunks[1].text == "ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris"
    assert chunks[2].text == "co laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in volupt"
    assert chunks[3].text == " in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat n"
    assert chunks[4].text == "upidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    assert len(chunks) == 5
    assert all(chunk.collection == "tc" for chunk in chunks)
    assert all(len(chunk.text) <= 100 for chunk in chunks)

def test_chunker_fixed_size_empty_string():
    assert chunker_fixed_size("","test","123","tc",100,10) == []

def test_chunker_fixed_size_small_text():
    small_size_chunk = chunker_fixed_size("Foo bar baz", "test", "123", "tc", 100, 10)
    assert len(small_size_chunk) == 1
    assert all(chunk.collection == "tc" for chunk in small_size_chunk)
    assert all(len(chunk.text) == len("Foo bar baz") for chunk in small_size_chunk)



def test_chunker_sentence_aware_good():
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    chunks = chunker_sentence_aware(text, "test", "123", "tc", 3, 1)
    assert len(chunks) == 2
    assert chunks[0].text == "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
    assert chunks[1].text == "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    assert all(chunk.collection == "tc" for chunk in chunks)

def test_chunk_sentence_aware_empty_string():
    assert chunker_sentence_aware("", "test", "123", "tc", 2, 1) == []

def test_chunk_sentence_aware_small_text():
    text = "A single, test sentence"
    chunks = chunker_sentence_aware(text, "test", "123", "tc", 5, 2)
    assert len(chunks) == 1
    assert chunks[0].text == "A single, test sentence"
    assert all(chunk.collection == "tc" for chunk in chunks)



def test_chunker_recursive_good():
    text = """Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n\nUt enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.\nDuis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""
    chunks = chunker_recursive(text, "test", "123", "tc", 400, 1, ["\n\n", "\n", "."])
    assert len(chunks) == 2
    assert "Lorem ipsum" in chunks[0].text
    assert "Ut enim ad minim" in chunks[1].text
    assert all(chunk.collection == "tc" for chunk in chunks)

def test_chunker_recursive_empty_string():
    assert chunker_recursive("", "test", "123", "tc", 400, 1, ["\n\n", "\n", "."]) == []

def test_chunker_recursive_small_text():
    text = "Short test text."
    chunks = chunker_recursive(text, "test", "123", "tc", 500, 100, ["\n\n", "\n", "."])
    assert len(chunks) == 1
    assert chunks[0].text == "Short test text."
    assert all(chunk.collection == "tc" for chunk in chunks)

def test_chunker_recursive_fallback_fixed_size():
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    chunks = chunker_recursive(text, "test", "123", "tc", 200, 15)
    assert len(chunks) == 3
    assert len(chunks[0].text) == 200
    assert len(chunks[1].text) == 200
    assert len(chunks[2].text) == 75
    assert all(chunk.collection == "tc" for chunk in chunks)