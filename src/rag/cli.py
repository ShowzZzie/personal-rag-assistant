import pathlib
from typing import Annotated

import typer

import rag.pipeline
import rag.query

app = typer.Typer()


@app.command()
def ingest(
    collection: Annotated[str, typer.Option()],
    file: Annotated[str | None, typer.Option()] = None,
    dir: Annotated[str | None, typer.Option()] = None
) -> None:

    if not file and not dir:
        raise typer.BadParameter("Need with a file or a directory provided")
    if file and dir:
        raise typer.BadParameter("Either a single file or directory is required — not both")

    if file:
        result = rag.pipeline.ingest(
            str(file),
            collection,
        )
        print(
            f"Successfully ingested {file}! Details: doc_id={result.document_id} | "
            f"chunk_count={result.chunk_count} | ingested_at={result.ingested_at}"
            )

    if dir:
        for item in pathlib.Path(dir).glob("*.pdf"):
            try:
                result = rag.pipeline.ingest(
                    str(item),
                    collection
                )
                print(
                    f"Successfully ingested {str(item)}! Details: doc_id={result.document_id} | "
                    f"chunk_count={result.chunk_count} | ingested_at={result.ingested_at}"
                    )
            except Exception as e:
                print(f"Encountered an error with {str(item)}: {e}")


@app.command()
def ask(
    question: str,
    collection: Annotated[str, typer.Option()],
    show_sources: Annotated[bool, typer.Option("--show-sources")] = False
) -> None: 
    
    result = rag.query.query(
        question,
        collection
    )

    print("ANSWER: ", result.answer)
    if show_sources:
        source_lines: list[str] = []
        for i, chunk in enumerate(result.sources):
            title = chunk.chunk.chunk_metadata.source_file
            text = chunk.chunk.text
            source_lines.append(f"[{i+1}] (source: {title}) \"{text}\"")

        sources = "\n".join(source_lines)
        print("SOURCES: ", sources)