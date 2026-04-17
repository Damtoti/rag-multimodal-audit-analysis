"""CLI d\'ingestion — traite tous les PDF d\'un répertoire."""
import argparse
import logging
from pathlib import Path
 
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
 
from audit_rag.config import get_settings
from audit_rag.extractor import PDFExtractor
from audit_rag.vectorstore import AuditVectorStore
 
console = Console()
logging.basicConfig(level=logging.WARNING)
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion de rapports d\'audit PDF")
    parser.add_argument("--dir", type=Path, default=get_settings().data_dir)
    parser.add_argument("--reset", action="store_true", help="Réinitialise l\'index")
    args = parser.parse_args()
 
    pdf_files = list(args.dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]⚠ Aucun PDF trouvé dans {args.dir}[/yellow]")
        return
 
    console.print(f"[bold blue]Fichiers PDF: {len(pdf_files)} rapport(s) à traiter[/bold blue]")
 
    extractor = PDFExtractor()
    store     = AuditVectorStore()
    all_elems = []
 
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Extraction...", total=len(pdf_files))
        for pdf in pdf_files:
            p.update(task, description=f"  {pdf.name}")
            all_elems.extend(extractor.process(pdf))
            p.advance(task)
 
    counts = {t: sum(1 for e in all_elems if e.element_type == t)
              for t in ("text", "table", "image")}
    console.print(f"[green]Extrait : {counts}[/green]")
 
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        p.add_task("Indexation ChromaDB...", total=None)
        store.build(all_elems)
 
    console.print("[bold green]Indexation terminee ![/bold green]")
 
 
if __name__ == "__main__":
    main()