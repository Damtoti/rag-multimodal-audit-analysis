"""Extraction multi-modale de rapports d'audit PDF."""
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
 
import fitz
import numpy as np
import pdfplumber
import torch
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from transformers import CLIPModel, CLIPProcessor
 
from audit_rag.config import Settings, get_settings
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class ExtractedElement:
    element_type: str        # "text" | "table" | "image"
    content: str
    page_number: int
    source_file: str
    metadata: dict[str, Any] = field(default_factory=dict)
    image_data: Optional[bytes] = None
 
 
class PDFExtractor:
    """Extraction texte + tableaux + images d'un PDF financier."""
 
    def __init__(self, settings: Optional[Settings] = None):
        self.cfg    = settings or get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_model: Optional[CLIPModel]     = None
        self._clip_proc:  Optional[CLIPProcessor] = None
        self._llm: Optional[ChatOpenAI]           = None
 
    # ── Lazy loading des modèles lourds ─────────────────────
    @property
    def clip_model(self) -> CLIPModel:
        if self._clip_model is None:
            logger.info("Chargement CLIP %s sur %s", self.cfg.clip_model, self.device)
            self._clip_model = CLIPModel.from_pretrained(
                self.cfg.clip_model
            ).to(self.device)
        return self._clip_model
 
    @property
    def clip_processor(self) -> CLIPProcessor:
        if self._clip_proc is None:
            self._clip_proc = CLIPProcessor.from_pretrained(self.cfg.clip_model)
        return self._clip_proc
 
    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.cfg.llm_model,
                temperature=0,
                max_tokens=500,
            )
        return self._llm
 
    # ── Extraction texte ────────────────────────────────────
    def extract_text(self, pdf_path: Path) -> list[ExtractedElement]:
        elements: list[ExtractedElement] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    elements.append(ExtractedElement(
                        element_type="text",
                        content=text.strip(),
                        page_number=page_num,
                        source_file=pdf_path.name,
                        metadata={"char_count": len(text)},
                    ))
        logger.debug("%d blocs texte extraits de %s", len(elements), pdf_path.name)
        return elements
 
    # ── Extraction tableaux ─────────────────────────────────
    def extract_tables(self, pdf_path: Path) -> list[ExtractedElement]:
        elements: list[ExtractedElement] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                    if not tables:
                        continue
                    
                    for table_idx, table in enumerate(tables):
                        if len(table) < 2 or len(table[0]) < 2:
                            continue
                        
                        # Convertir en dataframe et markdown
                        import pandas as pd
                        df = pd.DataFrame(table[1:], columns=table[0])
                        md_table = df.to_markdown(index=False)
                        description = self._describe_table(md_table)
                        
                        elements.append(ExtractedElement(
                            element_type="table",
                            content=f"{description}\n\n{md_table}",
                            page_number=page_num,
                            source_file=pdf_path.name,
                            metadata={
                                "rows": df.shape[0],
                                "cols": df.shape[1],
                                "table_index": table_idx,
                            },
                        ))
                except Exception as exc:
                    logger.warning("Extraction table p.%d de %s : %s", page_num, pdf_path.name, exc)
        
        logger.debug("%d tableaux extraits de %s", len(elements), pdf_path.name)
        return elements
 
    # ── Extraction images ───────────────────────────────────
    def extract_images(self, pdf_path: Path) -> list[ExtractedElement]:
        elements: list[ExtractedElement] = []
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            for img_index, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes  = base_image["image"]
                    pil_img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    if pil_img.width < 100 or pil_img.height < 100:
                        continue
 
                    clip_emb    = self._get_clip_embedding(pil_img)
                    description = self._describe_image(pil_img)
 
                    elements.append(ExtractedElement(
                        element_type="image",
                        content=description,
                        page_number=page_num,
                        source_file=pdf_path.name,
                        image_data=img_bytes,
                        metadata={
                            "format":         base_image["ext"],
                            "width":          pil_img.width,
                            "height":         pil_img.height,
                            "clip_embedding": clip_emb.tolist(),
                            "image_index":    img_index,
                        },
                    ))
                except Exception as exc:
                    logger.warning(
                        "Image %d p.%d de %s : %s",
                        img_index, page_num, pdf_path.name, exc
                    )
        doc.close()
        return elements
 
    # ── Pipeline principal ──────────────────────────────────
    def process(self, pdf_path: Path) -> list[ExtractedElement]:
        logger.info("Traitement : %s", pdf_path.name)
        elements = (
            self.extract_text(pdf_path)
            + self.extract_tables(pdf_path)
            + self.extract_images(pdf_path)
        )
        counts = {t: sum(1 for e in elements if e.element_type == t)
                  for t in ("text", "table", "image")}
        logger.info("Extrait depuis %s : %s", pdf_path.name, counts)
        return elements
 
    # ── Helpers ─────────────────────────────────────────────
    def _get_clip_embedding(self, pil_img: Image.Image) -> np.ndarray:
        inputs = self.clip_processor(
            images=pil_img, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
        return embedding.cpu().numpy().flatten()
 
    def _describe_image(self, pil_img: Image.Image) -> str:
        try:
            import base64
            buf    = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            b64img = base64.b64encode(buf.getvalue()).decode()
            response = self.llm.invoke([{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64img}", "detail": "high"}},
                    {"type": "text", "text": (
                        "Décris cette image d'un rapport d'audit financier. "
                        "Inclus : type de graphique, métriques clés, tendances, anomalies. "
                        "Max 150 mots, en français."
                    )},
                ],
            }])
            return response.content
        except Exception as exc:
            logger.warning("Description image impossible : %s", exc)
            return "Image extraite d'un rapport d'audit (description non disponible)"
 
    def _describe_table(self, md_table: str) -> str:
        try:
            prompt = ChatPromptTemplate.from_template(
                "Tableau financier :\\n{table}\\n\\n"
                "Description courte (max 80 mots) : type de données, "
                "indicateurs clés, tendances. Commence par \'Ce tableau présente\'."
            )
            return (prompt | self.llm).invoke({"table": md_table[:2000]}).content
        except Exception as exc:
            logger.warning("Description tableau impossible : %s", exc)
            return "Tableau financier extrait (description non disponible)"