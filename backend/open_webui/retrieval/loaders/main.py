import requests
import logging
import ftfy
import os
import hashlib
import pandas as pd
from dataclasses import dataclass
from unstructured.partition.pdf import partition_pdf
import sys

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredXMLLoader,
    YoutubeLoader,
)
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from open_webui.env import SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL

# OCR Agent
os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

known_source_ext = [
    "go",
    "py",
    "java",
    "sh",
    "bat",
    "ps1",
    "cmd",
    "js",
    "ts",
    "css",
    "cpp",
    "hpp",
    "h",
    "c",
    "cs",
    "sql",
    "log",
    "ini",
    "pl",
    "pm",
    "r",
    "dart",
    "dockerfile",
    "env",
    "php",
    "hs",
    "hsc",
    "lua",
    "nginxconf",
    "conf",
    "m",
    "mm",
    "plsql",
    "perl",
    "rb",
    "rs",
    "db2",
    "scala",
    "bash",
    "swift",
    "vue",
    "svelte",
    "msg",
    "ex",
    "exs",
    "erl",
    "tsx",
    "jsx",
    "hs",
    "lhs",
]


# PDF Loader
@dataclass
class Element:
    type: str
    text: str


def table_chunking(file_path):
    raw_pdf_elements = partition_pdf(
        file_path,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy='by_title',
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        strategy='hi_res',
        languages=["en"]
    )

    categorized_elements = []
    for i, element in enumerate(raw_pdf_elements):
        if 'CompositeElement' in element.category:
            categorized_elements.append(Element(type="text", text=element.text))
        elif 'Table' == element.category:
            if raw_pdf_elements[i - 1].metadata.orig_elements[-1].category == 'Title':
                txt = f'{raw_pdf_elements[i - 1].metadata.orig_elements[-1].text}\n{element.metadata.text_as_html}'
            else:
                txt = element.metadata.text_as_html
            categorized_elements.append(Element(type="table", text=txt))

    return categorized_elements


class CustomPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self) -> list[Document]:
        elements = table_chunking(self.file_path)
        if not elements:
            return []

        docs = []
        filename = os.path.basename(self.file_path)
        filename = filename[:filename.rindex('.')].lower()

        # Text elements
        text_elements = [e for e in elements if e.type == 'text']
        for elem in text_elements:
            docs.append(Document(
                page_content=elem.text,
                metadata={
                    "filename": filename,
                    "hash": hashlib.md5(elem.text.encode()).hexdigest(),
                    "type": "text"
                }
            ))

        # Table elements
        table_elements = filter(lambda x: x.type == "table", elements)
        summary_prompt = ChatPromptTemplate.from_template("""
        Provide a comprehensive and accurate description of the following table. 
        - Include all figures and facts without adding any information not present in the table.
        - Describe the purpose of the table and summarize the content.
        - Detail the values in each row and column clearly.

        Table Data:
        {element}
        """)

        llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=0,
            cache=False,  # TODO: Maybe true ?
            model='llama3.1:70b-instruct-q8_0',
            seed=42
        )
        table_texts = [e.text for e in table_elements]
        chain = {"element": lambda x: x} | summary_prompt | llm

        # Process table summaries in batches
        for summary, text in zip(chain.batch(table_texts, {"max_concurrency": 5}), table_texts):
            docs.append(Document(
                page_content=summary.content,
                metadata={
                    "filename": filename,
                    "hash": hashlib.md5(summary.content.encode()).hexdigest(),
                    "original_content": text,
                    "type": "table"
                }
            ))

        return docs


class TikaLoader:
    def __init__(self, url, file_path, mime_type=None):
        self.url = url
        self.file_path = file_path
        self.mime_type = mime_type

    def load(self) -> list[Document]:
        with open(self.file_path, "rb") as f:
            data = f.read()

        if self.mime_type is not None:
            headers = {"Content-Type": self.mime_type}
        else:
            headers = {}

        endpoint = self.url
        if not endpoint.endswith("/"):
            endpoint += "/"
        endpoint += "tika/text"

        r = requests.put(endpoint, data=data, headers=headers)

        if r.ok:
            raw_metadata = r.json()
            text = raw_metadata.get("X-TIKA:content", "<No text content found>")

            if "Content-Type" in raw_metadata:
                headers["Content-Type"] = raw_metadata["Content-Type"]

            log.debug("Tika extracted text: %s", text)

            return [Document(page_content=text, metadata=headers)]
        else:
            raise Exception(f"Error calling Tika: {r.reason}")


class Loader:
    def __init__(self, engine: str = "", **kwargs):
        self.engine = engine
        self.kwargs = kwargs

    def load(
            self, filename: str, file_content_type: str, file_path: str
    ) -> list[Document]:
        loader = self._get_loader(filename, file_content_type, file_path)
        docs = loader.load()

        return [
            Document(
                page_content=ftfy.fix_text(doc.page_content), metadata=doc.metadata
            )
            for doc in docs
        ]

    def _get_loader(self, filename: str, file_content_type: str, file_path: str):
        file_ext = filename.split(".")[-1].lower()

        if self.engine == "tika" and self.kwargs.get("TIKA_SERVER_URL"):
            if file_ext in known_source_ext or (
                    file_content_type and file_content_type.find("text/") >= 0
            ):
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                loader = TikaLoader(
                    url=self.kwargs.get("TIKA_SERVER_URL"),
                    file_path=file_path,
                    mime_type=file_content_type,
                )
        else:
            if file_ext == "pdf":
                loader = CustomPDFLoader(file_path)
            elif file_ext == "csv":
                loader = CSVLoader(file_path)
            elif file_ext == "rst":
                loader = UnstructuredRSTLoader(file_path, mode="elements")
            elif file_ext == "xml":
                loader = UnstructuredXMLLoader(file_path)
            elif file_ext in ["htm", "html"]:
                loader = BSHTMLLoader(file_path, open_encoding="unicode_escape")
            elif file_ext == "md":
                loader = TextLoader(file_path, autodetect_encoding=True)
            elif file_content_type == "application/epub+zip":
                loader = UnstructuredEPubLoader(file_path)
            elif (
                    file_content_type
                    == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    or file_ext == "docx"
            ):
                loader = Docx2txtLoader(file_path)
            elif file_content_type in [
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ] or file_ext in ["xls", "xlsx"]:
                loader = UnstructuredExcelLoader(file_path)
            elif file_content_type in [
                "application/vnd.ms-powerpoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ] or file_ext in ["ppt", "pptx"]:
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_ext == "msg":
                loader = OutlookMessageLoader(file_path)
            elif file_ext in known_source_ext or (
                    file_content_type and file_content_type.find("text/") >= 0
            ):
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                loader = TextLoader(file_path, autodetect_encoding=True)

        return loader
