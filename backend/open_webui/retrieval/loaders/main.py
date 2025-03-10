import hashlib
import itertools
import logging
import os
import sys
from dataclasses import dataclass

import ftfy
import requests
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredXMLLoader,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from unstructured.documents.elements import Table, Title, NarrativeText, Header
from unstructured.partition.pdf import partition_pdf

from open_webui import config
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


def combine_elements(elements):
    current_elements = []
    current_table_elements = []
    combined_elements = []
    skip_next = False
    elems = list(itertools.chain(*[d.metadata.orig_elements for d in elements]))
    for i, element in enumerate(elems):
        if skip_next:
            skip_next = False
            continue
        if isinstance(element, Header):
            continue

        if isinstance(element, Title) and i + 1 < len(elems) and isinstance(elems[i + 1], Table):
            current_table_elements.append(element)
            current_table_elements.append(elems[i + 1])
            skip_next = True
        elif isinstance(element, NarrativeText) and i - 1 > 0 and isinstance(elems[i - 1], Table):
            current_table_elements.append(element)
        else:
            if current_table_elements:
                combined_elements.append(merge_elements(current_table_elements))
                current_table_elements = []

            if isinstance(element, Title) and current_elements:
                combined_elements.append(merge_elements(current_elements))
                current_elements = [element]
            else:
                current_elements.append(element)

    if current_elements:
        combined_elements.append(merge_elements(current_elements))
    elif current_table_elements:
        combined_elements.append(merge_elements(current_table_elements))

    return combined_elements


def merge_elements(elements):
    combined_text = None
    type = 'text'
    for element in elements:
        if isinstance(element, Table):
            type = 'table'
            if combined_text:
                combined_text = f'{combined_text}\n{element.metadata.text_as_html}'
            else:
                combined_text = element.metadata.text_as_html
        else:
            if combined_text:
                combined_text = f'{combined_text}\n{element.text}'
            else:
                combined_text = element.text

    return Element(type=type, text=combined_text)


def table_chunking(file_path):
    raw_pdf_elements = partition_pdf(
        file_path,
        infer_table_structure=True,
        chunking_strategy='by_title',
        strategy='hi_res',
        extract_tables=True,
        max_characters=3000,
        new_after_n_chars=2800,
        combine_text_under_n_chars=200,
        languages=['en']
    )

    return combine_elements(raw_pdf_elements)


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

        summary_prompt = ChatPromptTemplate.from_template("""
                 You are an expert report writer specializing in summarizing tabular data for seminar presentations. Your task is to analyze the provided HTML code representing a table and extract all relevant information to create a detailed report suitable for a seminar audience.

                **Input:** You will receive an HTML code snippet containing a table.

                **Output:** Generate a report summarizing the table's content. The report must be formatted as a series of bullet points, with each bullet point representing a distinct piece of information from the table. Ensure the report is comprehensive and covers all details present in the table. The report should be understandable and informative to a seminar audience.

                **Instructions:**

                1.  **Understand the HTML:** Carefully examine the provided HTML code to understand the structure and content of the table. Identify table headers ( `<th>` ) and data cells ( `<td>` ).
                2.  **Extract Data:** Extract all the data present in the table, including headers and cell values.
                3.  **Synthesize Information:** Combine the extracted data into meaningful statements. For example, if a table has columns "Name" and "Age", a row with "John" and "30" should be represented as "Name: John, Age: 30". If a cell is empty, represent it as "N/A" or "Not Available" unless the context suggests a better alternative.
                4.  **Format as Bullet Points:** Present the synthesized information as a series of bullet points. Each bullet point should represent a distinct piece of information from the table.
                5.  **Comprehensive Coverage:** Ensure that the report covers all the details present in the table. Do not omit any relevant information. Include information from the table header ( `<th>` ) for context.
                6.  **Clarity and Readability:** The report should be clear, concise, and easy to understand for a seminar audience. Use descriptive language to explain the data. Avoid technical jargon unless it is necessary and well-defined.
                7.  **No Table Structure:** Do not attempt to recreate the table structure in the report. The report should be a textual summary of the table's content.
                8.  **Handle Complex Tables:** If the table contains nested tables or complex structures, focus on extracting the primary data and presenting it in a clear and understandable manner.
                9. **Follow the example format strictly.**

                **Example:**

                **HTML Input:**  
                ```html      
                <table>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Age</th>
                      <th>City</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Alice</td>
                      <td>25</td>
                      <td>New York</td>
                    </tr>
                    <tr>
                      <td>Bob</td>
                      <td>30</td>
                      <td>London</td>
                    </tr>
                    <tr>
                      <td>Charlie</td>
                      <td></td>
                      <td>Paris</td>
                    </tr>
                  </tbody>
                </table>

                **Report Output:**
                Name: Alice, Age: 25, City: New York
                Name: Bob, Age: 30, City: London
                Name: Charlie, Age: Not Available, City: Paris
                ### Now, analyze the following HTML code and generate the report:

                ```html
                  {element}
                """)

        llm = ChatOpenAI(
            base_url=config.OPENAI_API_BASE_URLS.value[0],
            api_key=config.OPENAI_API_KEYS.value[0],
            temperature=0,
            cache=False,  # TODO: Maybe true ?
            model=os.getenv("MODEL_NAME"),
            seed=42
        )
        current_doc = None
        for elem in elements:
            if elem.type == "table":
                chain = {"element": lambda x: x} | summary_prompt | llm
                elem.text = chain.invoke(elem.text).content
            if current_doc:
                combined_texts = f'{current_doc.page_content}\n{elem.text}'
            else:
                combined_texts = elem.text

            if len(combined_texts) <= config.CHUNK_SIZE.value:
                current_doc = Document(
                    page_content=combined_texts
                )
            else:
                docs.append(Document(
                    page_content=current_doc.page_content,
                    metadata={
                        "filename": filename,
                        "hash": hashlib.md5(current_doc.page_content.encode()).hexdigest(),
                        "type": "text"
                    }
                ))
                current_doc = None

        if current_doc:
            docs.append(Document(
                page_content=current_doc.page_content,
                metadata={
                    "filename": filename,
                    "hash": hashlib.md5(current_doc.page_content.encode()).hexdigest(),
                    "type": "text"
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
