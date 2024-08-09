# ##############################################################################################################
# Author: martian0x80
# Description: Extract Invoice Information from PDF with LLM, Swipe
# ##############################################################################################################

import os
import argparse
import json
import asyncio
from typing import Dict, Any
from enum import Enum
import pdfplumber
from openai import AsyncOpenAI
from groq import AsyncGroq
import pytesseract
from PIL import Image


class Backend(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"


PROMPT = """
Extract invoice information from a PDF/Image file in JSON format. The invoice includes customer details, products, and total amount. The JSON output should contain the following fields:

* customer_details (key: customer_details)
* products (key: products) (list of products)
* total_amount (key: total_amount)

Extract this information from the invoice, whether it appears in a table or as text. Do not include any other information in the JSON object. If the information is not present on a page, leave the corresponding field empty.
"""


class Extractor:
    def __init__(
        self, file_path: str, type: str, api_key: str, model: str, backend: Backend
    ):
        if file_path is not None and os.path.exists(file_path):
            if type == "pdf":
                self.pdf = pdfplumber.open(file_path)
            elif type == "image":
                self.image = file_path
            else:
                raise ValueError("Invalid file type. Supported types are pdf and image")
        self.api_key = api_key
        self.model = model
        self.backend = backend

        if self.backend == Backend.GROQ:
            self.client = AsyncGroq(api_key=self.api_key)
        else:
            self.client = AsyncOpenAI(
                api_key=self.api_key, base_url="https://api.groq.com/openai/v1"
            )

    async def inferer_pdf(self, page: pdfplumber.page.Page) -> str:
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": page.extract_text()},
        ]

        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.5, max_tokens=1024
        )
        return response.choices[0].message.content

    @staticmethod
    def dump(data: Dict[str, Any], output_path: str) -> None:
        print("Valid JSON. Successfully extracted data\nDumping to JSON file...")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    async def process_pdf(self) -> Dict[str, Any]:
        tasks = [self.inferer_pdf(page) for page in self.pdf.pages]
        results = await asyncio.gather(*tasks)

        combined_data = {}
        for result in results:
            try:
                if "```" in result:
                    result = result.split("```")[1]
                page_data = json.loads(result)
                combined_data.update(page_data)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {result}")
        return combined_data

    async def process_image(self) -> Dict[str, Any]:
        image = Image.open(self.image)
        extracted_text = pytesseract.image_to_string(image)
        print("Extracting text from image...")

        messages = [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                # The following only works with vision models
                # "content": [
                #     {"type": "text", "text": PROMPT},
                #     {
                #         "type": "image_url",
                #         "image_url": {"url": f"data:image/png;base64,{image_data}"},
                #     },
                # ],
                "content": extracted_text,
            },
        ]

        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.5, max_tokens=1024
        )

        result = response.choices[0].message.content
        try:
            if "```" in result:
                result = result.split("```")[1]
            return json.loads(result)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {result}")
            return {}


async def main():
    parser = argparse.ArgumentParser(
        description="Extract Invoice Information from PDF with LLM\nAuthor: martian0x80\nSet API_KEY and MODEL environment variables before use",
        prog="extractLLM.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to the PDF/Image file"
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="Type of the input file (pdf/image). Leave empty for auto-detection",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to the output JSON file"
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=Backend,
        choices=list(Backend),
        help="Use Groq or OpenAI as backend",
        default=Backend.OPENAI,
    )
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY")
    model = os.environ.get("MODEL", "llama-3.1-70b-versatile")

    if not api_key:
        raise ValueError("API_KEY environment variable is not set")

    if not args.input:
        raise ValueError("PDF path is not provided")

    data = None
    if args.type == "image" or args.input.endswith((".jpg", ".jpeg", ".png")):
        data = await Extractor(
            args.input, "image", api_key, model, args.backend
        ).process_image()
    elif args.type == "pdf" or args.input.endswith(".pdf"):
        data = await Extractor(
            args.input, "pdf", api_key, model, args.backend
        ).process_pdf()

    if data:
        Extractor.dump(data, args.output)


if __name__ == "__main__":
    asyncio.run(main())
