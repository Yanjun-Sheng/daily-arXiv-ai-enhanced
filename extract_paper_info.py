#!/usr/bin/env python3
"""Standalone script to extract paper info and AI sections for a single arXiv paper.

It outputs:
- authors
- tldr
- motivation
- method
- result
- conclusion
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass

import arxiv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class PaperSections(BaseModel):
    tldr: str = Field(description="generate a too long; didn't read summary")
    motivation: str = Field(description="describe the motivation in this paper")
    method: str = Field(description="method of this paper")
    result: str = Field(description="result of this paper")
    conclusion: str = Field(description="conclusion of this paper")


@dataclass
class ExtractedPaperInfo:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    tldr: str
    motivation: str
    method: str
    result: str
    conclusion: str


SYSTEM_PROMPT = """You are a professional paper analyst.
You should avoid unnecessarily long replies and instead provide concise, detailed, and precise answers using correct terminology.
It is prohibited to output any sensitive content such as politics, ethnicity, religion, violence, pornography, terrorism, gambling, regional discrimination, leaders and their relatives; once it is detected that the question or original text contains the above elements, the unified reply will be: "This content has not passed the compliance test and has been hidden."
Your output should in {language}."""

HUMAN_PROMPT = """Please analyze the following abstract of papers.

Content:
{content}"""


def parse_arxiv_id(value: str) -> str:
    """Accept raw ID, abs URL, or pdf URL, and return clean arXiv ID."""
    value = value.strip()
    patterns = [
        r"arxiv\.org/abs/([^/?#]+)",
        r"arxiv\.org/pdf/([^/?#]+)(?:\.pdf)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return value


def fetch_paper(arxiv_id_or_url: str) -> arxiv.Result:
    paper_id = parse_arxiv_id(arxiv_id_or_url)
    search = arxiv.Search(id_list=[paper_id], max_results=1)
    client = arxiv.Client()
    result = list(client.results(search))
    if not result:
        raise ValueError(f"Cannot find paper for ID/URL: {arxiv_id_or_url}")
    return result[0]


def summarize_abstract(abstract: str, model: str, language: str) -> PaperSections:
    llm = ChatOpenAI(model=model).with_structured_output(
        PaperSections, method="function_calling"
    )
    prompt = [
        ("system", SYSTEM_PROMPT.format(language=language)),
        ("human", HUMAN_PROMPT.format(content=abstract)),
    ]
    return llm.invoke(prompt)


def extract_info(arxiv_id_or_url: str, model: str, language: str) -> ExtractedPaperInfo:
    paper = fetch_paper(arxiv_id_or_url)
    sections = summarize_abstract(
        abstract=paper.summary,
        model=model,
        language=language,
    )
    return ExtractedPaperInfo(
        arxiv_id=paper.get_short_id(),
        title=paper.title.strip(),
        authors=[author.name for author in paper.authors],
        abstract=paper.summary.strip(),
        tldr=sections.tldr,
        motivation=sections.motivation,
        method=sections.method,
        result=sections.result,
        conclusion=sections.conclusion,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Authors, TL;DR, Motivation, Method, Result, Conclusion for a random arXiv paper."
    )
    parser.add_argument(
        "paper",
        help="arXiv ID or URL (e.g., 2401.00001 or https://arxiv.org/abs/2401.00001)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="LLM model name for structured extraction.",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Output language for extracted sections.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = extract_info(args.paper, model=args.model, language=args.language)
    print(json.dumps(asdict(info), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
