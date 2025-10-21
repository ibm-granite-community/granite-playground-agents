# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


# Portions of this file are derived from the Apache 2.0 licensed project "gpt-researcher"
# Original source: https://github.com/assafelovic/gpt-researcher
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes made:

from httpx import Client
from langchain_community.retrievers import ArxivRetriever

from granite_core.search.scraping.scraper import SyncScraper


class ArxivScraper(SyncScraper):
    def scrape(self, link: str, _: Client) -> tuple:
        """
        The function scrapes relevant documents from Arxiv based on a given link and returns the content
        of the first document.

        Returns:
          The code is returning the page content of the first document retrieved by the ArxivRetriever
        for a given query extracted from the link.
        """
        query = link.split("/")[-1]
        retriever = ArxivRetriever(load_max_docs=2, doc_content_chars_max=None)  # type: ignore[call-arg]
        docs = retriever.invoke(query)

        # Include the published date and author to provide additional context,
        # aligning with APA-style formatting in the report.
        context = f"Published: {docs[0].metadata['Published']}; Author: {docs[0].metadata['Authors']}; Content: {docs[0].page_content}"  # noqa: E501
        image: list[str] = []

        return context, image, docs[0].metadata["Title"]
