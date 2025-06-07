# ─── utils/tax_rag_advisor.py ─────────────────────────────────────────────────

import os
import pickle
import json
from typing import Dict, Any

from granite.client import GraniteAPI
from config.settings import TAX_VECTOR_INDEX_PATH

class TaxRAGAdvisor:
    """
    1. Attempts to load a pre‐built FAISS (or similar) vector index of global SMB tax documents.
    2. If the index exists, runs a semantic search over “SMB tax code for {country}” 
       and retrieves top‐k contexts, then prompts Granite to structure them as JSON.
    3. If the index file is missing or semantic lookup fails, it simply prompts Granite directly
       to “generate a JSON object describing typical SMB tax brackets, deductions, and subsidies for {country}.”
    """

    def __init__(self, granite_client: GraniteAPI, index_path: str = None):
        self.granite = granite_client
        self.index_path = index_path or TAX_VECTOR_INDEX_PATH
        self.index = None
        self._load_index()

    def _load_index(self):
        """
        Load a pickled FAISS (or other) index from disk. It should map 
        “country‐specific tax documents” to vectors.
        If missing, index stays None (Granite fallback).
        """
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    self.index = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load tax vector index: {e}")
                self.index = None
        else:
            # No index file—will rely on Granite prompt only
            self.index = None

    def _semantic_search(self, country: str, top_k: int = 3) -> str:
    """
    Given a country string, run a semantic search in self.index to retrieve 
    the top_k most relevant passages. Returns a concatenated string of those passages.
    """
        if self.index is None or not self.index_docs:
            return ""
    
        # 1) Build a query prompt for embedding
        query = f"SMB tax code for {country}"
    
        # 2) Embed the query text into a vector
        #    (Assumes you have a wrapper self.embed_text that returns a 1D list or np.array)
        query_vec = self.embed_text(query)  # e.g. [0.12, -0.03, …]
    
        # 3) Perform the FAISS search
        #    FAISS expects a 2D array of shape (n_queries, dim)
        import numpy as np
        qv = np.array([query_vec], dtype="float32")
        distances, indices = self.index.search(qv, top_k)  
        # distances: shape (1, top_k), indices: shape (1, top_k)
    
        # 4) Gather your original text passages by index
        hits = []
        for idx in indices[0]:
            if 0 <= idx < len(self.index_docs):
                hits.append(self.index_docs[idx])
    
        # 5) Return them joined by double-newline for readability
        return "\n\n".join(hits)


    def fetch_tax_brackets(self, country: str) -> Dict[str, Any]:
        """
        Returns a JSON‐like Python dict with at least:
          {
            "brackets": [
              {"min_income": 0,    "max_income": 10000, "rate": 0.0},
              {"min_income": 10001,"max_income": 30000, "rate": 0.10},
              ...
            ],
            "deductions": [
              {"name": "Small Business Equipment Deduction", "max_amount": 5000},
              ...
            ],
            "subsidies": [
              {"name": "SMB Tech Adoption Credit", "description": "..."},
              ...
            ]
          }
        1) If vector index is available, pull context → call Granite for final JSON.
        2) Otherwise, call Granite directly to “invent” a plausible structure.
        3) If Granite fails, return an empty dict.
        """
        # 1) Attempt semantic search to get context
        context_text = self._semantic_search(country)
        if context_text:
            prompt = f"""
Below are excerpts from official SMB tax code documents for {country}. 
Using only the information provided, extract a JSON object with:
1) "brackets": a list of tax bracket objects, each with "min_income", "max_income", and "rate" (decimal).
2) "deductions": a list of objects, each with "name" and any limit or percentage ("max_amount" or "percent").
3) "subsidies": a list of objects, each with "name" and a short "description" field.

Context Excerpts:
\"\"\"{context_text}\"\"\"

Respond ONLY with a valid JSON object.
"""
        else:
            # 2) Granite fallback: ask directly about typical SMB brackets/deductions/subsidies
            prompt = f"""
You are a knowledgeable global tax advisor. Provide, in JSON format, the key SMB 
tax information for {country}, including:
1) "brackets": a list of bracket objects with fields 
   - "min_income" (int), "max_income" (int), "rate" (decimal, e.g., 0.10 for 10%).
2) "deductions": a list of objects, each with "name" (string) and 
   either "max_amount" (int) or "percent" (decimal).
3) "subsidies": a list of objects, each with "name" and "description" (string).

Ensure that the JSON keys match exactly: "brackets", "deductions", "subsidies".
Example:
{
  "brackets": [
    {"min_income": 0,    "max_income": 50000, "rate": 0.10},
    {"min_income": 50001,"max_income": 100000, "rate": 0.20}
  ],
  "deductions": [
    {"name": "Business Expense Deduction", "max_amount": 10000}
  ],
  "subsidies": [
    {"name": "SMB Digitalization Credit", "description": "…"}
  ]
}
"""
        try:
            granite_reply = self.granite.generate_text(prompt, max_tokens=512, temperature=0.0)
            # Attempt to load as JSON
            tax_data = json.loads(granite_reply)
            return tax_data
        except Exception as e:
            print(f"⚠️ Failed to parse Granite JSON for {country}: {e}")
            return {}

