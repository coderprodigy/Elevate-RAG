"""Small deterministic synthetic corpus for offline tests and the demo (FR-D7).

SYNTHETIC — every record carries `source="synthetic"` and the manifest sets `synthetic: true`.
The facts are invented (fictional people, towns and companies) so no model can answer them from
memory, and the questions cover single-hop, multi-hop, comparison, temporal, global and
unanswerable cases with known gold documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from elevate_rag.data.datasets import LoadResult
from elevate_rag.data.prepare import write_processed
from elevate_rag.types import Document, Question

_DOCS: list[tuple[str, str, str]] = [
    (
        "syn-001",
        "Velmora",
        "Velmora is a coastal town in the province of Ostrand. Velmora was "
        "founded in 1712 by the cartographer Ilse Brandt. The town is known for its amber harbour.",
    ),
    (
        "syn-002",
        "Ilse Brandt",
        "Ilse Brandt was a cartographer born in Kessering in 1680. Ilse "
        "Brandt mapped the Ostrand coast and later founded the town of Velmora.",
    ),
    (
        "syn-003",
        "Kessering",
        "Kessering is an inland city in Ostrand. Kessering hosts the Royal "
        "Survey Academy, where Ilse Brandt trained as a cartographer.",
    ),
    (
        "syn-004",
        "Tamsin Okoro",
        "Tamsin Okoro is an engineer who founded the company Lumenward in "
        "2009. Lumenward builds tidal turbines and is headquartered in Velmora.",
    ),
    (
        "syn-005",
        "Lumenward",
        "Lumenward is a renewable energy company headquartered in Velmora. "
        "Lumenward was founded in 2009 by Tamsin Okoro and employs 340 people.",
    ),
    (
        "syn-006",
        "Corvane Systems",
        "Corvane Systems is a software company headquartered in "
        "Kessering. Corvane Systems was founded in 1998 by Dario Pell and employs 1200 people.",
    ),
    (
        "syn-007",
        "Dario Pell",
        "Dario Pell is a programmer born in Velmora in 1965. Dario Pell "
        "founded Corvane Systems in 1998 after studying at the Royal Survey Academy.",
    ),
    (
        "syn-008",
        "Ostrand",
        "Ostrand is a province with two major settlements, Velmora on the "
        "coast and Kessering inland. The economy of Ostrand combines software, tidal energy and "
        "amber trade.",
    ),
    (
        "syn-009",
        "Amber Festival",
        "The Amber Festival takes place every August in Velmora. The "
        "first Amber Festival was held in 1850 to celebrate the amber harbour.",
    ),
    (
        "syn-010",
        "Tidal Act",
        "The Ostrand Tidal Act of 2015 granted Lumenward permission to "
        "install turbines in the Velmora harbour. The act was signed by governor Mirela Voss.",
    ),
    (
        "syn-011",
        "Mirela Voss",
        "Mirela Voss served as governor of Ostrand from 2012 to 2020. "
        "Mirela Voss was born in Kessering and signed the Ostrand Tidal Act in 2015.",
    ),
    (
        "syn-012",
        "Royal Survey Academy",
        "The Royal Survey Academy in Kessering was established in "
        "1650. It trains cartographers and surveyors; alumni include Ilse Brandt and Dario Pell.",
    ),
]

_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "q01",
        "q": "In which year was Velmora founded?",
        "a": ["1712"],
        "t": "single_hop",
        "gold": ["syn-001"],
    },
    {
        "id": "q02",
        "q": "Who founded the company Lumenward?",
        "a": ["Tamsin Okoro"],
        "t": "single_hop",
        "gold": ["syn-005"],
    },
    {
        "id": "q03",
        "q": "In which city was the founder of Velmora born?",
        "a": ["Kessering"],
        "t": "multi_hop",
        "gold": ["syn-001", "syn-002"],
    },
    {
        "id": "q04",
        "q": "Where is the company founded by Dario Pell headquartered?",
        "a": ["Kessering"],
        "t": "multi_hop",
        "gold": ["syn-007", "syn-006"],
    },
    {
        "id": "q05",
        "q": "Which company has more employees, Lumenward or Corvane Systems?",
        "a": ["Corvane Systems"],
        "t": "comparison",
        "gold": ["syn-005", "syn-006"],
    },
    {
        "id": "q06",
        "q": "Was Lumenward founded before or after Corvane Systems?",
        "a": ["after"],
        "t": "temporal",
        "gold": ["syn-005", "syn-006"],
    },
    {
        "id": "q07",
        "q": "What are the main themes of the economy of Ostrand?",
        "a": ["software, tidal energy and amber trade"],
        "t": "global",
        "gold": ["syn-008"],
    },
    {
        "id": "q08",
        "q": "Who signed the act that allowed Lumenward to install turbines?",
        "a": ["Mirela Voss"],
        "t": "multi_hop",
        "gold": ["syn-010"],
    },
    {
        "id": "q09",
        "q": "What is the population of the moon colony Arkhaven?",
        "a": [],
        "t": "null",
        "gold": [],
        "answerable": False,
    },
    {
        "id": "q10",
        "q": "Which academy trained both Ilse Brandt and Dario Pell?",
        "a": ["Royal Survey Academy"],
        "t": "comparison",
        "gold": ["syn-012"],
    },
    {
        "id": "q11",
        "q": "In which month does the Amber Festival take place?",
        "a": ["August"],
        "t": "single_hop",
        "gold": ["syn-009"],
    },
    {
        "id": "q12",
        "q": "In which year was the governor who signed the Tidal Act first in office?",
        "a": ["2012"],
        "t": "multi_hop",
        "gold": ["syn-010", "syn-011"],
    },
    {
        "id": "q13",
        "q": "Who designed the submarine fleet of Velmora?",
        "a": [],
        "t": "null",
        "gold": [],
        "answerable": False,
    },
    {
        "id": "q14",
        "q": "Where was Dario Pell born?",
        "a": ["Velmora"],
        "t": "single_hop",
        "gold": ["syn-007"],
    },
    {
        "id": "q15",
        "q": "Which settlements make up the province of Ostrand?",
        "a": ["Velmora and Kessering"],
        "t": "global",
        "gold": ["syn-008"],
    },
]


def synthetic_load_result() -> LoadResult:
    docs = [Document(doc_id=i, title=t, text=x, source="synthetic") for i, t, x in _DOCS]
    qs = [
        Question(
            question_id=f"syn-{q['id']}",
            dataset="synthetic",
            question=q["q"],
            answers=q["a"],
            query_type=q["t"],
            answerable=q.get("answerable", True),
            gold_doc_ids=q["gold"],
            metadata={"synthetic": True},
        )
        for q in _QUESTIONS
    ]
    return LoadResult(docs, qs, [], ["SYNTHETIC fixture — not research data"])


def write_synthetic(data_dir: Path) -> dict[str, Any]:
    return write_processed(
        "synthetic",
        synthetic_load_result(),
        data_dir / "processed" / "synthetic",
        params={"seed": 13},
        sources={"builtin": "elevate_rag.data.synthetic"},
        synthetic=True,
        license="CC0 (project-authored)",
        description="Synthetic fictional-facts corpus for offline testing",
    )
