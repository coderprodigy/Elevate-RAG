"""Generate the bundled synthetic Tiny_Sample fixture.

This script is the single source of truth for the fixture. It writes:

    corpus.jsonl                 Normalised_Corpus records {doc_id, title, text, source}
    questions.jsonl              Question_Set records (20 questions, every Query_Type)
    fake_llm/extraction.jsonl    scripted extraction responses, one per document
    fake_llm/generation.jsonl    scripted generator responses per question and variant
    fake_llm/judge.jsonl         scripted LLM_Judge verdicts for every scripted prediction
    fake_llm/malformed.jsonl     deliberately malformed responses for error-path tests
    manifest.json                counts, formats and constants

Run from the repository root:  python fixtures/tiny_sample/build.py
The output is deterministic: rerunning produces byte-identical files.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE_VERSION = "1.0.0"
SOURCE = "tiny_sample"
REFUSAL = "insufficient information"
TUPLE_DELIM = "<|#|>"  # LightRAG 1.5 DEFAULT_TUPLE_DELIMITER
COMPLETE = "<|COMPLETE|>"  # LightRAG 1.5 DEFAULT_COMPLETION_DELIMITER
CITE_RE = re.compile(r"\[\[(tiny-doc-\d{2})\]\]")

# --------------------------------------------------------------------------- corpus
# Fictional world: tidal energy around the Varn Strait. All names are invented so
# that a closed-book model cannot answer from memory.
DOCS: list[tuple[str, str, str]] = [
    (
        "tiny-doc-01",
        "Halvren Institute of Tidal Engineering",
        "The Halvren Institute of Tidal Engineering is a research institute based in Port "
        "Ossery. It was founded in 1987 by the engineer Mirela Danthorpe to study how tidal "
        "currents could be turned into electricity. The institute runs laboratories on the "
        "Port Ossery waterfront and field stations along the Varn Strait. Its best-known "
        "project is the Kestrel Tidal Array. The institute is funded by public grants and by "
        "licensing fees from its turbine designs.",
    ),
    (
        "tiny-doc-02",
        "Mirela Danthorpe",
        "Mirela Danthorpe is an engineer known for her work on underwater turbines. She "
        "studied mechanical engineering at Quellmoor University, where she wrote a thesis on "
        "the fatigue of submerged blades. After graduating she spent several years as a "
        "consultant on harbour construction projects. In 1987 she moved to Port Ossery to set "
        "up her own research organisation, and in 2009 she stepped back from administration "
        "to return to design work.",
    ),
    (
        "tiny-doc-03",
        "Quellmoor University",
        "Quellmoor University is a public university in the city of Brindlehaven. It was "
        "established in 1891 and is organised into eleven faculties, including a faculty of "
        "engineering with a marine technology department. The main campus sits on the "
        "northern bank of the Ammer River. Quellmoor University is the second largest "
        "employer in Brindlehaven after the port.",
    ),
    (
        "tiny-doc-04",
        "Kestrel Tidal Array",
        "The Kestrel Tidal Array is a tidal power station in the Varn Strait. It was designed "
        "and built by the Halvren Institute of Tidal Engineering and was commissioned in 2014. "
        "The array consists of twenty-four seabed turbines with a combined capacity of 42 "
        "megawatts, enough to supply roughly 30,000 homes. Power from the array is carried by "
        "an undersea cable to a substation near Port Ossery.",
    ),
    (
        "tiny-doc-05",
        "Leadership of the Halvren Institute",
        "The Halvren Institute of Tidal Engineering has had three directors. Its founder led "
        "it from 1987 until 2009. Tobin Arkwright, a former naval engineer, was director from "
        "2009 to 2018 and oversaw the construction of the institute's largest project. In 2018 "
        "Arkwright was succeeded by Ysolde Fenwick, who had previously led the Saltmarsh "
        "Observatory for eight years. Fenwick remains director today.",
    ),
    (
        "tiny-doc-06",
        "Orrex Maritime",
        "Orrex Maritime is a marine engineering company headquartered in Brindlehaven. It was "
        "founded in 1974 by the shipbuilder Caspar Orrex. The company began by building cargo "
        "barges and moved into offshore energy in the 1990s. Orrex Maritime built and "
        "operates the Lumen Bay Tidal Array and supplies foundations for offshore wind farms.",
    ),
    (
        "tiny-doc-07",
        "Lumen Bay Tidal Array",
        "The Lumen Bay Tidal Array is a tidal power station in Lumen Bay, south of "
        "Brindlehaven. It was built by Orrex Maritime and commissioned in 2011. The array has "
        "sixteen turbines with a combined capacity of 30 megawatts. It was the first "
        "commercial tidal array in the region, and its turbine foundations were later reused "
        "as a model for offshore wind projects.",
    ),
    (
        "tiny-doc-08",
        "Port Ossery",
        "Port Ossery is a harbour city on the eastern shore of the Varn Strait with a "
        "population of about 210,000. The economy of the city is based on fishing, ship "
        "repair and marine research. A ferry link connects Port Ossery with Corrin Isle, and "
        "the old harbour district is a popular destination for visitors in summer.",
    ),
    (
        "tiny-doc-09",
        "Saltmarsh Observatory",
        "The Saltmarsh Observatory is a coastal research station on Corrin Isle, at the "
        "northern mouth of the Varn Strait. It was established in 1962 to monitor tides and "
        "seabirds. Today the observatory studies how human activity affects marine wildlife "
        "in the strait. Ysolde Fenwick directed the observatory from 2010 to 2018.",
    ),
    (
        "tiny-doc-10",
        "Wren Calloway",
        "Wren Calloway is a marine biologist at the Saltmarsh Observatory who studies grey "
        "seals. Between 2014 and 2017 she tracked seal colonies near the Kestrel Tidal Array "
        "to measure whether the turbines changed their feeding routes. She found that seals "
        "kept their distance from the turbines but that their feeding success was unchanged. "
        "Her results were published in 2017 in the book Currents and Colonies.",
    ),
    (
        "tiny-doc-11",
        "Ossery Accord",
        "The Ossery Accord was an agreement signed in 1999 between the Halvren Institute of "
        "Tidal Engineering and Orrex Maritime. Under the accord the two organisations shared "
        "tidal current measurements and turbine performance data. The accord ended in 2016 "
        "after a dispute over access to performance data from the Kestrel Tidal Array, and "
        "since then the two organisations have operated as competitors.",
    ),
    (
        "tiny-doc-12",
        "Brindlehaven",
        "Brindlehaven is the largest city in the region, with a population of about 480,000. "
        "It lies at the mouth of the Ammer River, where a deep-water port handles most of the "
        "regional cargo traffic. The city is governed by an elected council of thirty-one "
        "members and is known for its covered markets.",
    ),
    # ---- distractors (no question depends on them; several are near misses)
    (
        "tiny-doc-13",
        "Kestrel Ridge",
        "Kestrel Ridge is a granite ridge in the uplands west of Brindlehaven, named after the "
        "kestrels that nest on its cliffs. Its highest point reaches 1,140 metres. A "
        "restoration of the hiking trails on the ridge was completed in 2015 at a cost of 3 "
        "million crowns.",
    ),
    (
        "tiny-doc-14",
        "Ossery Harbour Festival",
        "The Ossery Harbour Festival is an annual summer festival in Port Ossery, first held "
        "in 1952. It features boat races, a seafood market and a lantern procession along the "
        "old harbour. The festival has been cancelled only once, in 1988, because of a severe "
        "storm.",
    ),
    (
        "tiny-doc-15",
        "Quellmoor Conservatory",
        "The Quellmoor Conservatory is the music school of Quellmoor University. It opened in "
        "1923 and trains performers and composers. Its best-known graduate is the cellist "
        "Anja Merrow, who founded the Ammer String Quartet in 1961.",
    ),
    (
        "tiny-doc-16",
        "Corrin Isle Lighthouse",
        "The Corrin Isle Lighthouse stands on the southern cliffs of Corrin Isle. It was "
        "built in 1874 and automated in 1979, when the last keeper left the island. The "
        "former keeper's cottage now houses a small museum about shipwrecks in the strait.",
    ),
    (
        "tiny-doc-17",
        "Orrex Ferries",
        "Orrex Ferries was a passenger ferry service between Brindlehaven and Corrin Isle. It "
        "was spun off from Orrex Maritime in 2003 and sold to Brindle Lines in 2012. The "
        "service used three vessels and carried about 400,000 passengers a year at its peak.",
    ),
]

# --------------------------------------------------------------------------- extraction
# Per document: entities (name, type, description) and relations
# (source, target, keywords, description, weight). Names are title case and consistent
# across documents so cross-document links form multi-hop paths. tiny-doc-08 deliberately
# writes "Varn strait" (lower-case s) to exercise identical-normalised-name merging.
E = tuple[str, str, str]
R = tuple[str, str, str, str, float]
HALVREN = "Halvren Institute Of Tidal Engineering"

EXTRACTION: dict[str, tuple[list[E], list[R]]] = {
    "tiny-doc-01": (
        [
            (
                HALVREN,
                "Organization",
                "Research institute in Port Ossery founded in 1987 to "
                "study tidal power; best known for the Kestrel Tidal Array.",
            ),
            ("Port Ossery", "Location", "City where the Halvren Institute is based."),
            (
                "Mirela Danthorpe",
                "Person",
                "Engineer who founded the Halvren Institute of Tidal Engineering in 1987.",
            ),
            (
                "Varn Strait",
                "Location",
                "Strait along which the Halvren Institute runs field stations.",
            ),
            ("Kestrel Tidal Array", "Artifact", "Best-known project of the Halvren Institute."),
        ],
        [
            (
                "Mirela Danthorpe",
                HALVREN,
                "founded, founder",
                "Mirela Danthorpe founded the Halvren Institute in 1987.",
                1.0,
            ),
            (
                HALVREN,
                "Port Ossery",
                "located in, headquarters",
                "The Halvren Institute is based in Port Ossery.",
                0.9,
            ),
            (
                HALVREN,
                "Varn Strait",
                "field stations",
                "The Halvren Institute runs field stations along the Varn Strait.",
                0.6,
            ),
            (
                HALVREN,
                "Kestrel Tidal Array",
                "project",
                "The Kestrel Tidal Array is the best-known project of the Halvren Institute.",
                0.8,
            ),
        ],
    ),
    "tiny-doc-02": (
        [
            (
                "Mirela Danthorpe",
                "Person",
                "Engineer who studied mechanical engineering at "
                "Quellmoor University and set up a research organisation in Port Ossery in "
                "1987.",
            ),
            (
                "Quellmoor University",
                "Organization",
                "University where Mirela Danthorpe studied mechanical engineering.",
            ),
            ("Port Ossery", "Location", "City Mirela Danthorpe moved to in 1987."),
        ],
        [
            (
                "Mirela Danthorpe",
                "Quellmoor University",
                "education, alumna",
                "Mirela Danthorpe studied mechanical engineering at Quellmoor University.",
                1.0,
            ),
            (
                "Mirela Danthorpe",
                "Port Ossery",
                "relocation, research organisation",
                "Mirela Danthorpe moved to Port Ossery in 1987 to set up a research organisation.",
                0.7,
            ),
        ],
    ),
    "tiny-doc-03": (
        [
            (
                "Quellmoor University",
                "Organization",
                "Public university in Brindlehaven "
                "established in 1891 with a marine technology department.",
            ),
            ("Brindlehaven", "Location", "City where Quellmoor University is located."),
            ("Ammer River", "Location", "River on whose northern bank the Quellmoor campus sits."),
        ],
        [
            (
                "Quellmoor University",
                "Brindlehaven",
                "located in",
                "Quellmoor University is a public university in Brindlehaven.",
                1.0,
            ),
            (
                "Quellmoor University",
                "Ammer River",
                "campus location",
                "The main campus of Quellmoor University is on the northern bank of the Ammer "
                "River.",
                0.5,
            ),
        ],
    ),
    "tiny-doc-04": (
        [
            (
                "Kestrel Tidal Array",
                "Artifact",
                "Tidal power station in the Varn Strait, "
                "commissioned in 2014, 24 turbines, capacity 42 megawatts.",
            ),
            ("Varn Strait", "Location", "Strait where the Kestrel Tidal Array is located."),
            (HALVREN, "Organization", "Designed and built the Kestrel Tidal Array."),
            (
                "Port Ossery",
                "Location",
                "City near the substation that receives power from the Kestrel Tidal Array.",
            ),
        ],
        [
            (
                "Kestrel Tidal Array",
                "Varn Strait",
                "located in",
                "The Kestrel Tidal Array is in the Varn Strait.",
                1.0,
            ),
            (
                HALVREN,
                "Kestrel Tidal Array",
                "designed, built, commissioned 2014",
                "The Halvren Institute designed and built the Kestrel Tidal Array, commissioned "
                "in 2014.",
                1.0,
            ),
            (
                "Kestrel Tidal Array",
                "Port Ossery",
                "power supply, undersea cable",
                "An undersea cable carries power from the array to a substation near Port Ossery.",
                0.6,
            ),
        ],
    ),
    "tiny-doc-05": (
        [
            (HALVREN, "Organization", "Institute that has had three directors since 1987."),
            (
                "Tobin Arkwright",
                "Person",
                "Former naval engineer, director of the Halvren Institute from 2009 to 2018.",
            ),
            (
                "Ysolde Fenwick",
                "Person",
                "Director of the Halvren Institute since 2018; "
                "previously led the Saltmarsh Observatory for eight years.",
            ),
            (
                "Saltmarsh Observatory",
                "Organization",
                "Research station previously led by Ysolde Fenwick.",
            ),
        ],
        [
            (
                "Tobin Arkwright",
                HALVREN,
                "director, 2009-2018",
                "Tobin Arkwright was director of the Halvren Institute from 2009 to 2018.",
                1.0,
            ),
            (
                "Ysolde Fenwick",
                HALVREN,
                "director, since 2018",
                "Ysolde Fenwick succeeded Tobin Arkwright as director in 2018.",
                1.0,
            ),
            (
                "Ysolde Fenwick",
                "Tobin Arkwright",
                "succession",
                "Ysolde Fenwick succeeded Tobin Arkwright in 2018.",
                0.8,
            ),
            (
                "Ysolde Fenwick",
                "Saltmarsh Observatory",
                "previous leadership",
                "Ysolde Fenwick led the Saltmarsh Observatory for eight years before 2018.",
                0.9,
            ),
        ],
    ),
    "tiny-doc-06": (
        [
            (
                "Orrex Maritime",
                "Organization",
                "Marine engineering company in Brindlehaven "
                "founded in 1974; builds and operates the Lumen Bay Tidal Array.",
            ),
            ("Caspar Orrex", "Person", "Shipbuilder who founded Orrex Maritime in 1974."),
            ("Brindlehaven", "Location", "City where Orrex Maritime is headquartered."),
            (
                "Lumen Bay Tidal Array",
                "Artifact",
                "Tidal array built and operated by Orrex Maritime.",
            ),
        ],
        [
            (
                "Caspar Orrex",
                "Orrex Maritime",
                "founded, founder",
                "Caspar Orrex founded Orrex Maritime in 1974.",
                1.0,
            ),
            (
                "Orrex Maritime",
                "Brindlehaven",
                "headquarters",
                "Orrex Maritime is headquartered in Brindlehaven.",
                0.9,
            ),
            (
                "Orrex Maritime",
                "Lumen Bay Tidal Array",
                "built, operates",
                "Orrex Maritime built and operates the Lumen Bay Tidal Array.",
                1.0,
            ),
        ],
    ),
    "tiny-doc-07": (
        [
            (
                "Lumen Bay Tidal Array",
                "Artifact",
                "Tidal power station in Lumen Bay, "
                "commissioned in 2011, 16 turbines, capacity 30 megawatts; first commercial "
                "tidal array in the region.",
            ),
            ("Lumen Bay", "Location", "Bay south of Brindlehaven."),
            ("Orrex Maritime", "Organization", "Built the Lumen Bay Tidal Array."),
            ("Brindlehaven", "Location", "City north of Lumen Bay."),
        ],
        [
            (
                "Lumen Bay Tidal Array",
                "Lumen Bay",
                "located in",
                "The Lumen Bay Tidal Array is in Lumen Bay.",
                1.0,
            ),
            (
                "Orrex Maritime",
                "Lumen Bay Tidal Array",
                "built, commissioned 2011",
                "Orrex Maritime built the array, commissioned in 2011.",
                1.0,
            ),
            (
                "Lumen Bay",
                "Brindlehaven",
                "geography",
                "Lumen Bay lies south of Brindlehaven.",
                0.4,
            ),
        ],
    ),
    "tiny-doc-08": (
        [
            (
                "Port Ossery",
                "Location",
                "Harbour city on the eastern shore of the Varn Strait "
                "with a population of about 210,000.",
            ),
            ("Varn strait", "Location", "Strait on whose eastern shore Port Ossery lies."),
            ("Corrin Isle", "Location", "Island connected to Port Ossery by ferry."),
        ],
        [
            (
                "Port Ossery",
                "Varn strait",
                "located on",
                "Port Ossery lies on the eastern shore of the Varn Strait.",
                0.9,
            ),
            (
                "Port Ossery",
                "Corrin Isle",
                "ferry link",
                "A ferry link connects Port Ossery with Corrin Isle.",
                0.5,
            ),
        ],
    ),
    "tiny-doc-09": (
        [
            (
                "Saltmarsh Observatory",
                "Organization",
                "Coastal research station on Corrin "
                "Isle established in 1962; studies effects of human activity on marine "
                "wildlife.",
            ),
            (
                "Corrin Isle",
                "Location",
                "Island at the northern mouth of the Varn Strait, "
                "home of the Saltmarsh Observatory.",
            ),
            ("Varn Strait", "Location", "Strait whose wildlife the Saltmarsh Observatory studies."),
            ("Ysolde Fenwick", "Person", "Directed the Saltmarsh Observatory from 2010 to 2018."),
        ],
        [
            (
                "Saltmarsh Observatory",
                "Corrin Isle",
                "located on",
                "The Saltmarsh Observatory is on Corrin Isle.",
                1.0,
            ),
            (
                "Corrin Isle",
                "Varn Strait",
                "geography",
                "Corrin Isle is at the northern mouth of the Varn Strait.",
                0.6,
            ),
            (
                "Ysolde Fenwick",
                "Saltmarsh Observatory",
                "director, 2010-2018",
                "Ysolde Fenwick directed the Saltmarsh Observatory from 2010 to 2018.",
                1.0,
            ),
        ],
    ),
    "tiny-doc-10": (
        [
            (
                "Wren Calloway",
                "Person",
                "Marine biologist at the Saltmarsh Observatory who studies grey seals.",
            ),
            ("Saltmarsh Observatory", "Organization", "Employer of Wren Calloway."),
            (
                "Kestrel Tidal Array",
                "Artifact",
                "Tidal array near which Wren Calloway tracked seal colonies from 2014 to 2017.",
            ),
            (
                "Currents And Colonies",
                "Content",
                "Book by Wren Calloway published in 2017 on seal behaviour near tidal turbines.",
            ),
        ],
        [
            (
                "Wren Calloway",
                "Saltmarsh Observatory",
                "employment",
                "Wren Calloway works at the Saltmarsh Observatory.",
                0.9,
            ),
            (
                "Wren Calloway",
                "Kestrel Tidal Array",
                "environmental study, grey seals",
                "Wren Calloway studied seal colonies near the Kestrel Tidal Array.",
                1.0,
            ),
            (
                "Wren Calloway",
                "Currents And Colonies",
                "author, published 2017",
                "Wren Calloway published her results in Currents and Colonies.",
                0.8,
            ),
        ],
    ),
    "tiny-doc-11": (
        [
            ("Ossery Accord", "Event", "Data-sharing agreement signed in 1999 and ended in 2016."),
            (HALVREN, "Organization", "Party to the Ossery Accord."),
            ("Orrex Maritime", "Organization", "Party to the Ossery Accord."),
            (
                "Kestrel Tidal Array",
                "Artifact",
                "Its performance data was the subject of the dispute that ended the Ossery Accord.",
            ),
        ],
        [
            (
                HALVREN,
                "Orrex Maritime",
                "data sharing, later competitors",
                "The two organisations shared data under the Ossery Accord from 1999 to 2016 "
                "and have competed since.",
                1.0,
            ),
            (
                "Ossery Accord",
                HALVREN,
                "signatory",
                "The Halvren Institute signed the Ossery Accord in 1999.",
                0.8,
            ),
            (
                "Ossery Accord",
                "Orrex Maritime",
                "signatory",
                "Orrex Maritime signed the Ossery Accord in 1999.",
                0.8,
            ),
            (
                "Ossery Accord",
                "Kestrel Tidal Array",
                "dispute, termination",
                "A dispute over Kestrel performance data ended the accord in 2016.",
                0.7,
            ),
        ],
    ),
    "tiny-doc-12": (
        [
            (
                "Brindlehaven",
                "Location",
                "Largest city in the region with a population of "
                "about 480,000, at the mouth of the Ammer River.",
            ),
            ("Ammer River", "Location", "River at whose mouth Brindlehaven lies."),
        ],
        [
            (
                "Brindlehaven",
                "Ammer River",
                "located at",
                "Brindlehaven lies at the mouth of the Ammer River.",
                0.8,
            ),
        ],
    ),
    "tiny-doc-13": (
        [
            (
                "Kestrel Ridge",
                "Location",
                "Granite ridge west of Brindlehaven, 1,140 metres "
                "high; trail restoration cost 3 million crowns in 2015.",
            ),
            ("Brindlehaven", "Location", "City east of Kestrel Ridge."),
        ],
        [
            (
                "Kestrel Ridge",
                "Brindlehaven",
                "geography",
                "Kestrel Ridge lies in the uplands west of Brindlehaven.",
                0.4,
            ),
        ],
    ),
    "tiny-doc-14": (
        [
            (
                "Ossery Harbour Festival",
                "Event",
                "Annual summer festival in Port Ossery first held in 1952.",
            ),
            ("Port Ossery", "Location", "City hosting the Ossery Harbour Festival."),
        ],
        [
            (
                "Ossery Harbour Festival",
                "Port Ossery",
                "held in",
                "The Ossery Harbour Festival takes place in Port Ossery.",
                0.9,
            ),
        ],
    ),
    "tiny-doc-15": (
        [
            (
                "Quellmoor Conservatory",
                "Organization",
                "Music school of Quellmoor University opened in 1923.",
            ),
            ("Quellmoor University", "Organization", "Parent university of the conservatory."),
            (
                "Anja Merrow",
                "Person",
                "Cellist, graduate of the conservatory, founded the Ammer String Quartet in 1961.",
            ),
        ],
        [
            (
                "Quellmoor Conservatory",
                "Quellmoor University",
                "part of",
                "The conservatory is the music school of Quellmoor University.",
                0.9,
            ),
            (
                "Anja Merrow",
                "Quellmoor Conservatory",
                "alumna",
                "Anja Merrow graduated from the Quellmoor Conservatory.",
                0.7,
            ),
        ],
    ),
    "tiny-doc-16": (
        [
            (
                "Corrin Isle Lighthouse",
                "Artifact",
                "Lighthouse built in 1874 and automated in 1979.",
            ),
            ("Corrin Isle", "Location", "Island on whose southern cliffs the lighthouse stands."),
        ],
        [
            (
                "Corrin Isle Lighthouse",
                "Corrin Isle",
                "located on",
                "The lighthouse stands on the southern cliffs of Corrin Isle.",
                0.9,
            ),
        ],
    ),
    "tiny-doc-17": (
        [
            (
                "Orrex Ferries",
                "Organization",
                "Ferry service between Brindlehaven and Corrin "
                "Isle, spun off in 2003 and sold in 2012.",
            ),
            ("Orrex Maritime", "Organization", "Parent company of Orrex Ferries until 2003."),
            ("Brindle Lines", "Organization", "Bought Orrex Ferries in 2012."),
        ],
        [
            (
                "Orrex Ferries",
                "Orrex Maritime",
                "spin-off",
                "Orrex Ferries was spun off from Orrex Maritime in 2003.",
                0.7,
            ),
            (
                "Brindle Lines",
                "Orrex Ferries",
                "acquisition",
                "Brindle Lines bought Orrex Ferries in 2012.",
                0.6,
            ),
        ],
    ),
}

# --------------------------------------------------------------------------- questions
# (question_id, query_type, question, answers, gold_doc_ids, rag_response_template)
# Templates mark citations as [[doc_id]]; see README for how fakes render them.
Q = tuple[str, str, str, list[str], list[str], str]
QUESTIONS: list[Q] = [
    # single-hop
    (
        "tiny-q01",
        "single-hop",
        "In what year was the Halvren Institute of Tidal Engineering founded?",
        ["1987"],
        ["tiny-doc-01"],
        "1987 [[tiny-doc-01]]",
    ),
    (
        "tiny-q02",
        "single-hop",
        "What is the combined capacity of the Kestrel Tidal Array?",
        ["42 megawatts", "42 MW"],
        ["tiny-doc-04"],
        "42 megawatts [[tiny-doc-04]]",
    ),
    (
        "tiny-q03",
        "single-hop",
        "Who founded Orrex Maritime?",
        ["Caspar Orrex"],
        ["tiny-doc-06"],
        "Caspar Orrex [[tiny-doc-06]]",
    ),
    (
        "tiny-q04",
        "single-hop",
        "On which island is the Saltmarsh Observatory located?",
        ["Corrin Isle"],
        ["tiny-doc-09"],
        "Corrin Isle [[tiny-doc-09]]",
    ),
    # multi-hop
    (
        "tiny-q05",
        "multi-hop",
        "At which university did the founder of the Halvren Institute of Tidal Engineering study?",
        ["Quellmoor University"],
        ["tiny-doc-01", "tiny-doc-02"],
        "Quellmoor University [[tiny-doc-01]] [[tiny-doc-02]]",
    ),
    (
        "tiny-q06",
        "multi-hop",
        "In which city is the university attended by the founder of "
        "the Halvren Institute of Tidal Engineering located?",
        ["Brindlehaven"],
        ["tiny-doc-01", "tiny-doc-02", "tiny-doc-03"],
        "Brindlehaven [[tiny-doc-01]] [[tiny-doc-02]] [[tiny-doc-03]]",
    ),
    (
        "tiny-q07",
        "multi-hop",
        "Which organisation did the current director of the institute "
        "that built the Kestrel Tidal Array lead before becoming director?",
        ["Saltmarsh Observatory", "the Saltmarsh Observatory"],
        ["tiny-doc-04", "tiny-doc-05"],
        "the Saltmarsh Observatory [[tiny-doc-04]] [[tiny-doc-05]]",
    ),
    (
        "tiny-q08",
        "multi-hop",
        "In which body of water is the tidal array whose effect on "
        "grey seals Wren Calloway studied?",
        ["Varn Strait", "the Varn Strait"],
        ["tiny-doc-10", "tiny-doc-04"],
        "the Varn Strait [[tiny-doc-10]] [[tiny-doc-04]]",
    ),
    # comparison
    (
        "tiny-q09",
        "comparison",
        "Which has the larger capacity, the Kestrel Tidal Array or the Lumen Bay Tidal Array?",
        ["Kestrel Tidal Array", "the Kestrel Tidal Array"],
        ["tiny-doc-04", "tiny-doc-07"],
        "the Kestrel Tidal Array [[tiny-doc-04]] [[tiny-doc-07]]",
    ),
    (
        "tiny-q10",
        "comparison",
        "Which was founded earlier, the Halvren Institute of Tidal Engineering or Orrex Maritime?",
        ["Orrex Maritime"],
        ["tiny-doc-01", "tiny-doc-06"],
        "Orrex Maritime [[tiny-doc-01]] [[tiny-doc-06]]",
    ),
    (
        "tiny-q11",
        "comparison",
        "Which city has the larger population, Port Ossery or Brindlehaven?",
        ["Brindlehaven"],
        ["tiny-doc-08", "tiny-doc-12"],
        "Brindlehaven [[tiny-doc-08]] [[tiny-doc-12]]",
    ),
    # temporal
    (
        "tiny-q12",
        "temporal",
        "Who was director of the Halvren Institute of Tidal Engineering "
        "when the Kestrel Tidal Array was commissioned?",
        ["Tobin Arkwright"],
        ["tiny-doc-04", "tiny-doc-05"],
        "Tobin Arkwright [[tiny-doc-04]] [[tiny-doc-05]]",
    ),
    (
        "tiny-q13",
        "temporal",
        "For how many years was the Ossery Accord in force?",
        ["17 years", "17"],
        ["tiny-doc-11"],
        "17 years [[tiny-doc-11]]",
    ),
    (
        "tiny-q14",
        "temporal",
        "Was the Ossery Accord still in force when the Lumen Bay Tidal Array was commissioned?",
        ["yes"],
        ["tiny-doc-07", "tiny-doc-11"],
        "yes [[tiny-doc-07]] [[tiny-doc-11]]",
    ),
    # global
    (
        "tiny-q15",
        "global",
        "What are the main organisations working on tidal energy in this "
        "collection, and how are they connected?",
        [
            "The Halvren Institute of Tidal Engineering, which built the Kestrel Tidal Array, and "
            "Orrex Maritime, which built the Lumen Bay Tidal Array, shared data under the Ossery "
            "Accord from 1999 to 2016 and have since been competitors; the Saltmarsh Observatory "
            "studies the effect of the arrays on marine wildlife."
        ],
        ["tiny-doc-01", "tiny-doc-04", "tiny-doc-06", "tiny-doc-07", "tiny-doc-09", "tiny-doc-11"],
        "The Halvren Institute of Tidal Engineering built the Kestrel Tidal Array "
        "[[tiny-doc-01]] [[tiny-doc-04]] and Orrex Maritime built the Lumen Bay Tidal Array "
        "[[tiny-doc-06]] [[tiny-doc-07]]. They shared data under the Ossery Accord from 1999 to "
        "2016 and have since competed [[tiny-doc-11]]. The Saltmarsh Observatory studies the "
        "effect of the arrays on marine wildlife [[tiny-doc-09]].",
    ),
    (
        "tiny-q16",
        "global",
        "What recurring themes run through the history of tidal energy in the Varn Strait region?",
        [
            "Cooperation followed by rivalry between the Halvren Institute and Orrex Maritime, "
            "leadership succession at the Halvren Institute, and monitoring of the effect of tidal "
            "turbines on wildlife by the Saltmarsh Observatory."
        ],
        ["tiny-doc-04", "tiny-doc-05", "tiny-doc-07", "tiny-doc-10", "tiny-doc-11"],
        "Three themes recur: cooperation and then rivalry between the Halvren Institute and "
        "Orrex Maritime [[tiny-doc-11]] as each built its own array [[tiny-doc-04]] "
        "[[tiny-doc-07]]; leadership succession at the Halvren Institute [[tiny-doc-05]]; and "
        "monitoring of the effect of the turbines on seals by the Saltmarsh Observatory "
        "[[tiny-doc-10]].",
    ),
    (
        "tiny-q17",
        "global",
        "Summarise the main people in the collection and the roles they "
        "played in regional tidal energy.",
        [
            "Mirela Danthorpe founded the Halvren Institute; Tobin Arkwright directed it while the "
            "Kestrel Tidal Array was built; Ysolde Fenwick moved from the Saltmarsh Observatory to "
            "direct it; Caspar Orrex founded Orrex Maritime; Wren Calloway studied the effect of "
            "the Kestrel Tidal Array on grey seals."
        ],
        ["tiny-doc-02", "tiny-doc-05", "tiny-doc-06", "tiny-doc-10"],
        "Mirela Danthorpe founded the Halvren Institute [[tiny-doc-02]] [[tiny-doc-05]]. Tobin "
        "Arkwright directed it during construction of its largest project and Ysolde Fenwick "
        "moved from the Saltmarsh Observatory to succeed him [[tiny-doc-05]]. Caspar Orrex "
        "founded Orrex Maritime [[tiny-doc-06]]. Wren Calloway studied how the Kestrel Tidal "
        "Array affected grey seals [[tiny-doc-10]].",
    ),
    # null (unanswerable: answer absent from corpus; near-miss distractors exist)
    (
        "tiny-q18",
        "null",
        "What was the construction cost of the Kestrel Tidal Array?",
        [],
        [],
        REFUSAL,
    ),
    (
        "tiny-q19",
        "null",
        "Who succeeded Ysolde Fenwick as director of the Halvren Institute of Tidal Engineering?",
        [],
        [],
        REFUSAL,
    ),
    ("tiny-q20", "null", "In what year did Wren Calloway receive her doctorate?", [], [], REFUSAL),
]

# Closed-book (V0) responses: refusal by default; a few confident hallucinations so the
# judge and EM see wrong answers.
CLOSED_BOOK_HALLUCINATIONS = {
    "tiny-q01": "1975",
    "tiny-q02": "50 megawatts",
    "tiny-q10": "the Halvren Institute of Tidal Engineering",
}

# Partially correct answers for the global questions (covers part of the gold summary).
PARTIAL_ANSWERS = {
    "tiny-q15": "The Halvren Institute of Tidal Engineering built the Kestrel Tidal Array "
    "[[tiny-doc-04]].",
    "tiny-q16": "The Halvren Institute and Orrex Maritime first cooperated and then became "
    "rivals [[tiny-doc-11]].",
    "tiny-q17": "Mirela Danthorpe founded the Halvren Institute [[tiny-doc-02]].",
}


# --------------------------------------------------------------------------- helpers
def strip_citations(text: str) -> str:
    return re.sub(r"\s+", " ", CITE_RE.sub("", text)).strip().replace(" .", ".").replace(" ;", ";")


def cited(text: str) -> list[str]:
    seen: list[str] = []
    for doc_id in CITE_RE.findall(text):
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def lightrag_text(entities: list[E], relations: list[R]) -> str:
    rows = [TUPLE_DELIM.join(("entity", n, t, d)) for n, t, d in entities]
    rows += [TUPLE_DELIM.join(("relation", s, t, k, d)) for s, t, k, d, _ in relations]
    return "\n".join([*rows, COMPLETE])


def judge_json(label: str, rationale: str) -> str:
    return json.dumps({"label": label, "rationale": rationale}, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- build
def build() -> None:
    docs = {d: (t, x) for d, t, x in DOCS}
    corpus = [{"doc_id": d, "title": t, "text": x, "source": SOURCE} for d, t, x in DOCS]

    questions: list[dict[str, Any]] = []
    for qid, qtype, text, answers, gold, _ in QUESTIONS:
        questions.append(
            {
                "question_id": qid,
                "question": text,
                "answers": answers,
                "gold_passage_ids": gold,
                "query_type": qtype,
                "answerable": qtype != "null",
                "split": None,
            }
        )

    extraction: list[dict[str, Any]] = []
    for doc_id, (_title, text) in docs.items():
        ents, rels = EXTRACTION[doc_id]
        extraction.append(
            {
                "doc_id": doc_id,
                "match": text.split(". ")[0],
                "prompt_version": "lightrag-1.5.7-default",
                "response": lightrag_text(ents, rels),
                "entities": [{"name": n, "type": t, "description": d} for n, t, d in ents],
                "relations": [
                    {"source": s, "target": t, "relation": k, "description": d, "weight": w}
                    for s, t, k, d, w in rels
                ],
            }
        )

    generation: list[dict[str, Any]] = []
    judge: list[dict[str, Any]] = []

    def add(qid: str, variant: str, template: str, label: str, rationale: str) -> None:
        answer = strip_citations(template)
        generation.append(
            {
                "question_id": qid,
                "variant": variant,
                "response_template": template,
                "answer": answer,
                "cited_doc_ids": cited(template),
            }
        )
        judge.append(
            {
                "question_id": qid,
                "variant": variant,
                "prediction": answer,
                "response": judge_json(label, rationale),
            }
        )

    for qid, qtype, _, answers, _, template in QUESTIONS:
        answerable = qtype != "null"
        add(
            qid,
            "rag",
            template,
            "correct",
            "Matches the reference answer."
            if answerable
            else "The question is unanswerable from the corpus and the answer refuses.",
        )
        if answerable:
            add(
                qid,
                "refusal",
                REFUSAL,
                "wrong",
                "The answer refuses although the corpus contains the answer.",
            )
        cb = CLOSED_BOOK_HALLUCINATIONS.get(qid, REFUSAL)
        if cb == REFUSAL:
            label = "wrong" if answerable else "correct"
            why = (
                "Refuses an answerable question."
                if answerable
                else "Correctly refuses an unanswerable question."
            )
        else:
            label, why = "wrong", f"Contradicts the reference answer {answers[0]!r}."
        add(qid, "closed_book", cb, label, why)
        if qid in PARTIAL_ANSWERS:
            add(
                qid,
                "partial",
                PARTIAL_ANSWERS[qid],
                "partial",
                "Correct but covers only part of the reference answer.",
            )

    malformed = [
        {
            "role": "extraction",
            "doc_id": "tiny-doc-03",
            "description": "Truncated output: wrong tuple delimiter and no completion marker.",
            "response": "entity<|>Quellmoor University<|>Organization\nrelation<|#|>Quellmoor",
        },
        {
            "role": "judge",
            "question_id": "tiny-q01",
            "description": "Not JSON and no valid label; must be recorded as unjudged.",
            "response": "The answer looks fine to me, probably right?",
        },
        {
            "role": "judge",
            "question_id": "tiny-q02",
            "description": "Valid JSON with a label outside {correct, partial, wrong}.",
            "response": judge_json("mostly-right", "Close enough."),
        },
        {
            "role": "generation",
            "question_id": "tiny-q03",
            "description": "Cites passage 99, which is never in the context; record as invalid.",
            "response": "Caspar Orrex [99]",
        },
    ]

    counts = Counter(q["query_type"] for q in questions)
    manifest = {
        "fixture_version": FIXTURE_VERSION,
        "source": SOURCE,
        "refusal_string": REFUSAL,
        "num_documents": len(corpus),
        "num_gold_documents": len({g for q in questions for g in q["gold_passage_ids"]}),
        "num_questions": len(questions),
        "query_type_counts": dict(sorted(counts.items())),
        "num_unanswerable": sum(not q["answerable"] for q in questions),
        "extraction_format": {
            "style": "LightRAG 1.5.x delimited rows",
            "tuple_delimiter": TUPLE_DELIM,
            "completion_delimiter": COMPLETE,
            "entity_row": "entity<|#|>name<|#|>type<|#|>description",
            "relation_row": "relation<|#|>source<|#|>target<|#|>keywords<|#|>description",
        },
        "citation_placeholder": "[[doc_id]]",
        "judge_labels": ["correct", "partial", "wrong"],
        "generation_variants": ["rag", "refusal", "closed_book", "partial"],
        "files": {
            "corpus": "corpus.jsonl",
            "questions": "questions.jsonl",
            "extraction": "fake_llm/extraction.jsonl",
            "generation": "fake_llm/generation.jsonl",
            "judge": "fake_llm/judge.jsonl",
            "malformed": "fake_llm/malformed.jsonl",
        },
    }

    write_jsonl(HERE / "corpus.jsonl", corpus)
    write_jsonl(HERE / "questions.jsonl", questions)
    write_jsonl(HERE / "fake_llm" / "extraction.jsonl", extraction)
    write_jsonl(HERE / "fake_llm" / "generation.jsonl", generation)
    write_jsonl(HERE / "fake_llm" / "judge.jsonl", judge)
    write_jsonl(HERE / "fake_llm" / "malformed.jsonl", malformed)
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    build()
