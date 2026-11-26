from __future__ import annotations

import pickle

import pytest

from elevate.core.errors import (
    BackendError,
    BudgetExceeded,
    ConfigError,
    DatasetError,
    ElevateError,
    EmbeddingMismatchError,
    ModelCallError,
    TuningOnTestError,
)

ALL_ERRORS: list[ElevateError] = [
    ConfigError(file="configs/x.yaml", field="retrieval.top_k", expected="int"),
    BackendError(backend="Qdrant", reason="connection refused"),
    ModelCallError(model="gpt-4o-mini-2024-07-18", reason="HTTP 503", attempts=5),
    BudgetExceeded(spent_usd=50.0, cap_usd=50.0, remaining_items=120),
    DatasetError(
        file="data/raw/musique/dev.jsonl", missing_field="paragraphs", dataset_section="musique"
    ),
    EmbeddingMismatchError(query_model="bge-small-en-v1.5", index_model="bge-base-en-v1.5"),
    TuningOnTestError(operation="threshold", split="test"),
]


def test_config_error_names_file_field_and_type():
    msg = str(ALL_ERRORS[0])
    assert "configs/x.yaml" in msg and "retrieval.top_k" in msg and "int" in msg


def test_backend_error_names_backend_and_setup_section():
    err = BackendError(backend="Neo4j", reason="timeout")
    assert "Neo4j" in str(err) and "docs/SETUP.md#neo4j" in str(err)


def test_model_call_error_names_model_and_reason():
    msg = str(ALL_ERRORS[2])
    assert "gpt-4o-mini-2024-07-18" in msg and "HTTP 503" in msg


def test_budget_exceeded_reports_spent_and_remaining():
    msg = str(ALL_ERRORS[3])
    assert "$50.0000" in msg and "120" in msg


def test_dataset_error_names_file_field_and_doc_section():
    msg = str(ALL_ERRORS[4])
    assert "dev.jsonl" in msg and "paragraphs" in msg and "docs/DATASETS.md#musique" in msg


def test_embedding_mismatch_names_both_models():
    msg = str(ALL_ERRORS[5])
    assert "bge-small-en-v1.5" in msg and "bge-base-en-v1.5" in msg


def test_tuning_on_test_states_dev_restriction():
    assert "restricted to dev" in str(ALL_ERRORS[6])


@pytest.mark.parametrize("err", ALL_ERRORS, ids=lambda e: type(e).__name__)
def test_errors_share_base_and_survive_pickling(err: ElevateError):
    assert isinstance(err, ElevateError)
    clone = pickle.loads(pickle.dumps(err))
    assert type(clone) is type(err) and str(clone) == str(err)
