"""Live API smoke test (costs well under $0.001). Run with: make test-int"""

import os

import pytest
from dotenv import load_dotenv

from hgrag.config import PROJECT_ROOT, AppConfig

load_dotenv(PROJECT_ROOT / ".env")

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_haiku_responds_and_is_metered(config: AppConfig) -> None:
    from hgrag.core.runs import RunContext
    from hgrag.llm.gateway import build_gateway

    gateway = build_gateway(config, RunContext(run_id="int-ping"))
    response = gateway.complete("smoke", "Reply with the single word: pong", max_tokens=16)
    assert response.status == "ok"
    assert "pong" in response.text.lower()
    assert 0 < response.cost_usd < 0.001
