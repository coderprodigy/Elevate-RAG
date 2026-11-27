"""Run a native framework against deterministic loopback model responses."""

import argparse
import json
import re
import threading
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from elevate_rag.interop import export_reference, run_reference


def reply(prompt: str, final_message: str) -> str:
    match = re.findall(r"Station ([A-Z][a-z]+)", final_message)
    name = match[-1] if match else "Aster"
    answer = f"Beacon {name}"
    if "Many entities were missed" in final_message or "still entities" in final_message:
        return "<|COMPLETE|>"
    if "relationship_strength" in prompt and "entity_description" in prompt:
        name = name.upper()
        return (
            f'("entity"<|>STATION {name}<|>ORGANIZATION<|>Station that maintains a beacon)##'
            f'("entity"<|>BEACON {name}<|>OBJECT<|>Beacon maintained by the station)##'
            f'("relationship"<|>STATION {name}<|>BEACON {name}<|>The station maintains it<|>9)'
            "<|COMPLETE|>"
        )
    if "rating_explanation" in prompt:
        return json.dumps(
            {
                "title": "Station and beacon",
                "summary": "Station Aster maintains Beacon Aster.",
                "rating": 7.0,
                "rating_explanation": "Fixture evidence",
                "findings": [{"summary": "Beacon", "explanation": "Maintained by station"}],
            }
        )
    if '"points"' in prompt:
        return json.dumps({"points": [{"description": answer, "score": 80}]})
    if '"triples"' in prompt:
        return json.dumps({"triples": [[f"Station {name}", "maintains", answer]]})
    if '"named_entities"' in prompt:
        return json.dumps({"named_entities": [f"Station {name}", answer]})
    return f"Thought: The context identifies the station beacon.\nAnswer: {answer}"


class FixtureHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = payload.get("model", "fixture")
        if self.path == "/v1/embeddings":
            inputs = payload["input"]
            if isinstance(inputs, str) or (inputs and isinstance(inputs[0], int)):
                inputs = [inputs]
            rows = [
                {
                    "index": i,
                    "object": "embedding",
                    "embedding": [value / 255 for value in sha256(str(text).encode()).digest()],
                }
                for i, text in enumerate(inputs)
            ]
            self.send_body(
                {
                    "object": "list",
                    "model": model,
                    "data": rows,
                    "usage": {"prompt_tokens": len(rows), "total_tokens": len(rows)},
                }
            )
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        messages = [str(message["content"]) for message in payload["messages"]]
        answer = reply("\n".join(messages), messages[-1])
        header = {"id": "chatcmpl-fixture", "created": 0, "model": model}
        index = {"index": 0, "finish_reason": "stop"}
        if payload.get("stream"):
            pieces = [
                {
                    **header,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": answer},
                            "finish_reason": None,
                        }
                    ],
                },
                {**header, "object": "chat.completion.chunk", "choices": [{**index, "delta": {}}]},
            ]
            body = (
                "".join(f"data: {json.dumps(piece)}\n\n" for piece in pieces) + "data: [DONE]\n\n"
            ).encode()
            self.send_body(body, "text/event-stream")
        else:
            self.send_body(
                {
                    **header,
                    "object": "chat.completion",
                    "choices": [{**index, "message": {"role": "assistant", "content": answer}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )

    def send_body(self, value, kind="application/json"):
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=("microsoft-graphrag", "hipporag2"))
    parser.add_argument("version")
    parser.add_argument("python", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--port", type=int, default=28710)
    args = parser.parse_args()
    dataset = Path("data/synthetic")
    export = args.out / "inputs"
    export_reference(dataset, export, "dev", framework=args.framework, limit=2, seed=42)
    with ThreadingHTTPServer(("127.0.0.1", args.port), FixtureHandler) as server:
        address = f"http://127.0.0.1:{server.server_port}/v1"
        settings = args.out / "settings.toml"
        settings.write_text(
            f'framework_version = "{args.version}"\nchat_url = "{address}"\n'
            f'chat_model = "gpt-4o-mini"\nembedding_url = "{address}"\n'
            'embedding_model = "text-embedding-3-small"\nembedding_dimensions = 32\n'
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            run_reference(dataset, export, args.out / "run", args.python, settings, timeout=150)
            rows = (args.out / "run" / "native-questions.jsonl").read_text().splitlines()
            if len(rows) != 2 or any(json.loads(row)["status"] != "ok" for row in rows):
                raise RuntimeError("native framework did not complete both fixture questions")
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()
