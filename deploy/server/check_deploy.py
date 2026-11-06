"""Acceptance check for the deployed demo (docs/INSTALL.md, server step 7). Standard library only.

python deploy/server/check_deploy.py https://<your-domain>/medrag
The demo password is read from MEDRAG_DEMO_PASSWORD or prompted (never printed); user name defaults to "demo".
"""

from __future__ import annotations

import base64
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request

QUESTIONS = [
    "Can warfarin be taken with aspirin or NSAIDs such as naproxen?",
    "How should gabapentin be titrated when starting treatment for postherpetic neuralgia?",
]


def call(url: str, auth: str | None, body: dict | None = None, timeout: float = 120) -> tuple[int, bytes, float]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if auth:
        req.add_header("Authorization", "Basic " + auth)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), time.perf_counter() - start
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.perf_counter() - start


def main() -> int:
    base = sys.argv[1].rstrip("/")
    user = os.environ.get("MEDRAG_DEMO_USER", "demo")
    password = os.environ.get("MEDRAG_DEMO_PASSWORD") or getpass.getpass(f"password for {user}: ")
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    origin = base.rsplit("/", 1)[0]
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(("PASS " if ok else "FAIL ") + name + (f" - {detail}" if detail else ""), flush=True)

    status, _, _ = call(base + "/health", None)
    check("401 without password", status == 401, str(status))
    status, _, _ = call(base + "/health", base64.b64encode(f"{user}:wrong-password".encode()).decode())
    check("401 with a wrong password", status == 401, str(status))
    status, raw, _ = call(base + "/health", auth)
    health = json.loads(raw or b"{}") if status == 200 else {}
    check("health green", status == 200 and health.get("opensearch", {}).get("status") in ("green", "yellow"),
          f"{status} {health.get('opensearch', {}).get('status')}")  # fmt: skip
    status, raw, _ = call(base + "/", auth)
    check("chat UI loads with relative assets", status == 200 and b'src="./assets/' in raw, str(status))
    status, raw, secs = call(base + "/search", auth, {"query": "gabapentin mechanism of action", "k": 8})
    hits = json.loads(raw).get("passages", []) if status == 200 else []
    check("search returns 8 passages", status == 200 and len(hits) == 8, f"{status}, {secs:.2f}s")
    for q in QUESTIONS:
        status, raw, secs = call(base + "/ask", auth, {"question": q, "k": 8}, timeout=180)
        a = json.loads(raw) if status == 200 else {}
        cited = a.get("citations", [])
        check(f"ask: {q[:45]}...", status == 200 and not a.get("abstained") and bool(cited),
              f"{status}, {len(cited)} citations, {secs:.1f}s")  # fmt: skip
        if a.get("answer"):
            print("   " + a["answer"].replace("\n", "\n   "))
    status, _, _ = call(origin + "/asr/health", None, timeout=20)
    check("/asr/ still routed (not the maintenance page)", status in (200, 401, 404), str(status))
    status, raw, _ = call(origin + "/", None, timeout=20)
    check("maintenance page still served at /", status == 200, str(status))
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
