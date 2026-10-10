"""Probe the handwriting API /health endpoint.

Examples:
  uv run python scripts/health_check.py
  uv run python scripts/health_check.py --url https://handwriting-api-687921010800.europe-central2.run.app
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def check_health(base_url: str, timeout: float) -> tuple[int, dict | str]:
    url = base_url.rstrip("/") + "/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except urllib.error.URLError as exc:
        return 0, str(exc.reason)

    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def main() -> None:
    parser = argparse.ArgumentParser(description="Health check for the handwriting API.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:5000",
        help="API base URL (default: http://127.0.0.1:5000)",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    args = parser.parse_args()

    status, payload = check_health(args.url, args.timeout)
    if status == 200 and isinstance(payload, dict) and payload.get("status") == "ok":
        print(f"healthy: {args.url}/health")
        print(json.dumps(payload, indent=2))
        raise SystemExit(0)

    print(f"unhealthy: {args.url}/health", file=sys.stderr)
    if status:
        print(f"HTTP {status}", file=sys.stderr)
    print(payload, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
