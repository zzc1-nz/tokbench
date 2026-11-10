"""One-shot generator for ``tasks/coding_bench.jsonl``.

The JSONL is the committed artefact; this script exists so the *tool payloads*
(the 800-1500 token blobs that the ``compress_tool_output`` strategy ablates)
can be assembled from realistic data structures and pretty-printed the way a
real tool response would arrive, instead of being typed by hand as one giant
escaped string.

Run from the ``tokbench`` project root::

    python scripts/build_taskset.py

It is idempotent: the prompts are literal strings below, so re-running produces
byte-identical output.  ``scripts/validate_tasks.py`` is the authoritative
constraint check; this script only assembles and prints a summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tasks" / "coding_bench.jsonl"


def blob(obj: Any) -> str:
    """Serialise a tool response the way a CLI would print it (indent=2)."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Tool payload 1 — failing CI run (pytest + git diff)
# --------------------------------------------------------------------------- #
def payload_ci_run() -> str:
    failures = [
        {
            "nodeid": "tests/test_outbox.py::test_publish_is_idempotent",
            "outcome": "failed",
            "duration_s": 12.41,
            "traceback": [
                "tests/test_outbox.py:88: in test_publish_is_idempotent",
                "    assert dispatcher.publish(order) == dispatcher.publish(order)",
                "E   AssertionError: assert {'event_id': 'evt_01J9K2...'} == "
                "{'event_id': 'evt_01J9K3...'}",
                "E     Differing items:",
                "E     {'event_id': 'evt_01J9K2Q7X3M4P8R'} != {'event_id': 'evt_01J9K3B1N6T0V2W'}",
                "E     Use -v to get more diff",
                "",
                "During handling of the above exception, another exception occurred:",
                "tests/test_outbox.py:92: in test_publish_is_idempotent",
                "    assert row.status == 'pending'",
                "E   KeyError: 'status'",
            ],
            "captured_stdout": [
                "dispatcher: acquiring advisory lock pg_advisory_xact_lock(889231)",
                "dispatcher: lock acquired after 0.004s",
                "dispatcher: inserting outbox row event=evt_01J9K2Q7X3M4P8R",
                "dispatcher: inserting outbox row event=evt_01J9K3B1N6T0V2W",
                "dispatcher: commit",
            ],
            "captured_stderr": ["WARNING sqlalchemy.pool: connection reset (attempt 1/3)"],
        },
        {
            "nodeid": "tests/test_outbox.py::test_retry_after_transient_failure",
            "outcome": "failed",
            "duration_s": 4.02,
            "traceback": [
                "tests/test_outbox.py:141: in test_retry_after_transient_failure",
                "    assert len(published) == 1",
                "E   AssertionError: assert 2 == 1",
                "E    +  where 2 = len([<PublishedEvent evt_a>, <PublishedEvent evt_b>])",
            ],
            "captured_stdout": [
                "publisher: attempt 1 failed with OperationalError, retrying in 0.05s",
                "publisher: attempt 2 succeeded",
                "publisher: attempt 2 succeeded (duplicate side effect)",
            ],
        },
        {
            "nodeid": "tests/test_outbox.py::test_consumer_ack_after_partial_batch",
            "outcome": "failed",
            "duration_s": 31.77,
            "traceback": [
                "tests/test_outbox.py:203: in test_consumer_ack_after_partial_batch",
                "    await consumer.drain(timeout=30)",
                "E   TimeoutError: consumer did not drain 7 of 40 messages within 30.0s",
            ],
            "captured_stderr": [
                "consumer: batch=40 acked=33 nacked=0 in_flight=7",
                "consumer: heartbeat missed 2 intervals (last_heartbeat_age=12.4s)",
            ],
        },
        {
            "nodeid": "tests/test_orders_api.py::test_create_order_returns_201",
            "outcome": "failed",
            "duration_s": 0.61,
            "traceback": [
                "tests/test_orders_api.py:57: in test_create_order_returns_201",
                "    assert response.status_code == 201",
                "E   assert 500 == 201",
                "E    +  where 500 = <Response [500 Internal Server Error]>.status_code",
            ],
            "captured_stderr": [
                "ERROR app.middleware.error_handler: unhandled exception "
                "psycopg.errors.UniqueViolation: duplicate key value violates unique "
                'constraint "outbox_dedupe_key_key"',
            ],
        },
    ]
    return blob(
        {
            "provider": "github-actions",
            "run_id": 8812345567,
            "workflow": ".github/workflows/ci.yml",
            "job": "test (3.12)",
            "status": "failure",
            "attempt": 1,
            "trigger": {
                "event": "pull_request",
                "branch": "feat/order-events-outbox",
                "base_branch": "main",
                "sha": "4f2c9ab7d1e0c3a95b6f2d8e7c104a3b9f8e2d61",
                "actor": "dpetrov",
            },
            "environment": {"python": "3.12.4", "os": "ubuntu-24.04", "postgres": "16.3"},
            "steps": [
                {"name": "pip install -e .[dev]", "conclusion": "success", "duration_s": 41.9},
                {"name": "alembic upgrade head", "conclusion": "success", "duration_s": 2.6},
                {"name": "pytest -q -x --timeout=60", "conclusion": "failure", "duration_s": 218.3},
            ],
            "test_summary": {
                "passed": 412,
                "failed": 4,
                "skipped": 3,
                "errors": 0,
                "duration_s": 214.7,
                "warnings": 11,
            },
            "failures": failures,
            "git_diff": {
                "base_sha": "9d1f7e2c04b85a6e3f0d1c9b7a2e5f804c3d6b19",
                "files": [
                    {
                        "path": "app/outbox/dispatcher.py",
                        "additions": 64,
                        "deletions": 12,
                        "patch": (
                            "@@ -41,9 +41,17 @@ class OutboxDispatcher:\n"
                            "     async def publish(self, order: Order) -> PublishedEvent:\n"
                            "-        row = OutboxRow(order_id=order.id, payload=order.to_json())\n"
                            "-        await self.session.commit()\n"
                            "-        return PublishedEvent(event_id=row.event_id)\n"
                            '+        dedupe_key = f"order:{order.id}:created"\n'
                            "+        row = await self.repo.find_by_dedupe_key(dedupe_key)\n"
                            "+        if row is None:\n"
                            "+            row = OutboxRow(\n"
                            "+                order_id=order.id,\n"
                            "+                dedupe_key=dedupe_key,\n"
                            "+                payload=order.to_json(),\n"
                            "+            )\n"
                            "+            self.session.add(row)\n"
                            "+            await self.session.commit()\n"
                            "+        return PublishedEvent(event_id=row.event_id)\n"
                        ),
                    },
                    {
                        "path": "app/outbox/repository.py",
                        "additions": 38,
                        "deletions": 0,
                        "patch": (
                            "@@ -0,0 +1,12 @@\n"
                            "+class OutboxRepository:\n"
                            "+    async def find_by_dedupe_key(self, key: str) -> OutboxRow | None:\n"
                            "+        stmt = select(OutboxRow).where(OutboxRow.dedupe_key == key)\n"
                            "+        result = await self.session.execute(stmt)\n"
                            "+        return result.scalar_one_or_none()\n"
                        ),
                    },
                    {
                        "path": "migrations/versions/0042_outbox_dedupe.py",
                        "additions": 27,
                        "deletions": 0,
                        "patch": (
                            "@@ -0,0 +1,7 @@\n"
                            "+def upgrade() -> None:\n"
                            '+    op.add_column("outbox", sa.Column("dedupe_key", sa.Text()))\n'
                            "+    op.create_unique_constraint(\n"
                            '+        "outbox_dedupe_key_key", "outbox", ["dedupe_key"]\n'
                            "+    )\n"
                        ),
                    },
                ],
                "stat": "4 files changed, 184 insertions(+), 16 deletions(-)",
            },
        }
    )


# --------------------------------------------------------------------------- #
# Tool payload 2 — monorepo build failure (file listing + manifests + tsc)
# --------------------------------------------------------------------------- #
def payload_ts_build() -> str:
    files = [
        "package.json",
        "pnpm-workspace.yaml",
        "tsconfig.base.json",
        "packages/ui/package.json",
        "packages/ui/tsconfig.json",
        "packages/ui/src/index.ts",
        "packages/ui/src/button.tsx",
        "packages/ui/src/theme.ts",
        "packages/ui/dist/index.d.ts",
        "packages/tokens/package.json",
        "packages/tokens/tsconfig.json",
        "packages/tokens/src/index.ts",
        "packages/tokens/src/colors.ts",
        "packages/tokens/dist/index.js",
        "packages/tokens/dist/index.d.ts",
        "apps/console/package.json",
        "apps/console/tsconfig.json",
        "apps/console/vite.config.ts",
        "apps/console/src/main.tsx",
        "apps/console/src/App.tsx",
        "apps/console/src/components/OrderTable.tsx",
        "apps/console/src/components/OrderTable.test.tsx",
        "apps/console/src/lib/api.ts",
        "apps/console/src/lib/format.ts",
        "apps/console/src/routes/orders.tsx",
        "apps/console/src/routes/settings.tsx",
        "apps/console/tsconfig.node.json",
        "node_modules/.modules.yaml",
        "pnpm-lock.yaml",
    ]
    return blob(
        {
            "command": "pnpm -r --filter @acme/console build",
            "exit_code": 2,
            "duration_s": 18.4,
            "cwd": "/home/runner/work/console/console",
            "package_manager": {"name": "pnpm", "version": "9.12.3"},
            "node": "v22.9.0",
            "typescript": "5.6.3",
            "files": files,
            "root_package_json": {
                "name": "acme-console",
                "private": True,
                "packageManager": "pnpm@9.12.3",
                "scripts": {
                    "build": "pnpm -r build",
                    "test": "vitest run",
                    "typecheck": "tsc -b --pretty false",
                },
                "devDependencies": {
                    "typescript": "5.6.3",
                    "vite": "5.4.10",
                    "vitest": "2.1.4",
                    "@types/react": "18.3.12",
                    "@types/node": "22.9.0",
                },
            },
            "workspace_yaml": (
                "packages:\n  - 'apps/*'\n  - 'packages/*'\n  - '!packages/**/dist'\n"
            ),
            "app_package_json": {
                "name": "@acme/console",
                "version": "0.14.2",
                "type": "module",
                "dependencies": {
                    "react": "18.3.1",
                    "react-dom": "18.3.1",
                    "react-router-dom": "6.28.0",
                    "@tanstack/react-query": "5.59.16",
                    "@acme/ui": "workspace:*",
                    "@acme/tokens": "workspace:*",
                    "@acme/sdk": "0.9.1",
                },
                "devDependencies": {
                    "vite": "5.4.10",
                    "typescript": "5.6.3",
                    "@vitejs/plugin-react": "4.3.3",
                },
                "exports": {".": {"types": "./dist/index.d.ts", "import": "./dist/index.js"}},
            },
            "app_tsconfig": {
                "extends": "../../tsconfig.base.json",
                "compilerOptions": {
                    "composite": True,
                    "module": "ESNext",
                    "moduleResolution": "Bundler",
                    "jsx": "react-jsx",
                    "strict": True,
                    "noUncheckedIndexedAccess": True,
                    "paths": {
                        "@acme/ui": ["../../packages/ui/src/index.ts"],
                        "@acme/tokens": ["../../packages/tokens/src/index.ts"],
                    },
                    "references": [
                        {"path": "../../packages/ui"},
                        {"path": "../../packages/tokens"},
                    ],
                },
                "include": ["src"],
            },
            "base_tsconfig": {
                "compilerOptions": {
                    "target": "ES2022",
                    "lib": ["ES2022", "DOM", "DOM.Iterable"],
                    "declaration": True,
                    "declarationMap": True,
                    "sourceMap": True,
                    "skipLibCheck": True,
                    "esModuleInterop": True,
                    "isolatedModules": True,
                    "verbatimModuleSyntax": True,
                },
            },
            "ui_package_json": {
                "name": "@acme/ui",
                "version": "0.8.0",
                "type": "module",
                "main": "./dist/index.js",
                "types": "./dist/index.d.ts",
                "exports": {
                    ".": {"types": "./dist/index.d.ts", "import": "./dist/index.js"},
                    "./theme": {"types": "./dist/theme.d.ts", "import": "./dist/theme.js"},
                },
                "peerDependencies": {"react": ">=18"},
            },
            "tsc_output": [
                "src/components/OrderTable.tsx(14,28): error TS2307: Cannot find module "
                "'@acme/tokens' or its corresponding type declarations.",
                "src/components/OrderTable.tsx(31,9): error TS7006: Parameter 'row' implicitly "
                "has an 'any' type.",
                "src/lib/api.ts(8,23): error TS2307: Cannot find module '@acme/sdk' or its "
                "corresponding type declarations.",
                "src/routes/orders.tsx(22,11): error TS2322: Type 'string' is not assignable to "
                "type 'OrderStatus'.",
                "src/routes/settings.tsx(41,7): error TS2304: Cannot find name 'process'.",
                "vite.config.ts(3,18): error TS2307: Cannot find module 'vite' or its "
                "corresponding type declarations.",
                "Found 6 errors in 5 files.",
            ],
            "pnpm_ls": {
                "dependencies": {
                    "@acme/sdk": {"version": "0.9.1", "resolved": "link:../../packages/sdk"},
                    "@acme/tokens": {"version": "link:packages/tokens"},
                    "@acme/ui": {"version": "link:packages/ui"},
                    "react": {"version": "18.3.1"},
                    "react-router-dom": {"version": "6.28.0"},
                },
                "problems": [
                    "missing: @acme/sdk@0.9.1, required by @acme/console@0.14.2",
                    "extraneous: @acme/tokens@0.6.2 (dist-only, no source link)",
                ],
            },
            "notes": [
                "packages/tokens/package.json has no 'exports' field, only main/types",
                "packages/sdk does not exist in the workspace (removed in v0.14 refactor)",
                "apps/console/tsconfig.json references ../../packages/tokens but the package "
                "directory contains no tsconfig.json",
                "CI uses 'pnpm install --frozen-lockfile'; pnpm-lock.yaml was last updated "
                "on the release branch",
            ],
        }
    )


# --------------------------------------------------------------------------- #
# Tool payload 3 — Kubernetes CrashLoopBackOff investigation
# --------------------------------------------------------------------------- #
def payload_k8s_crashloop() -> str:
    return blob(
        {
            "command": "kubectl -n payments get pods,events -o json",
            "fetched_at": "2026-11-04T02:13:47Z",
            "pods": [
                {
                    "name": "ledger-writer-7d9f8c6b5-x2kq9",
                    "namespace": "payments",
                    "labels": {"app": "ledger-writer", "version": "v3.7.1"},
                    "node": "ip-10-42-7-118.ec2.internal",
                    "image": "registry.internal/payments/ledger-writer:v3.7.1",
                    "resources": {"requests": {"cpu": "250m", "memory": "512Mi"}},
                    "probes": {
                        "readiness": {"path": "/healthz", "periodSeconds": 5},
                        "liveness": {"path": "/livez", "initialDelaySeconds": 20},
                    },
                    "env_keys": ["DATABASE_URL", "KAFKA_BROKERS", "LOG_LEVEL"],
                    "phase": "Running",
                    "conditions": [
                        {"type": "Ready", "status": "False", "reason": "ContainersNotReady"},
                        {"type": "PodScheduled", "status": "True"},
                    ],
                    "restartCount": 9,
                    "state": {
                        "waiting": {
                            "reason": "CrashLoopBackOff",
                            "message": "back-off 5m0s restarting failed container",
                        }
                    },
                    "lastState": {
                        "terminated": {
                            "exitCode": 1,
                            "reason": "Error",
                            "startedAt": "2026-11-04T02:08:31Z",
                            "finishedAt": "2026-11-04T02:08:39Z",
                        }
                    },
                },
                {
                    "name": "ledger-writer-7d9f8c6b5-h4v7t",
                    "namespace": "payments",
                    "labels": {"app": "ledger-writer", "version": "v3.7.1"},
                    "phase": "Running",
                    "restartCount": 9,
                    "state": {"waiting": {"reason": "CrashLoopBackOff"}},
                    "lastState": {"terminated": {"exitCode": 1, "reason": "Error"}},
                },
                {
                    "name": "ledger-writer-5b6c7d8f9-nn2pl",
                    "namespace": "payments",
                    "labels": {"app": "ledger-writer", "version": "v3.6.4"},
                    "phase": "Running",
                    "restartCount": 0,
                    "state": {"running": {"startedAt": "2026-10-28T14:02:57Z"}},
                    "ready": True,
                },
            ],
            "events": [
                {
                    "type": "Normal",
                    "reason": "Scheduled",
                    "count": 1,
                    "message": "Successfully assigned payments/ledger-writer-7d9f8c6b5-x2kq9",
                    "lastTimestamp": "2026-11-04T01:58:12Z",
                },
                {
                    "type": "Warning",
                    "reason": "Unhealthy",
                    "count": 12,
                    "message": "Readiness probe failed: HTTP probe failed with statuscode: 503",
                    "lastTimestamp": "2026-11-04T02:07:58Z",
                },
                {
                    "type": "Normal",
                    "reason": "Killing",
                    "count": 4,
                    "message": "Container writer failed liveness probe, will be restarted",
                    "lastTimestamp": "2026-11-04T02:08:31Z",
                },
                {
                    "type": "Warning",
                    "reason": "BackOff",
                    "count": 9,
                    "message": "Back-off restarting failed container writer",
                    "lastTimestamp": "2026-11-04T02:08:40Z",
                },
            ],
            "previous_logs": [
                "02:08:31.114Z INFO  ledger-writer starting version=v3.7.1 commit=8c1d4e2",
                "02:08:31.402Z INFO  otel: exporter connected",
                "02:08:31.655Z INFO  kafka: connecting brokers=[kafka-0:9092 kafka-1:9092]",
                "02:08:31.902Z WARN  kafka: consumer group rebalance in progress",
                "02:08:32.441Z INFO  http: listening addr=:8080",
                "02:08:32.443Z INFO  readiness: probing database",
                "02:08:32.998Z ERROR db: ping failed: dial tcp 10.42.19.7:5432: "
                "connect: connection refused",
                "02:08:33.512Z ERROR db: ping failed (attempt 2/5): dial tcp 10.42.19.7:5432: "
                "connect: connection refused",
                "02:08:35.048Z ERROR db: ping failed (attempt 5/5): dial tcp 10.42.19.7:5432: "
                "connect: connection refused",
                "02:08:35.049Z ERROR readiness: database unreachable, serving 503",
                "02:08:35.050Z FATAL startup: migration check failed: could not connect to "
                "postgres after 5 attempts",
                "02:08:35.410Z INFO  exit code 1",
            ],
            "configmap_diff": {
                "configmap": "ledger-writer-config",
                "changed_keys": {
                    "DB_HOST": {
                        "old": "ledger-db-primary.payments.svc.cluster.local",
                        "new": "10.42.19.7",
                    },
                    "MIGRATE_ON_START": {"old": "false", "new": "true"},
                },
            },
            "deployment": {
                "name": "ledger-writer",
                "replicas": 3,
                "readyReplicas": 1,
                "updatedReplicas": 2,
                "image": "registry.internal/payments/ledger-writer:v3.7.1",
                "previous_image": "registry.internal/payments/ledger-writer:v3.6.4",
            },
        }
    )


# --------------------------------------------------------------------------- #
# Tool payload 4 — paginated REST API capture
# --------------------------------------------------------------------------- #
def payload_api_pages() -> str:
    def customer(i: int) -> dict[str, Any]:
        return {
            "id": f"cus_{i:06d}",
            "created": 1730505600 + i * 137,
            "email": f"user{i:04d}@example.com",
            "currency": "usd",
            "delinquent": i % 17 == 0,
            "metadata": {
                "plan": ["free", "pro", "team"][i % 3],
                "region": ["us", "eu", "apac"][i % 3],
            },
            "subscription": {
                "id": f"sub_{i:06d}",
                "status": ["active", "trialing", "past_due"][i % 3],
                "current_period_end": 1735000000 + i * 86400,
            },
        }

    return blob(
        {
            "captured_at": "2026-11-05T10:22:41Z",
            "api_version": "2024-06-20",
            "requests": [
                {
                    "url": "https://api.example-pay.com/v1/customers?limit=3",
                    "status": 200,
                    "latency_ms": 184,
                    "headers": {"request-id": "req_8f2a1c", "x-ratelimit-remaining": "87"},
                    "body": {
                        "object": "list",
                        "data": [customer(i) for i in range(1, 4)],
                        "has_more": True,
                        "next_cursor": "cus_000003",
                        "total_count": 12,
                    },
                },
                {
                    "url": "https://api.example-pay.com/v1/customers?limit=3&starting_after="
                    "cus_000003",
                    "status": 429,
                    "latency_ms": 12,
                    "headers": {
                        "request-id": "req_8f2a1e",
                        "retry-after": "2",
                        "x-ratelimit-remaining": "0",
                    },
                    "body": {
                        "error": {
                            "type": "rate_limit_error",
                            "code": "rate_limit_exceeded",
                            "message": "Too many requests. Retry after 2 seconds.",
                            "request_id": "req_8f2a1e",
                        }
                    },
                },
                {
                    "url": "https://api.example-pay.com/v1/customers?limit=3&starting_after="
                    "cus_000003",
                    "status": 200,
                    "latency_ms": 201,
                    "headers": {"request-id": "req_8f2a1f", "x-ratelimit-remaining": "99"},
                    "body": {
                        "object": "list",
                        "data": [customer(i) for i in range(4, 7)],
                        "has_more": True,
                        "next_cursor": "cus_000006",
                    },
                },
            ],
            "openapi_fragment": {
                "paths": {
                    "/v1/customers": {
                        "get": {
                            "parameters": [
                                {
                                    "name": "limit",
                                    "in": "query",
                                    "schema": {"type": "integer", "minimum": 1, "maximum": 100},
                                },
                                {
                                    "name": "starting_after",
                                    "in": "query",
                                    "schema": {"type": "string"},
                                },
                            ],
                            "responses": {
                                "200": {"description": "A paginated list of customers"},
                                "429": {"description": "Rate limited; honour Retry-After"},
                            },
                        }
                    }
                },
            },
            "observed_quirks": [
                "next_cursor is the id of the last item on the page, not an opaque token",
                "a 429 response has no body items; the client must resume from the same cursor",
                "retrying a 429 with the original cursor instead of next_cursor duplicated "
                "cus_000003 in the previous import",
                "total_count is only present on the first page",
            ],
        }
    )


# --------------------------------------------------------------------------- #
# Tool payload 5 — git history for release notes
# --------------------------------------------------------------------------- #
def payload_git_history() -> str:
    commits = [
        (
            "a1b2c3d",
            "Priya Raman",
            "2026-10-21",
            "feat(billing): add usage-based pricing tiers",
            12,
            640,
            118,
            ["feat"],
        ),
        (
            "b2c3d4e",
            "Priya Raman",
            "2026-10-21",
            "fix(billing): round proration to cents before persisting",
            3,
            41,
            9,
            ["fix"],
        ),
        (
            "c3d4e5f",
            "Marco Bianchi",
            "2026-10-22",
            "feat(api): cursor pagination for /v1/events",
            9,
            512,
            63,
            ["feat", "breaking"],
        ),
        (
            "d4e5f60",
            "Marco Bianchi",
            "2026-10-22",
            "test(api): cover pagination boundary conditions",
            2,
            188,
            0,
            ["test"],
        ),
        (
            "e5f6071",
            "Aiko Tanaka",
            "2026-10-23",
            "perf(db): replace offset scans in event replay with keyset queries",
            4,
            96,
            141,
            ["perf"],
        ),
        (
            "0718293",
            "Daniel Okoro",
            "2026-10-24",
            "fix(webhooks): retry deliveries on 5xx with exponential backoff",
            5,
            214,
            37,
            ["fix"],
        ),
        (
            "1829304",
            "Daniel Okoro",
            "2026-10-24",
            "fix(webhooks): do not retry on 4xx client errors",
            2,
            39,
            12,
            ["fix"],
        ),
        (
            "2930415",
            "Sofia Lindqvist",
            "2026-10-25",
            "feat(dashboard): latency breakdown by route and percentile",
            17,
            1204,
            88,
            ["feat"],
        ),
        (
            "4152637",
            "Priya Raman",
            "2026-10-26",
            "refactor(worker): extract job registry from main loop",
            8,
            388,
            402,
            ["refactor"],
        ),
        (
            "7485960",
            "Aiko Tanaka",
            "2026-10-27",
            "feat(auth): scoped API keys with per-resource permissions",
            14,
            903,
            121,
            ["feat", "security"],
        ),
        (
            "9607182",
            "Daniel Okoro",
            "2026-10-28",
            "feat(cli): benchmark command for prompt cost",
            6,
            341,
            19,
            ["feat"],
        ),
        (
            "4152637f",
            "Marco Bianchi",
            "2026-10-30",
            "feat(events): include idempotency_key in event payloads",
            7,
            274,
            44,
            ["feat"],
        ),
        (
            "5263748g",
            "Marco Bianchi",
            "2026-10-30",
            "fix(events): dedupe events by idempotency_key on ingest",
            3,
            129,
            17,
            ["fix"],
        ),
        (
            "9607182k",
            "Priya Raman",
            "2026-11-01",
            "revert: feat(billing): add usage-based pricing tiers",
            9,
            118,
            640,
            ["revert", "breaking"],
        ),
        (
            "0718293l",
            "Priya Raman",
            "2026-11-01",
            "fix(billing): guard usage aggregation behind a feature flag",
            4,
            97,
            23,
            ["fix"],
        ),
        (
            "1829304m",
            "Aiko Tanaka",
            "2026-11-02",
            "perf(db): add composite index on (account_id, created_at) for events",
            2,
            34,
            0,
            ["perf"],
        ),
        (
            "3041526o",
            "Marco Bianchi",
            "2026-11-02",
            "feat(api): return Retry-After on 429 responses",
            3,
            71,
            14,
            ["feat"],
        ),
    ]
    return blob(
        {
            "repo": "acme/platform",
            "range": "v2.14.0..HEAD",
            "previous_tag": "v2.14.0",
            "head_sha": "b7e1c94f2a6d0b83e5c17a4f9d2e6b0c3a8f5d71",
            "commit_count": len(commits),
            "authors": {
                "Priya Raman": 5,
                "Marco Bianchi": 5,
                "Aiko Tanaka": 4,
                "Daniel Okoro": 4,
                "Sofia Lindqvist": 3,
            },
            "commits": [
                {
                    "sha": sha,
                    "author": author,
                    "date": date,
                    "subject": subject,
                    "files_changed": files,
                    "insertions": ins,
                    "deletions": dele,
                    "labels": labels,
                }
                for sha, author, date, subject, files, ins, dele, labels in commits
            ],
            "pull_requests": [
                {
                    "number": 1481,
                    "title": "Usage-based pricing (behind flag)",
                    "labels": ["billing"],
                },
                {
                    "number": 1493,
                    "title": "Cursor pagination for /v1/events",
                    "labels": ["api", "breaking"],
                },
                {"number": 1502, "title": "Scoped API keys", "labels": ["auth", "security"]},
            ],
            "diff_stat": "27 files changed, 5,880 insertions(+), 1,945 deletions(-)",
            "breaking_changes_detected_by_ci": [
                "GET /v1/events now returns cursor/next_cursor instead of page/total_pages",
                "webhook payloads now include idempotency_key (schema is strict in two "
                "downstream consumers)",
            ],
        }
    )


# --------------------------------------------------------------------------- #
# Tool payload 6 — slow query log + EXPLAIN ANALYZE
# --------------------------------------------------------------------------- #
def payload_slow_queries() -> str:
    return blob(
        {
            "source": "pg_stat_statements",
            "database": "app_prod",
            "window": {"from": "2026-10-28T00:00:00Z", "to": "2026-10-29T00:00:00Z"},
            "instance": "db-primary-3 (db.r6g.2xlarge, 8 vCPU, 64 GiB)",
            "top_queries": [
                {
                    "queryid": 8842190311,
                    "calls": 412873,
                    "mean_ms": 118.4,
                    "total_ms": 48881240.0,
                    "rows_per_call": 842.1,
                    "query": "SELECT e.id, e.account_id, e.type, e.payload, e.created_at FROM "
                    "events e JOIN accounts a ON a.id = e.account_id WHERE a.plan = $1 AND "
                    "e.created_at >= $2 AND e.created_at < $3 ORDER BY e.created_at DESC "
                    "LIMIT 100",
                    "indexes_used": ["events_account_id_created_at_idx"],
                },
                {
                    "queryid": 7712034455,
                    "calls": 88124,
                    "mean_ms": 214.7,
                    "total_ms": 18923440.0,
                    "rows_per_call": 1.0,
                    "query": "SELECT count(*) FROM events e WHERE date_trunc('day', "
                    "e.created_at) = $1 AND e.account_id = $2",
                    "indexes_used": [],
                },
                {
                    "queryid": 6620941177,
                    "calls": 51002,
                    "mean_ms": 402.9,
                    "total_ms": 20546720.0,
                    "rows_per_call": 20.0,
                    "query": "SELECT DISTINCT ON (e.account_id) e.account_id, e.created_at "
                    "FROM events e WHERE e.type = $1 ORDER BY e.account_id, "
                    "e.created_at DESC",
                    "indexes_used": ["events_type_idx"],
                },
                {
                    "queryid": 5593102288,
                    "calls": 30291,
                    "mean_ms": 88.1,
                    "total_ms": 2668640.0,
                    "rows_per_call": 118.4,
                    "query": "SELECT a.id, a.name, (SELECT count(*) FROM events e WHERE "
                    "e.account_id = a.id AND e.created_at > now() - interval '30 days') AS "
                    "event_count FROM accounts a WHERE a.plan = $1 ORDER BY event_count DESC "
                    "LIMIT 50",
                    "indexes_used": ["accounts_plan_idx"],
                },
                {
                    "queryid": 3381770022,
                    "calls": 18420,
                    "mean_ms": 155.3,
                    "total_ms": 2860900.0,
                    "rows_per_call": 500.0,
                    "query": "SELECT * FROM events WHERE payload->>'source' = $1 AND "
                    "created_at > $2",
                    "indexes_used": [],
                },
            ],
            "explain_analyze": [
                {
                    "queryid": 8842190311,
                    "plan": {
                        "Node Type": "Limit",
                        "Actual Rows": 100,
                        "Actual Total Time": 118.412,
                        "Plans": [
                            {
                                "Node Type": "Nested Loop",
                                "Join Type": "Inner",
                                "Actual Rows": 100,
                                "Actual Total Time": 118.401,
                                "Plans": [
                                    {
                                        "Node Type": "Index Scan",
                                        "Relation Name": "accounts",
                                        "Index Name": "accounts_plan_idx",
                                        "Actual Rows": 412,
                                        "Actual Total Time": 1.204,
                                    },
                                    {
                                        "Node Type": "Index Scan",
                                        "Relation Name": "events",
                                        "Index Name": "events_account_id_created_at_idx",
                                        "Actual Rows": 24,
                                        "Actual Total Time": 0.281,
                                        "Rows Removed by Filter": 1180,
                                    },
                                ],
                            }
                        ],
                    },
                    "note": "24 rows scanned per account, 1180 filtered; sort avoided by index",
                },
                {
                    "queryid": 7712034455,
                    "plan": {
                        "Node Type": "Aggregate",
                        "Actual Rows": 1,
                        "Actual Total Time": 214.702,
                        "Plans": [
                            {
                                "Node Type": "Seq Scan",
                                "Relation Name": "events",
                                "Actual Rows": 9412,
                                "Actual Total Time": 210.115,
                                "Filter": "(date_trunc('day'::text, created_at) = "
                                "'2026-10-27'::timestamp)",
                                "Rows Removed by Filter": 8211540,
                            }
                        ],
                    },
                    "note": "function on the indexed column defeats the index entirely",
                },
            ],
            "table_stats": [
                {
                    "table": "events",
                    "rows": 8220952,
                    "size": "14 GB",
                    "dead_tuples": 812004,
                    "last_autovacuum": "2026-10-27T04:12:00Z",
                },
                {"table": "accounts", "rows": 41882, "size": "62 MB", "dead_tuples": 1204},
            ],
            "existing_indexes": [
                {"name": "events_pkey", "definition": "PRIMARY KEY (id)"},
                {
                    "name": "events_account_id_created_at_idx",
                    "definition": "btree (account_id, created_at DESC)",
                },
                {"name": "events_type_idx", "definition": "btree (type)"},
                {"name": "events_created_at_idx", "definition": "btree (created_at)"},
            ],
            "notes": [
                "payload is jsonb but the only jsonb index is on (payload->>'kind')",
                "the 30-day window subquery runs once per returned account row",
                "work_mem is 4MB; two queries spilled to disk",
            ],
        }
    )


# --------------------------------------------------------------------------- #
# The 50 tasks.
#
# Distribution (documented in tasks/README.md, asserted by the validator):
#   codegen 10 | debug 8 | refactor 6 | test-gen 6 | sql 5 | explain 5
#   tool-use 6 | reasoning 4
#
# Conventions used below:
#   * ``must_include`` holds substrings that any *correct* answer contains, so
#     containment grading is meaningful rather than cosmetic.
#   * ``reference`` is set only where the whole expected answer is short enough
#     for exact-match grading (the grader switches tiers on a <=64-char
#     reference), which is why exactly three tasks carry one.
#   * ``tool_payload`` is present on all six tool-use tasks and is the blob the
#     ``compress_tool_output`` strategy ablates.
# --------------------------------------------------------------------------- #
def T(
    task_id: str,
    category: str,
    prompt: str,
    *,
    must_include: list[str] | None = None,
    reference: str | None = None,
    tool_payload: str | None = None,
    max_output_tokens: int = 1024,
    difficulty: float = 0.5,
    language: str = "python",
    tags: list[str] | None = None,
    ambiguous: bool = False,
) -> dict[str, Any]:
    """Build one task dict with consistent metadata."""
    metadata: dict[str, Any] = {
        "language": language,
        "tags": tags or [],
        "expected_output_tokens": max_output_tokens,
    }
    if ambiguous:
        metadata["ambiguity"] = (
            "underspecified on purpose: a good answer states assumptions and asks "
            "the missing question before proposing a solution"
        )
    return {
        "id": task_id,
        "category": category,
        "prompt": prompt,
        "reference": reference,
        "must_include": must_include or [],
        "tool_payload": tool_payload,
        "max_output_tokens": max_output_tokens,
        "difficulty": difficulty,
        "metadata": metadata,
    }


CI_RUN = payload_ci_run()
TS_BUILD = payload_ts_build()
K8S = payload_k8s_crashloop()
API_PAGES = payload_api_pages()
GIT_HISTORY = payload_git_history()
SLOW_QUERIES = payload_slow_queries()

TASKS: list[dict[str, Any]] = [
    # ------------------------------------------------------------- codegen 10
    T(
        "cb-001",
        "codegen",
        "Our internal HTTP client uses httpx synchronously and has no retry logic, so every "
        "transient 502 from the payments gateway surfaces as a user-visible error. Write a "
        "`retry` decorator that retries only the exception types passed in, uses exponential "
        "backoff with full jitter, caps total elapsed wall time, and re-raises the last "
        "exception with the attempt count attached. It must not retry KeyboardInterrupt and "
        "must be safe when the wrapped call runs from several threads. Python 3.12, stdlib only.",
        must_include=["def retry", "except", "raise", "sleep"],
        max_output_tokens=1400,
        difficulty=0.4,
        tags=["http", "resilience", "decorator"],
    ),
    T(
        "cb-002",
        "codegen",
        "We are dropping lodash from the analytics tracker. Implement `debounce` in strict "
        "TypeScript with no dependencies and the surface we actually use: `cancel()`, "
        "`flush()`, an optional leading-edge call, and correct `this` typing on the returned "
        "function. The tracker fires roughly 400 events a minute during scroll and currently "
        "loses the final event when the tab closes, so say how your design fixes that.",
        must_include=["function debounce", "setTimeout", "cancel", "flush"],
        max_output_tokens=1200,
        difficulty=0.45,
        language="typescript",
        tags=["frontend", "timing", "types"],
    ),
    T(
        "cb-003",
        "codegen",
        "Write a bounded worker pool for a Go 1.23 ingest service. Callers submit "
        "`func(context.Context, Job) error`; the pool runs N workers, stops accepting new jobs "
        "when the context is cancelled, and returns every error rather than only the first. "
        "Shutdown must drain in-flight jobs and must not leak goroutines if the caller abandons "
        "the context. Show the struct, the constructor and the submit path.",
        must_include=["func ", "context.Context", "chan ", "WaitGroup"],
        max_output_tokens=1400,
        difficulty=0.6,
        language="go",
        tags=["concurrency", "worker-pool"],
    ),
    T(
        "cb-004",
        "codegen",
        "Our Rust CLI reads a TOML config with optional fields and panics on the first bad "
        "value, which is unacceptable for a tool that runs in CI. Rework it into "
        "`Config::load(path) -> Result<Config, ConfigError>` with serde and a custom error enum "
        "distinguishing missing file, parse failure and validation failure (unknown log level, "
        "timeout <= 0). Use thiserror, no unwrap or expect outside tests. Show structs, error "
        "enum and impl.",
        must_include=["fn ", "Result", "impl", "thiserror"],
        max_output_tokens=1400,
        difficulty=0.55,
        language="rust",
        tags=["config", "errors", "serde"],
    ),
    T(
        "cb-005",
        "codegen",
        "Write a bash script that rotates /var/log/app/*.log: gzip anything older than a day, "
        "delete gzipped archives older than 14 days, then signal the app with HUP so it reopens "
        "its file handles. It runs from cron as root, must fail loudly instead of half-rotating, "
        "must be safe to run twice concurrently, and must not follow symlinks out of the log "
        "directory. Explain the locking choice in one sentence at the end.",
        must_include=["set -euo pipefail", "find", "gzip", "flock"],
        max_output_tokens=1200,
        difficulty=0.5,
        language="shell",
        tags=["ops", "cron", "files"],
    ),
    T(
        "cb-006",
        "codegen",
        "Add a streaming upload helper to our Python SDK: read a local file in chunks, PUT it to "
        "a presigned URL with httpx, and report progress through a callback. Requirements: "
        "constant memory regardless of file size, a per-chunk timeout, retry a chunk twice on "
        "transport errors by re-seeking the file, and close the file on every path. Take a Path, "
        "a URL and a Callable[[int, int], None].",
        must_include=["def ", "yield", "httpx", "finally"],
        max_output_tokens=1300,
        difficulty=0.5,
        tags=["streaming", "http", "sdk"],
    ),
    T(
        "cb-007",
        "codegen",
        "Build a strongly typed event emitter for our browser bus. An `Events` interface maps "
        "event names to payload types, `on` returns an unsubscribe function, `once` removes "
        "itself after firing, and a handler that throws must not stop the others (collect the "
        "errors and surface them to the emitter's error handler). No Node EventEmitter. The "
        "generic signature matters: a wrong payload type must be a compile error.",
        must_include=["interface", "Map", "on(", "once"],
        max_output_tokens=1300,
        difficulty=0.55,
        language="typescript",
        tags=["types", "events", "browser"],
    ),
    T(
        "cb-008",
        "codegen",
        "We call a partner API capped at 50 requests per second with bursts up to 120. Implement "
        "a thread-safe token bucket with `acquire(n=1, timeout=None)` that blocks instead of "
        "raising, refills monotonically, and does not drift under sustained load. Include a note "
        "on why `time.monotonic()` rather than `time.time()`, and what happens to waiters when "
        "the timeout expires.",
        must_include=["def ", "Lock", "monotonic", "timeout"],
        max_output_tokens=1100,
        difficulty=0.45,
        tags=["concurrency", "rate-limit"],
    ),
    T(
        "cb-009",
        "codegen",
        "Add a net/http middleware to our Go service that generates a request id when the "
        "incoming X-Request-Id header is missing, stores it on the request context under a "
        "private key type, logs method, path, status and duration as one structured JSON line, "
        "and recovers panics into a 500 without losing that log line. It must wrap any "
        "http.Handler and leave successful response bodies untouched.",
        must_include=["func ", "http.Handler", "context.", "recover()"],
        max_output_tokens=1200,
        difficulty=0.5,
        language="go",
        tags=["http", "middleware", "observability"],
    ),
    T(
        "cb-010",
        "codegen",
        "Our ingest binary drops messages on Ctrl-C because tokio just aborts the task. Write "
        "the main loop with `tokio::select!` over a bounded mpsc receiver and a shutdown signal: "
        "stop accepting new work, finish the in-flight batch, flush once, exit 0 on graceful "
        "shutdown and non-zero if the channel closed unexpectedly. Show the channel types and "
        "say where backpressure appears.",
        must_include=["tokio::select!", "mpsc", "await", "async fn"],
        max_output_tokens=1300,
        difficulty=0.65,
        language="rust",
        tags=["tokio", "shutdown", "streaming"],
    ),
    # --------------------------------------------------------------- debug 8
    T(
        "cb-011",
        "debug",
        "Production shows intermittent asyncpg TimeoutErrors under load, always after about "
        "twenty minutes of traffic, and the container memory graph climbs alongside them. The "
        "only suspicious code is a helper that does `conn = await pool.acquire()` then "
        "`await conn.fetch(...)` inside a try with a bare `except Exception: log(...)` and no "
        "finally. Explain what leaks and why it appears only later, then show the corrected "
        "helper. Python 3.12, asyncpg.",
        must_include=["async with", "finally", "release", "pool"],
        max_output_tokens=1100,
        difficulty=0.55,
        tags=["asyncio", "connection-pool", "leak"],
    ),
    T(
        "cb-012",
        "debug",
        "A dashboard component reads `filters` inside a setInterval callback registered once on "
        "mount. Changing a filter in the UI does nothing until a full page reload, and the code "
        "looks correct to its author. Explain the stale-closure bug precisely, then give two "
        "idiomatic fixes (one with a ref, one by re-registering the effect) and say which you "
        "would ship. React 18, function component, no state library.",
        must_include=["useEffect", "useRef", "dependency", "setInterval"],
        max_output_tokens=1100,
        difficulty=0.4,
        language="typescript",
        tags=["react", "hooks", "closures"],
    ),
    T(
        "cb-013",
        "debug",
        "Our metrics registry is a plain map[string]int64 written from request handlers. "
        "`go test -race` fails only with `-count=20`, and production has logged one `fatal error: "
        "concurrent map writes` plus silent lost increments. Give the corrected registry: the "
        "mutex type, the write path, and a Snapshot method that does not hold the lock while "
        "copying. Also say why the race detector only fires under repetition.",
        must_include=["sync.RWMutex", "Snapshot", "Lock"],
        max_output_tokens=1100,
        difficulty=0.45,
        language="go",
        tags=["concurrency", "race", "metrics"],
    ),
    T(
        "cb-014",
        "debug",
        "This function does not compile: E0499, two mutable borrows of self. The loop calls "
        "`self.entries.get_mut(i)` and then `self.normalize(entry)` in the same iteration. "
        "Explain why the borrow checker rejects it, then show two fixes — one that restructures "
        "the loop, one that changes the method signature — and say which preserves the invariant "
        "that normalize may mutate neighbouring entries. Rust 2021.",
        must_include=["borrow", "&mut", "self"],
        max_output_tokens=1200,
        difficulty=0.65,
        language="rust",
        tags=["borrow-checker", "compile-error"],
    ),
    T(
        "cb-015",
        "debug",
        "A long-lived worker's RSS grows about 40 MB per hour and never comes down. It memoizes a "
        "nearly pure function with `@lru_cache()`, and the argument is a dataclass built from "
        "request payloads. tracemalloc points at the cache dict as the top allocation site. "
        "Explain the mechanism, give the minimal fix, and name the correctness risk of the fix "
        "you chose.",
        must_include=["lru_cache", "maxsize", "cache", "memory"],
        max_output_tokens=900,
        difficulty=0.5,
        tags=["memory", "caching", "dataclass"],
    ),
    T(
        "cb-016",
        "debug",
        "A deploy script runs fine by hand as ./deploy.sh but from cron it exits 127 with "
        "`node: command not found`. Its first line is `#!/bin/sh`, and it calls node, pnpm and jq "
        "by bare name. Team convention is that these scripts must run under bash and inherit a "
        "login-style PATH from the shebang. Answer with only the single shebang line that fixes "
        "this class of failure: no explanation, no code fence.",
        reference="#!/usr/bin/env bash",
        max_output_tokens=48,
        difficulty=0.2,
        language="shell",
        tags=["cron", "shebang", "environment"],
    ),
    T(
        "cb-017",
        "debug",
        "Two workers deadlock a few times an hour updating the same rows: one updates orders then "
        "order_items, the other order_items then orders, both in a single transaction using "
        "SELECT ... FOR UPDATE. Postgres logs `deadlock detected` and kills one. Give the root "
        "cause in one sentence, then the two fixes we can ship independently (lock ordering and a "
        "retry policy), including what the retry must not retry.",
        must_include=["deadlock", "ORDER BY", "retry", "transaction"],
        max_output_tokens=900,
        difficulty=0.55,
        language="sql",
        tags=["postgres", "locking", "retry"],
    ),
    T(
        "cb-018",
        "debug",
        "Our Express handler calls sendEmail(user) without awaiting it and returns 202 straight "
        "away. Sentry shows unhandled rejections that crash the process about once a day, and the "
        "test suite intermittently reports 'Jest did not exit'. Explain what actually crashes and "
        "when, then show the corrected handler with an explicit log-and-continue policy for email "
        "failures plus how to make the tests deterministic.",
        must_include=["await", "catch", "unhandled"],
        max_output_tokens=1100,
        difficulty=0.45,
        language="typescript",
        tags=["node", "promises", "testing"],
    ),
    # ------------------------------------------------------------ refactor 6
    T(
        "cb-019",
        "refactor",
        "pricing.py is a 600-line function with a chain of `if plan == ...` branches that also "
        "does tax lookup and discount stacking. Break it into composable pieces: one strategy per "
        "plan, a discount pipeline, and a pure `quote(cart, customer) -> Quote` entry point. The "
        "public signature must stay stable because three services import it, and show the test "
        "seam that lets tests fake tax.",
        must_include=["class ", "def ", "return", "Quote"],
        max_output_tokens=1600,
        difficulty=0.6,
        tags=["architecture", "pricing", "testing"],
    ),
    T(
        "cb-020",
        "refactor",
        "A 40-line nested ternary in our form renderer maps field type to component and is now "
        "unmaintainable. Replace it with a typed lookup object, keep exhaustiveness so adding a "
        "new field type becomes a compile error, and preserve today's behaviour for unknown types "
        "(render a read-only fallback). Show it as a diff-sized before/after, not the whole file.",
        must_include=["Record", "const", "return"],
        max_output_tokens=900,
        difficulty=0.35,
        language="typescript",
        tags=["readability", "types", "forms"],
    ),
    T(
        "cb-021",
        "refactor",
        "We have four near-identical Map helpers written against interface{} with type "
        "assertions, all pre-generics. Collapse them into a generic core "
        "`Map[T, U any](in []T, f func(T) U) []U`, add Filter and Reduce only if the call sites "
        "actually use them, and delete the rest. Go 1.23; allocation behaviour must be no worse "
        "than today, and say how you checked that.",
        must_include=["func Map", "any", "[T "],
        max_output_tokens=1100,
        difficulty=0.5,
        language="go",
        tags=["generics", "cleanup", "allocations"],
    ),
    T(
        "cb-022",
        "refactor",
        "This module has 23 unwrap()/expect() calls in request handling and a malformed header "
        "currently panics a worker thread. Convert it to propagate errors with ?, introduce one "
        "error type for the handler, and map it to an HTTP status in exactly one place. Do not "
        "change happy-path behaviour. Show the error enum, the From impls you need and one "
        "converted function.",
        must_include=["Result", "map_err", "impl From", "thiserror"],
        max_output_tokens=1300,
        difficulty=0.55,
        language="rust",
        tags=["errors", "http", "panic"],
    ),
    T(
        "cb-023",
        "refactor",
        "This query runs per-account subqueries in the SELECT list and takes 4s across 40k "
        "accounts. Rewrite it as a join against a pre-aggregated derived table, or with a window "
        "function if that reads better, keeping the same columns and the same ordering by "
        "event_count DESC. State the plan you expect and whether results can change for accounts "
        "with zero events.",
        must_include=["JOIN", "GROUP BY", "ORDER BY"],
        max_output_tokens=1000,
        difficulty=0.6,
        language="sql",
        tags=["postgres", "query-rewrite", "performance"],
    ),
    T(
        "cb-024",
        "refactor",
        "A batch job makes 500 sequential requests.get calls at about 2s each and we need it under "
        "30s. Convert the module to httpx.AsyncClient with bounded concurrency (semaphore of 16), "
        "a per-request timeout, and a gather that tolerates individual failures instead of failing "
        "the batch. Keep the return shape identical and show how a sync caller should invoke it.",
        must_include=["async def", "await", "gather", "Semaphore"],
        max_output_tokens=1300,
        difficulty=0.5,
        tags=["asyncio", "http", "performance"],
    ),
    # ------------------------------------------------------------ test-gen 6
    T(
        "cb-025",
        "test-gen",
        "Write the pytest suite for `parse_range(spec) -> tuple[date, date]`, which accepts "
        "`2026-01-01..2026-01-31`, `last:7d` and `today`. Cover inclusive endpoints, month-end "
        "arithmetic, invalid specs raising ValueError, and an empty range. Use parametrize with "
        "readable ids. No freeze libraries: inject today or monkeypatch the module clock.",
        must_include=["@pytest.mark.parametrize", "def test_", "pytest.raises", "ids="],
        max_output_tokens=1400,
        difficulty=0.4,
        tags=["pytest", "dates", "parametrize"],
    ),
    T(
        "cb-026",
        "test-gen",
        "Write vitest tests for `cartTotal(items, coupon)` in a checkout module. It applies "
        "percentage and fixed-amount coupons, rounds half-up to cents, caps the discount at the "
        "subtotal, and returns 0 for an empty cart. Cover the cap, the 0.005 rounding edge, "
        "stacking order, and a coupon in a different currency (should throw). Use describe/it/"
        "expect and test.each for the rounding table.",
        must_include=["describe(", "it(", "expect(", "test.each"],
        max_output_tokens=1300,
        difficulty=0.45,
        language="typescript",
        tags=["vitest", "money", "rounding"],
    ),
    T(
        "cb-027",
        "test-gen",
        "Write table-driven Go tests for a SplitHostPort helper: it must reject missing brackets "
        "on IPv6 literals, empty ports and ports above 65535, and accept example.com:443, "
        "[::1]:8080 and an empty host with a port. Use t.Run subtests, compare a wantErr string "
        "rather than a bool so failures are diagnosable, and add one fuzz target seeded from the "
        "table.",
        must_include=["func Test", "t.Run", "want", "testing.F"],
        max_output_tokens=1300,
        difficulty=0.5,
        language="go",
        tags=["testing", "fuzz", "parsing"],
    ),
    T(
        "cb-028",
        "test-gen",
        "Write the test module for `parse_duration(&str) -> Result<Duration, ParseError>`, which "
        "accepts 1h30m, 250ms and 2d, and rejects 1h30, -5s, 0s and 1y. Include unit tests, one "
        "property-style loop over generated inputs (no proptest: a deterministic xorshift "
        "generator is fine), and a test asserting the error carries the offending byte span.",
        must_include=["#[test]", "assert_eq!", "Result", "mod tests"],
        max_output_tokens=1400,
        difficulty=0.55,
        language="rust",
        tags=["testing", "parsing", "errors"],
    ),
    T(
        "cb-029",
        "test-gen",
        "Write pytest tests for `retry(attempts=3, base=0.05, on=(TimeoutError,))`: it succeeds "
        "after two failures, gives up after attempts with the last exception, does not retry an "
        "unlisted exception, and sleeps with exponential backoff. Monkeypatch the module's sleep "
        "and clock so the suite is fast and deterministic, and assert the exact sequence of "
        "delays.",
        must_include=["def test_", "monkeypatch", "pytest.raises", "call_count"],
        max_output_tokens=1200,
        difficulty=0.5,
        tags=["pytest", "retry", "monkeypatch"],
    ),
    T(
        "cb-030",
        "test-gen",
        "Write a bats-core suite for bin/tokbench-run, a shell wrapper that validates a YAML "
        "path, refuses to start without --yes when the budget exceeds $1, and prints a summary "
        "line. Cover: missing argument exits 2, malformed YAML exits 3, the budget guard requires "
        "--yes, and a happy path with a stubbed binary on PATH. Keep fixtures in test/fixtures "
        "and clean up temp dirs.",
        must_include=["@test", "run ", "status", "teardown"],
        max_output_tokens=1200,
        difficulty=0.55,
        language="shell",
        tags=["bats", "cli", "testing"],
    ),
    # ---------------------------------------------------------------- sql 5
    T(
        "cb-031",
        "sql",
        "Write one Postgres query returning, per day and per plan, the number of accounts active "
        "that day that were also active in the previous 7 days (rolling weekly retention). Tables: "
        "events(account_id, created_at) and accounts(id, plan). Use a window function rather than "
        "seven self-joins, keep the predicate sargable on created_at, and name the index needed to "
        "stay under a second on 8M events.",
        must_include=["OVER", "PARTITION BY", "JOIN", "CREATE INDEX"],
        max_output_tokens=1300,
        difficulty=0.65,
        language="sql",
        tags=["postgres", "window-function", "retention"],
    ),
    T(
        "cb-032",
        "sql",
        "The events table picked up about 2M duplicate rows from a bad replay: same "
        "(account_id, idempotency_key, created_at), different payloads. Write the dedupe as a "
        "transaction that keeps the row with the smallest ctid per group, then adds the unique "
        "constraint that prevents a repeat. Show the DELETE, the constraint, and how you would "
        "verify counts before and after without locking the table for minutes.",
        must_include=["row_number", "DELETE", "UNIQUE", "ctid"],
        max_output_tokens=1200,
        difficulty=0.7,
        language="sql",
        tags=["postgres", "dedupe", "migration"],
    ),
    T(
        "cb-033",
        "sql",
        "Build the org-chart query: given employees(id, manager_id, name, level), return each "
        "employee with their full management chain as an array, the depth, and a flag for cycles. "
        "Postgres 16. Use a recursive CTE, cap recursion at depth 10, and make sure a cycle in the "
        "data cannot produce an infinite loop. Show the CTE and the cycle guard.",
        must_include=["WITH RECURSIVE", "UNION ALL", "ARRAY", "depth"],
        max_output_tokens=1200,
        difficulty=0.6,
        language="sql",
        tags=["postgres", "recursive-cte", "hierarchy"],
    ),
    T(
        "cb-034",
        "sql",
        "One query in the slow log filters `WHERE date_trunc('day', created_at) = $1 AND "
        "account_id = $2` on an 8M-row events table and does a sequential scan removing 8.2M rows. "
        "Rewriting the query is out of scope this sprint; the DBA will only accept an index, and "
        "account_id is already covered by an existing btree. Answer with only the single CREATE "
        "INDEX statement that makes the remaining predicate indexable, on one line.",
        reference="CREATE INDEX ON events (date_trunc('day', created_at))",
        max_output_tokens=96,
        difficulty=0.55,
        language="sql",
        tags=["postgres", "index", "expression-index"],
    ),
    T(
        "cb-035",
        "sql",
        "Our billing events carry a jsonb payload with keys like plan, seats and region. Write one "
        "query that returns, per region and month, the event count, the sum of seats (numeric, "
        "missing counts as 0) and the share of events whose plan is 'pro' — as columns, not rows. "
        "Use jsonb operators, GROUP BY and a FILTER clause, and note the GIN index that would "
        "help if the table grows tenfold.",
        must_include=["jsonb", "GROUP BY", "FILTER", "GIN"],
        max_output_tokens=1200,
        difficulty=0.6,
        language="sql",
        tags=["postgres", "jsonb", "aggregation"],
    ),
    # ------------------------------------------------------------ explain 5
    T(
        "cb-036",
        "explain",
        "We are choosing between a single-region Postgres primary and a multi-region "
        "active-active store for a payments ledger. Walk through what each option actually gives "
        "up under a regional partition, in terms of consistency, availability and reconciliation "
        "cost. Be concrete about the failure sequence for a double-spend attempt under each "
        "design, and end with a recommendation for a 40-person fintech doing $2M a day.",
        must_include=["consistency", "availability", "partition", "idempot"],
        max_output_tokens=1400,
        difficulty=0.75,
        language="architecture",
        tags=["cap", "payments", "trade-offs"],
    ),
    T(
        "cb-037",
        "explain",
        "After yesterday's deploy p99 latency went from 180ms to 2.4s while p50 stayed at 90ms "
        "and the error rate is flat. The deploy changed exactly one thing: DB pool max size from "
        "40 to 10. The service runs 8 pods behind an ALB and each request makes 3-6 queries. "
        "Explain the mechanism that produces this exact signature, and name the two metrics you "
        "would look at first to confirm it.",
        must_include=["pool", "queue", "p99", "saturat"],
        max_output_tokens=1000,
        difficulty=0.65,
        language="architecture",
        tags=["latency", "percentiles", "diagnosis"],
    ),
    T(
        "cb-038",
        "explain",
        "Explain to a backend team why prompt caching and request batching attack different parts "
        "of the cost curve. Cover what each does to input tokens, output tokens, latency and "
        "cache-hit economics, and the workload shapes where each is worthless. Use a worked "
        "example: a 2,000-token system prompt served 10,000 times a day.",
        must_include=["cache", "batch", "latency", "input token"],
        max_output_tokens=1200,
        difficulty=0.55,
        language="architecture",
        tags=["caching", "batching", "llm-cost"],
    ),
    T(
        "cb-039",
        "explain",
        "Explain Rust ownership, borrowing and lifetimes to a team that writes TypeScript. Use one "
        "running example — mutating an array while iterating it — and show what TS permits at "
        "runtime, what Rust rejects at compile time, and why the Rust error is a bug TypeScript "
        "would have shipped. Keep it under 400 words and do not use the phrase 'memory safety' "
        "without defining it.",
        must_include=["ownership", "borrow", "lifetime", "iterator"],
        max_output_tokens=1000,
        difficulty=0.5,
        language="rust",
        tags=["teaching", "ownership"],
    ),
    T(
        "cb-040",
        "explain",
        "We publish a Python SDK at 2.14.0 and need a deprecation policy clients can trust. "
        "Explain how semver maps onto Python realities (keyword arguments, return types, exception "
        "types, supported interpreter versions), which changes people wrongly assume are "
        "non-breaking, and how to run a deprecation window. Finish with the exact policy text you "
        "would put in CONTRIBUTING.md.",
        must_include=["major", "minor", "patch", "deprecat"],
        max_output_tokens=1300,
        difficulty=0.6,
        language="architecture",
        tags=["semver", "api-design", "policy"],
    ),
    # ----------------------------------------------------------- tool-use 6
    T(
        "cb-041",
        "tool-use",
        "Our CI job for the order-events outbox branch is red. The attached payload is the failing "
        "GitHub Actions run: pytest failures, captured logs, the full git diff under test, and the "
        "test summary. Identify the single root cause that explains both the idempotency failure "
        "and the 500 on order creation, give the minimal fix as a diff-sized patch, and name the "
        "regression test that must be added. Do not restate the payload back to me.",
        must_include=["dedupe_key", "UniqueViolation", "transaction"],
        tool_payload=CI_RUN,
        max_output_tokens=1600,
        difficulty=0.75,
        tags=["ci", "debugging", "postgres", "outbox"],
    ),
    T(
        "cb-042",
        "tool-use",
        "`pnpm -r --filter @acme/console build` fails with six TypeScript errors after the v0.14 "
        "workspace reshuffle. The attached payload has the workspace listing, package manifests, "
        "tsconfigs, the pnpm resolution report and the tsc output. Separate the errors that share "
        "one root cause from the noise, then give the ordered set of file edits that gets the "
        "build green without deleting type checks.",
        must_include=["references", "@acme/tokens", "exports"],
        tool_payload=TS_BUILD,
        max_output_tokens=1600,
        difficulty=0.8,
        language="typescript",
        tags=["monorepo", "typescript", "build"],
    ),
    T(
        "cb-043",
        "tool-use",
        "Two of three ledger-writer pods are in CrashLoopBackOff after a rolling update; the third "
        "pod, still on v3.6.4, is healthy. The payload contains pod status, events, previous "
        "container logs, the configmap diff and the deployment spec. Give the root cause with the "
        "one evidence line that proves it, the immediate rollback-or-fix call, and the guard that "
        "should have caught this in CI.",
        must_include=["connection refused", "readiness", "MIGRATE_ON_START"],
        tool_payload=K8S,
        max_output_tokens=1400,
        difficulty=0.7,
        language="kubernetes",
        tags=["kubernetes", "crashloop", "incident"],
    ),
    T(
        "cb-044",
        "tool-use",
        "We are importing about 50 customers from a partner's paginated REST API. The attached "
        "capture shows four real requests, the OpenAPI fragment for the list endpoint, and notes "
        "about quirks we already hit. Write the importer loop in Python (sync httpx is fine) that "
        "pages correctly, handles the 429 with Retry-After, cannot emit duplicate customers, and "
        "is resumable from a stored cursor.",
        must_include=["starting_after", "Retry-After", "next_cursor", "429"],
        tool_payload=API_PAGES,
        max_output_tokens=1500,
        difficulty=0.7,
        tags=["api", "pagination", "etl"],
    ),
    T(
        "cb-045",
        "tool-use",
        "Turn the attached git history into release notes for v2.15.0. The payload has the commit "
        "range, authors, labels, merged pull requests and the breaking changes CI flagged. Produce "
        "markdown with a two-sentence summary, grouped sections (features, fixes, performance, "
        "internal), a clearly separated breaking-changes section with migration steps, and a "
        "contributor list. Do not invent a commit that is not in the payload.",
        must_include=["Breaking", "pagination", "idempotency_key"],
        tool_payload=GIT_HISTORY,
        max_output_tokens=1500,
        difficulty=0.6,
        language="markdown",
        tags=["release-notes", "git", "writing"],
    ),
    T(
        "cb-046",
        "tool-use",
        "The attached pg_stat_statements snapshot, EXPLAIN plans and table stats come from our "
        "production events database. Rank the top three cost drivers by total time, then pick the "
        "single change with the best time-saved-per-risk ratio for this week. The payload lists "
        "the existing indexes. Show the exact SQL for your recommendation and say what you would "
        "measure afterwards to prove it worked.",
        must_include=["date_trunc", "jsonb", "CREATE INDEX"],
        tool_payload=SLOW_QUERIES,
        max_output_tokens=1500,
        difficulty=0.8,
        language="sql",
        tags=["postgres", "performance", "triage"],
    ),
    # ---------------------------------------------------------- reasoning 4
    T(
        "cb-047",
        "reasoning",
        "Our head of product says checkout feels slow and that we should make it faster this "
        "sprint. Nothing else is specified: no target, no traffic data, no budget for new "
        "infrastructure. Write the plan you would actually execute, starting with what you must "
        "measure and what you must ask before touching code. Include the specific numbers you "
        "would need to call it done, and what you would refuse to do in a one-sprint window.",
        must_include=["p50", "p99", "baseline", "measure"],
        max_output_tokens=1300,
        difficulty=0.9,
        language="architecture",
        tags=["ambiguity", "planning", "performance"],
        ambiguous=True,
    ),
    T(
        "cb-048",
        "reasoning",
        "The events table is 14GB and grows 400MB a day. Writes are fine; reads are the problem. A "
        "staff engineer proposes sharding by account_id across 8 Postgres instances. Argue for "
        "and against with numbers (storage per shard, cross-shard queries, operational cost), then "
        "make a call. If you would do something cheaper first, say exactly what and how you would "
        "know when to revisit.",
        must_include=["shard", "index", "cross-shard"],
        max_output_tokens=1400,
        difficulty=0.85,
        language="architecture",
        tags=["scaling", "trade-offs", "postgres"],
        ambiguous=True,
    ),
    T(
        "cb-049",
        "reasoning",
        "Cache reads are billed at 10% of the input price. A workload sends a 2,000-token prompt "
        "10,000 times a day and on average 60% of the prompt tokens are served from cache. "
        "Ignoring output tokens, what is the effective input price as a fraction of list price? "
        "Answer with only the decimal, at most four digits.",
        reference="0.46",
        max_output_tokens=32,
        difficulty=0.35,
        language="math",
        tags=["caching", "cost-model", "arithmetic"],
    ),
    T(
        "cb-050",
        "reasoning",
        "We need a global rate limit of 1,000 requests per minute per API key across 3 regions, "
        "with no single point of failure. A per-region limit of 333 is too coarse during traffic "
        "shifts, and a shared Redis counter adds a cross-region hop to every request. Design the "
        "scheme you would ship, including the error budget you accept during a partition and how "
        "you handle clock skew. Name the failure mode you consider acceptable and the one you do "
        "not.",
        must_include=["Redis", "quota", "partition", "clock"],
        max_output_tokens=1400,
        difficulty=0.9,
        language="architecture",
        tags=["distributed-systems", "rate-limit", "ambiguity"],
        ambiguous=True,
    ),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(task, ensure_ascii=False) for task in TASKS]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from collections import Counter

    cats = Counter(t["category"] for t in TASKS)
    payloads = [t for t in TASKS if t.get("tool_payload")]
    chars = [len(t["prompt"]) for t in TASKS]
    print(f"wrote {len(TASKS)} tasks -> {OUT}")
    print(f"categories: {dict(sorted(cats.items()))}")
    print(f"prompts: avg {sum(chars) / len(chars):.0f} chars, min {min(chars)}, max {max(chars)}")
    print(f"tool payloads: {len(payloads)}")
    for t in payloads:
        print(
            f"  {t['id']}: {len(t['tool_payload']):5d} chars ~ "
            f"{len(t['tool_payload']) / 4:.0f} tokens"
        )
    total_chars = sum(len(t["prompt"]) + len(t.get("tool_payload") or "") for t in TASKS)
    print(f"dataset chars (prompt+payload): {total_chars:,} ~ {total_chars / 4:,.0f} tokens")


if __name__ == "__main__":
    main()
