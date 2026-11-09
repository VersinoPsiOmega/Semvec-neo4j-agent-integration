# ABOUTME: Builds a Neo4j Browser guide from docs/demo_queries.md and serves it over HTTP with CORS.
# ABOUTME: Neo4j opens it after login via browser.post_connect_cmd (":play http://localhost:7477/semvec-demo.html").

from __future__ import annotations

import argparse
import html
import http.server
import re
from dataclasses import dataclass
from pathlib import Path

GUIDE_PATH = "/semvec-demo.html"
DEFAULT_SOURCE = Path(__file__).parent.parent / "docs" / "demo_queries.md"

_SECTION = re.compile(r"^## (.+)$")
_TITLE = re.compile(r"^### (.+)$")
_FENCE = re.compile(r"^```(\w*)\s*$")


@dataclass(frozen=True)
class Query:
    section: str
    title: str
    cypher: str


def parse_queries(markdown: str) -> list[Query]:
    """Collect every ```cypher block together with its ### title and ## section."""
    queries: list[Query] = []
    section = title = ""
    block: list[str] | None = None
    block_is_cypher = False
    for line in markdown.splitlines():
        fence = _FENCE.match(line)
        if block is not None:
            if fence:
                if block_is_cypher and title:
                    queries.append(Query(section=section, title=title, cypher="\n".join(block).strip()))
                block = None
            else:
                block.append(line)
            continue
        if fence:
            block, block_is_cypher = [], fence.group(1).lower() == "cypher"
        elif match := _SECTION.match(line):
            section = match.group(1).strip()
        elif match := _TITLE.match(line):
            title = match.group(1).strip()
    return queries


def _slide(heading: str, body: str) -> str:
    return (
        '    <slide class="row-fluid">\n'
        '      <div class="col-sm-12">\n'
        f"        <h3>{heading}</h3>\n"
        f"{body}"
        "      </div>\n"
        "    </slide>\n"
    )


def render_guide(queries: list[Query]) -> str:
    """Render the queries as a Neo4j Browser guide: an intro slide, then one runnable query per slide."""
    intro = (
        "        <p>The demo queries from <code>docs/demo_queries.md</code>, one per step.</p>\n"
        "        <p>Each query has two buttons: one runs it, the other opens it in the editor first.\n"
        "        Table queries read best in the table view, graph queries in the graph view. Seed the\n"
        "        graph and run demo scenarios 1–6 first, otherwise most queries return nothing.</p>\n"
    )
    slides = [_slide("Semvec × Neo4j demo queries", intro)]
    for query in queries:
        body = (
            f"        <p>{html.escape(query.section)}</p>\n"
            f'        <pre class="pre-scrollable code runnable">{html.escape(query.cypher)}</pre>\n'
        )
        slides.append(_slide(html.escape(query.title), body))
    return (
        '<article class="guide">\n'
        '  <carousel class="deck container-fluid">\n'
        f"{''.join(slides)}"
        "  </carousel>\n"
        "</article>\n"
    )


def make_server(host: str, port: int, source: Path) -> http.server.ThreadingHTTPServer:
    """HTTP server answering GUIDE_PATH with the guide rendered from ``source`` on every request."""

    class GuideHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # name fixed by BaseHTTPRequestHandler
            if self.path != GUIDE_PATH:
                self.send_error(404)
                return
            payload = render_guide(parse_queries(source.read_text(encoding="utf-8"))).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            # Neo4j Browser fetches the guide from the web browser, i.e. from another origin.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

    return http.server.ThreadingHTTPServer((host, port), GuideHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Semvec demo queries as a Neo4j Browser guide.")
    parser.add_argument("--host", default="0.0.0.0", help="interface to bind (container default: all)")
    parser.add_argument("--port", type=int, default=7477)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.source)
    print(f"Serving the Neo4j Browser guide at http://{args.host}:{args.port}{GUIDE_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
