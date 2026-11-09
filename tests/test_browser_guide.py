# ABOUTME: Tests for the Neo4j Browser guide built from docs/demo_queries.md and served over HTTP.
# ABOUTME: Pure Python plus a local HTTP server on an ephemeral port; no Neo4j needed.

import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts.browser_guide import GUIDE_PATH, make_server, parse_queries, render_guide

DEMO_QUERIES = Path(__file__).parent.parent / "docs" / "demo_queries.md"

SAMPLE = """# Title

Intro text.

## TABLE Queries

### 1. Node counts — per label

```cypher
MATCH (n)
RETURN labels(n)[0] AS type, count(n) AS count
```

### 2. Small values

```cypher
MATCH (s) WHERE s.score < 0.5 RETURN s
```

```bash
echo not cypher
```

## GRAPH Queries

### 3. Everything

```cypher
MATCH p=()-->() RETURN p LIMIT 300
```
"""


@pytest.fixture(autouse=True)
def clean_test_data():
    """Override the conftest autouse fixture — pure-Python module, no Neo4j."""
    yield


class TestParseQueries:
    def test_reads_title_section_and_cypher(self):
        # Arrange / Act
        queries = parse_queries(SAMPLE)

        # Assert
        assert queries[0].section == "TABLE Queries"
        assert queries[0].title == "1. Node counts — per label"
        assert queries[0].cypher == "MATCH (n)\nRETURN labels(n)[0] AS type, count(n) AS count"

    def test_ignores_non_cypher_code_blocks(self):
        # Arrange / Act
        queries = parse_queries(SAMPLE)

        # Assert
        assert [q.title for q in queries] == ["1. Node counts — per label", "2. Small values", "3. Everything"]

    def test_tracks_the_section_of_each_query(self):
        # Arrange / Act
        queries = parse_queries(SAMPLE)

        # Assert
        assert queries[2].section == "GRAPH Queries"

    def test_reads_all_demo_queries_of_the_repository(self):
        # Arrange
        markdown = DEMO_QUERIES.read_text(encoding="utf-8")

        # Act
        queries = parse_queries(markdown)

        # Assert
        assert len(queries) == 22


class TestRenderGuide:
    def test_one_slide_per_query_plus_an_intro(self):
        # Arrange
        queries = parse_queries(SAMPLE)

        # Act
        html = render_guide(queries)

        # Assert
        assert html.count("<slide") == len(queries) + 1

    def test_queries_are_runnable_code_blocks(self):
        # Arrange
        queries = parse_queries(SAMPLE)

        # Act
        html = render_guide(queries)

        # Assert
        assert html.count('class="pre-scrollable code runnable"') == len(queries)

    def test_cypher_and_titles_are_html_escaped(self):
        # Arrange
        queries = parse_queries(SAMPLE.replace("Small values", "Values <b>small</b>"))

        # Act
        html = render_guide(queries)

        # Assert
        assert "s.score &lt; 0.5" in html
        assert "<b>small</b>" not in html
        assert "Values &lt;b&gt;small&lt;/b&gt;" in html


class TestGuideServer:
    @pytest.fixture
    def server_url(self, tmp_path):
        source = tmp_path / "queries.md"
        source.write_text(SAMPLE, encoding="utf-8")
        server = make_server("127.0.0.1", 0, source)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}", source
        finally:
            server.shutdown()
            server.server_close()

    def test_serves_the_guide_with_a_cors_header(self, server_url):
        # Arrange
        base, _ = server_url

        # Act
        with urllib.request.urlopen(f"{base}{GUIDE_PATH}") as response:
            body = response.read().decode("utf-8")
            cors = response.headers["Access-Control-Allow-Origin"]

        # Assert
        assert cors == "*"
        assert "3. Everything" in body

    def test_renders_the_current_markdown_on_every_request(self, server_url):
        # Arrange
        base, source = server_url
        source.write_text(SAMPLE.replace("3. Everything", "3. Whole graph"), encoding="utf-8")

        # Act
        with urllib.request.urlopen(f"{base}{GUIDE_PATH}") as response:
            body = response.read().decode("utf-8")

        # Assert
        assert "3. Whole graph" in body

    def test_unknown_path_is_not_found(self, server_url):
        # Arrange
        base, _ = server_url

        # Act / Assert
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base}/other.html")
        assert exc.value.code == 404
