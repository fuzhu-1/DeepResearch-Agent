"""Document loader — loads content from PDF, HTML, Markdown, and plain text sources."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from various sources (PDF, HTML, Markdown, plain text).

    Uses PyMuPDF (``fitz``) for PDF extraction when available, falling back to
    ``textract``.  HTML is parsed with BeautifulSoup.  Markdown and plain text
    use built-in file I/O.
    """

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    async def load_pdf(self, path: str) -> str:
        """Extract text from a PDF file.

        Uses ``PyMuPDF`` (``fitz``) if installed, otherwise falls back to
        ``textract``.

        Args:
            path: Filesystem path to the PDF.

        Returns:
            Extracted text content.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If no text could be extracted.
        """
        self._ensure_file_exists(path)
        text = await self._try_fitz(path)
        if text:
            return text
        text = await self._try_textract(path)
        if text:
            return text
        raise ValueError(f"Could not extract any text from PDF: {path}")

    async def _try_fitz(self, path: str) -> Optional[str]:
        """Attempt PDF extraction with PyMuPDF (fitz)."""
        try:
            import fitz  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("PyMuPDF (fitz) not available — skipping fitz path")
            return None

        try:
            doc = fitz.open(path)
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    pages.append(text)
            doc.close()

            if pages:
                logger.info("Extracted %d pages from PDF via fitz: %s", len(pages), path)
                return "\n\n".join(pages)
            return None
        except Exception as exc:
            logger.warning("fitz extraction failed for %s: %s", path, exc)
            return None

    async def _try_textract(self, path: str) -> Optional[str]:
        """Fallback PDF extraction with textract."""
        try:
            import textract  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("textract not available — skipping textract fallback")
            return None

        try:
            text = textract.process(path).decode("utf-8")
            if text.strip():
                logger.info("Extracted text from PDF via textract: %s", path)
                return text
            return None
        except Exception as exc:
            logger.warning("textract extraction failed for %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    async def load_html(self, url_or_path: str) -> str:
        """Load and extract readable text from an HTML source.

        If *url_or_path* starts with ``http://`` or ``https://`` it is fetched
        over HTTP; otherwise it is treated as a local file path.

        Args:
            url_or_path: URL or local file path to an HTML document.

        Returns:
            Extracted text content (tags stripped, whitespace normalised).

        Raises:
            ValueError: If no text could be extracted.
            FileNotFoundError: If the local path does not exist.
        """
        import httpx
        from bs4 import BeautifulSoup

        if url_or_path.startswith(("http://", "https://")):
            logger.info("Fetching HTML from URL: %s", url_or_path)
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url_or_path, follow_redirects=True)
                resp.raise_for_status()
                html = resp.text
        else:
            self._ensure_file_exists(url_or_path)
            with open(url_or_path, "r", encoding="utf-8", errors="replace") as fh:
                html = fh.read()

        soup = BeautifulSoup(html, "html.parser")

        # Remove script / style elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Prefer content from <article> or <main>, otherwise use <body>
        container = soup.find("article") or soup.find("main") or soup.find("body")
        if container is None:
            container = soup

        text = container.get_text(separator="\n", strip=True)
        if not text:
            raise ValueError(f"No text content found in HTML: {url_or_path}")

        logger.info("Extracted %d characters from HTML: %s", len(text), url_or_path)
        return text

    # ------------------------------------------------------------------
    # Markdown / plain text
    # ------------------------------------------------------------------

    async def load_markdown(self, path: str) -> str:
        """Read a Markdown file as plain text.

        Args:
            path: Filesystem path to the ``.md`` file.

        Returns:
            Raw file content.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        self._ensure_file_exists(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        logger.info("Loaded %d characters from markdown: %s", len(content), path)
        return content

    async def load_text(self, path: str) -> str:
        """Read a plain text file.

        Args:
            path: Filesystem path to the ``.txt`` file.

        Returns:
            Raw file content.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        self._ensure_file_exists(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        logger.info("Loaded %d characters from text file: %s", len(content), path)
        return content

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_file_exists(path: str) -> None:
        """Raise ``FileNotFoundError`` if *path* does not exist."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Document not found: {path}")
