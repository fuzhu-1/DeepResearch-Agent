"""PDF generation utilities."""
import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)

_has_cjk_fonts = False
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    _cjk_candidates = {
        "NotoSansCJK-Regular": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "NotoSansCJK-Bold": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    }
    _windows_fonts = [
        ("msyh.ttc", "MicrosoftYaHei"),
        ("msyhbd.ttc", "MicrosoftYaHei-Bold"),
        ("simsun.ttc", "SimSun"),
        ("simhei.ttf", "SimHei"),
    ]
    _registered = {}
    for name, path in _cjk_candidates.items():
        try:
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont(name, path))
                _registered[name] = path
        except Exception:
            pass
    if not _registered:
        for fname, fname_alias in _windows_fonts:
            for base in ["C:/Windows/Fonts", "C:\\Windows\\Fonts"]:
                fpath = os.path.join(base, fname)
                try:
                    if os.path.exists(fpath):
                        pdfmetrics.registerFont(TTFont(fname_alias, fpath))
                        _registered[fname_alias] = fpath
                except Exception:
                    pass
    if _registered:
        _has_cjk_fonts = True
        _FONT_REGULAR = list(_registered.keys())[0]
        _FONT_BOLD = list(_registered.keys())[1] if len(_registered) > 1 else _FONT_REGULAR
    else:
        _FONT_REGULAR = "Helvetica"
        _FONT_BOLD = "Helvetica-Bold"
except ImportError:
    _FONT_REGULAR = "Helvetica"
    _FONT_BOLD = "Helvetica-Bold"

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    ListFlowable, ListItem, HRFlowable, Table, TableStyle,
)

_PRIMARY = colors.HexColor("#1a365d")
_SECONDARY = colors.HexColor("#2b6cb0")
_BODY = colors.HexColor("#2d3748")
_MUTED = colors.HexColor("#718096")
_BORDER = colors.HexColor("#e2e8f0")
_BG_CODE = colors.HexColor("#f7fafc")


class _ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename, title="", **kw):
        BaseDocTemplate.__init__(self, filename, **kw)
        self._title = title
        frame = Frame(self.leftMargin, self.bottomMargin + 30, self.width, self.height - 30, id="normal")
        self.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=self._header_footer)])

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(_SECONDARY)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, doc.height + doc.bottomMargin + 15, doc.width + doc.leftMargin, doc.height + doc.bottomMargin + 15)
        canvas.setFont(_FONT_REGULAR if _has_cjk_fonts else "Helvetica", 8)
        canvas.setFillColor(_MUTED)
        canvas.drawString(doc.leftMargin, doc.height + doc.bottomMargin + 18, "DeepResearch-Agent")
        canvas.drawCentredString(doc.width / 2 + doc.leftMargin, 15, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()


async def generate_pdf(markdown_content, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not markdown_content or not markdown_content.strip():
        markdown_content = "# Report\n\nEmpty."
    _build(markdown_content, output_path)
    return output_path


async def generate_pdf_from_html(html, output_path):
    text = re.sub(r"<[^>]+>", "", html)
    return await generate_pdf(text, output_path)


# ---- Build ----

def _build(md, out):
    first = md.split("\n")[0].lstrip("# ").strip() or "Report"
    doc = _ReportDocTemplate(out, title=first, pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm, topMargin=2.5*cm, bottomMargin=2.0*cm)
    story = []
    in_code, buf = False, []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.strip().startswith("```"):
            if in_code:
                story.append(_code("\n".join(buf)))
                buf = []; in_code = False
            else:
                in_code = True
            i += 1; continue
        if in_code:
            buf.append(l); i += 1; continue
        if l.strip() == "":
            i += 1; continue
        if l.strip() == "---":
            story.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER, spaceBefore=8, spaceAfter=8))
            i += 1; continue

        if l.strip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(cells) > 1 and all(set(c) <= set("-: ") for c in cells):
                    i += 1
                    continue  # separator row
                rows.append(cells)
                i += 1
            if rows:
                story.append(_table(rows))
            continue
        m = re.match(r"^(#{1,4})\s+(.+)$", l)
        if m:
            story.append(_h(m.group(2), len(m.group(1))))
            if len(m.group(1)) == 1:
                story.append(HRFlowable(width="100%", thickness=1, color=_PRIMARY, spaceBefore=2, spaceAfter=12))
            i += 1; continue
        if l.startswith(">"):
            story.append(_bq(l.lstrip(">").strip()))
            i += 1; continue
        if re.match(r"^[\-\*]\s+", l):
            items = []
            while i < len(lines) and re.match(r"^[\-\*]\s+", lines[i]):
                items.append(re.sub(r"^[\-\*]\s+", "", lines[i]))
                i += 1
            story.append(_li(items, "bullet"))
            continue
        if re.match(r"^\d+\.\s+", l):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i]))
                i += 1
            story.append(_li(items, "1"))
            continue
        para_lines = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("```") and lines[i].strip() != "---" and not re.match(r"^(#{1,4})\s", lines[i]) and not re.match(r"^[\-\*]\s", lines[i]) and not re.match(r"^\d+\.\s", lines[i]):
            para_lines.append(lines[i].strip())
            i += 1
        if para_lines:
            story.append(_p(" ".join(para_lines)))
            i += 1
        i += 1
    if in_code and buf:
        story.append(_code("\n".join(buf)))
    doc.build(story)


# ---- Flowables ----

def _h(text, level):
    sizes = {1: 22, 2: 16, 3: 13, 4: 11}
    clrs = {1: _PRIMARY, 2: _SECONDARY, 3: _BODY, 4: _BODY}
    s = ParagraphStyle(f"H{level}", fontName=_FONT_BOLD if _has_cjk_fonts else "Helvetica-Bold",
        fontSize=sizes.get(level, 12), leading=sizes.get(level,12)*1.4,
        spaceBefore={1:24,2:18,3:14,4:10}.get(level,8), spaceAfter=6,
        textColor=clrs.get(level, _BODY))
    try:
        return Paragraph(_esc(_strip_md(text)), s)
    except Exception:
        return Paragraph(_esc(text), s)


def _p(text):
    s = ParagraphStyle("Body", fontName=_FONT_REGULAR if _has_cjk_fonts else "Helvetica",
        fontSize=10.5, leading=17, spaceBefore=3, spaceAfter=6, alignment=TA_JUSTIFY, textColor=_BODY)
    try:
        return Paragraph(_fmt(text), s)
    except Exception:
        return Paragraph(_esc(_strip_md(text)), s)


def _code(text):
    s = ParagraphStyle("Code", fontName="Courier", fontSize=8, leading=11,
        spaceBefore=6, spaceAfter=6, leftIndent=8, rightIndent=8,
        backColor=_BG_CODE, borderColor=_BORDER, borderWidth=0.5, borderPadding=8,
        textColor=colors.HexColor("#744210"))
    return Paragraph(_esc(text).replace("\n","<br/>"), s)


def _bq(text):
    s = ParagraphStyle("Blockquote", fontName=_FONT_REGULAR if _has_cjk_fonts else "Helvetica",
        fontSize=10, leading=15, leftIndent=14, rightIndent=14,
        spaceBefore=6, spaceAfter=6, textColor=_MUTED,
        borderColor=_SECONDARY, borderLeftWidth=3, borderLeftPadding=10, backColor=_BG_CODE)
    return Paragraph(_esc(text), s)


def _li(items, btype):
    its = []
    for item in items:
        s = ParagraphStyle("Li", fontName=_FONT_REGULAR if _has_cjk_fonts else "Helvetica",
            fontSize=10.5, leading=16, spaceBefore=1, spaceAfter=1, textColor=_BODY)
        try:
            its.append(ListItem(Paragraph(_fmt(item), s)))
        except Exception:
            its.append(ListItem(Paragraph(_esc(_strip_md(item)), s)))
    return ListFlowable(its, bulletType=btype, start=None if btype=="bullet" else "1",
        bulletFontSize=8, leftIndent=20, bulletOffsetY=-1)




def _table(rows):
    """Render markdown table rows as a reportlab Table with escaped text."""
    data = [[_esc(_strip_md(c)) for c in row] for row in rows]
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2fb")),
        ("GRID", (0, 0), (-1, -1), 0.4, _BORDER),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (-1, -1), _BODY),
    ]))
    return t


# ---- Helpers ----

def _esc(t):
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _strip_md(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", t)
    t = re.sub(r"`(.+?)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    return t

def _fmt(t):
    t = _esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'><i>\1</i></font>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" color="#2b6cb0">\1</a>', t)
    return t
