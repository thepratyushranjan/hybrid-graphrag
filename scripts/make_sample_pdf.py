"""Generate data/samples/vendor_compliance_bulletin_q3_2025.pdf.

The PDF deliberately mixes three kinds of content so ingestion has to handle all of them:
  page 1: normal English text layer + an embedded IMAGE containing Hindi text (needs OCR)
  page 2: a fully SCANNED page (image only, no text layer; English + Hindi, needs OCR)
  page 3: normal English text layer

Run once (needs Noto fonts from the host):
  docker compose run --rm --no-deps -v /usr/share/fonts/truetype/noto:/fonts:ro \
      api python scripts/make_sample_pdf.py
"""

import random
from pathlib import Path

import pymupdf

FONTS = pymupdf.Archive("/fonts")
CSS = """
@font-face { font-family: latin; src: url(NotoSans-Regular.ttf); }
@font-face { font-family: deva; src: url(NotoSansDevanagari-Regular.ttf); }
body { font-family: latin, deva; font-size: 11pt; line-height: 1.45; }
h1 { font-size: 16pt; } h2 { font-size: 13pt; }
.hi { font-family: deva; font-size: 13pt; }
"""
A4 = pymupdf.paper_rect("a4")
MARGIN = 50
OUT = Path(__file__).resolve().parent.parent / "data" / "samples" / "vendor_compliance_bulletin_q3_2025.pdf"


def render_png(html: str, width: float, height: float, dpi: int) -> bytes:
    """Lay out HTML (HarfBuzz shapes the Devanagari correctly) and rasterise it to a PNG."""
    tmp = pymupdf.open()
    page = tmp.new_page(width=width, height=height)
    page.insert_htmlbox(pymupdf.Rect(10, 10, width - 10, height - 10), html, css=CSS, archive=FONTS)
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    tmp.close()
    return png


def add_text_page(doc: pymupdf.Document, html: str, top: float = MARGIN, bottom: float | None = None) -> pymupdf.Page:
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_htmlbox(
        pymupdf.Rect(MARGIN, top, A4.width - MARGIN, bottom or A4.height - MARGIN), html, css=CSS, archive=FONTS
    )
    return page


PAGE1 = """
<h1>Vendor Compliance Bulletin — Q3 2025</h1>
<p><i>Fictional sample document for the Hybrid GraphRAG demo. Issued by the Group Compliance Office of
Ganga Infra Holdings Ltd, Lucknow. Contact: Sunita Yadav, Group Head of Safety and Compliance.</i></p>
<h2>Clause 7.2 — Customer Data Residency</h2>
<p>From 1 April 2026, every vendor that stores or processes customer data on behalf of a Ganga subsidiary must keep
that data, including backups and disaster-recovery copies, on servers physically located in India.
Clause 7.2 applies to any vendor and any of its subcontractors that handle customer data.</p>
<p>Impacted vendors identified in this review: <b>DataSecure Cloud Pvt Ltd</b>, which hosts meter and customer data
for Ganga Smart Power Ltd and keeps a disaster-recovery site in Singapore, and <b>NovaGrid Electronics</b>, which
collects meter readings for Ganga Smart Power Ltd and subcontracts analytics to DataSecure Cloud.
Both vendors must submit a migration plan to Neha Kapoor, Managing Director of Ganga Smart Power, by 31 January 2026.</p>
<h2>Clause 9.1 — Safety Certification of Supplied Materials</h2>
<p>Every consignment of structural material must arrive with a BIS safety certificate. The notice below was
issued to site teams in Hindi.</p>
"""

HINDI_NOTICE = """
<div class="hi">
<p><b>सूचना: धारा 9.1 का पालन अनिवार्य</b></p>
<p>शक्ति स्टील वर्क्स द्वारा भेजी गई हर गार्ड रेल खेप के साथ बीआईएस सुरक्षा प्रमाणपत्र होना ज़रूरी है।
बिना प्रमाणपत्र वाली खेप को लखनऊ–सीतापुर राजमार्ग स्थल पर स्वीकार नहीं किया जाएगा।
यह नियम गंगा रोडवेज़ के सभी परियोजना स्थलों पर लागू है। जानकारी के लिए विक्रम सिंह से संपर्क करें।</p>
</div>
"""

SCANNED = """
<h2>Annex A — Penalties and Incident Record (scanned copy)</h2>
<p>A vendor that breaches Clause 7.2 pays a penalty of 2% of its annual contract value for each month of
non-compliance. A vendor that breaches Clause 9.1 has the affected consignment rejected and is put on a
90-day probation.</p>
<p>Incident record: on 14 October 2025 two workers were injured on the Lucknow–Sitapur Highway Upgrade while
installing a guard rail supplied by Shakti Steel Works. The consignment had no safety certificate.
Ganga Roadways Pvt Ltd placed Shakti Steel Works on probation from 20 October 2025.</p>
<p class="hi">घटना के बाद गंगा रोडवेज़ ने शक्ति स्टील वर्क्स को 90 दिनों की परिवीक्षा पर रखा।
सभी नई खेपों की जाँच सुनीता यादव की टीम करेगी।</p>
<p>Signed: Rohit Verma, Chief Financial Officer</p>
"""

PAGE3 = """
<h2>Clause 11.4 — Water Treatment Chemicals</h2>
<p>AquaPure Filters, the membrane and chemical supplier of Ganga Water Systems Ltd in Prayagraj, must publish
monthly test reports for every chemical batch delivered to Prayagraj Water Treatment Plant 3. Arif Khan,
Managing Director of Ganga Water Systems, will review the reports every quarter.</p>
<h2>Vendors not impacted this quarter</h2>
<p>Brightline Logistics, the transport vendor of Ganga Roadways Pvt Ltd and Ganga Water Systems Ltd, does not
handle customer data or structural material, so neither Clause 7.2 nor Clause 9.1 applies to it.</p>
<h2>Summary</h2>
<p>Clause 7.2 impacts DataSecure Cloud Pvt Ltd and NovaGrid Electronics (both connected to Ganga Smart Power Ltd).
Clause 9.1 impacts Shakti Steel Works (connected to Ganga Roadways Pvt Ltd).
Clause 11.4 impacts AquaPure Filters (connected to Ganga Water Systems Ltd).</p>
"""


def main() -> None:
    doc = pymupdf.open()

    # Page 1: text layer, plus the Hindi notice embedded as an image
    page = add_text_page(doc, PAGE1, bottom=560)
    notice = render_png(HINDI_NOTICE, width=480, height=150, dpi=200)
    page.insert_image(pymupdf.Rect(MARGIN, 570, A4.width - MARGIN, 570 + 150 * (A4.width - 2 * MARGIN) / 480),
                      stream=notice)

    # Page 2: scanned — rasterise a full page, rotate it slightly like a real scan, keep no text layer
    scan_png = render_png(SCANNED, width=A4.width - 2 * MARGIN, height=420, dpi=150)
    pix = pymupdf.Pixmap(scan_png)
    random.seed(7)
    for _ in range(pix.width * pix.height // 400):  # light speckle noise
        x, y = random.randrange(pix.width), random.randrange(pix.height)
        pix.set_pixel(x, y, (170, 170, 170))
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_image(pymupdf.Rect(MARGIN, MARGIN, A4.width - MARGIN, MARGIN + 420), pixmap=pix, rotate=0)

    # Page 3: text layer
    add_text_page(doc, PAGE3)

    doc.set_metadata({"title": "Vendor Compliance Bulletin Q3 2025 (fictional sample)", "author": "Ganga Infra Holdings"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT, garbage=4, deflate=True)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB, {doc.page_count} pages)")


if __name__ == "__main__":
    main()
