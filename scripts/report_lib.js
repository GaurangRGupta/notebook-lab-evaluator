// Shared builders for the Group 3 explanation documents (A4, Arial).
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType, Header, Footer, PageNumber, PageBreak,
} = require("docx");

const CONTENT_W = 9026; // A4 width 11906 - 2 x 1440 margins
const FONT = "Arial";
const MONO = "Consolas";
const C = { accent: "1F4E79", head: "D5E3F0", code: "F3F4F6", border: "BFC7D1",
  amber: "FFF4D6", amberB: "C98A00", blue: "E8F0FA", blueB: "2E6DB4", green: "E7F4EA", greenB: "2E7D32",
  red: "FDECEA", redB: "C0392B", grey: "EEF0F3", greyB: "6B7280" };

// inline markup: **bold**, `code`, _italic_
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;  // no _italic_: identifiers contain underscores
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: MONO, ...base, size: 19 }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const P = (t, opts = {}) => new Paragraph({ children: runs(t), spacing: { after: 120 }, ...opts });

function B(items, level = 0) {
  const out = [];
  for (const it of items) {
    if (Array.isArray(it)) out.push(...B(it, level + 1));
    else out.push(new Paragraph({ numbering: { reference: "bullets", level }, children: runs(it), spacing: { after: 60 } }));
  }
  return out;
}

let numRef = 0;
const numberingConfigs = [{
  reference: "bullets",
  levels: [0, 1, 2].map((l) => ({ level: l, format: LevelFormat.BULLET, text: ["•", "◦", "▪"][l], alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540 + l * 360, hanging: 270 } } } })),
}];
function N(items) { // each call restarts at 1
  const ref = `num${++numRef}`;
  numberingConfigs.push({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] });
  return items.map((it) => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(it), spacing: { after: 60 } }));
}

const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: C.border };
const borders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };
function T(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const w = widths.map((x) => Math.floor(x * scale));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  const mk = (txt, i, head) => new TableCell({
    borders, width: { size: w[i], type: WidthType.DXA },
    shading: head ? { fill: C.head, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: String(txt).split("\n").map((line) => new Paragraph({ children: runs(line, head ? { bold: true, size: 20 } : { size: 20 }) })),
  });
  const trs = [];
  if (headers) trs.push(new TableRow({ tableHeader: true, children: headers.map((h, i) => mk(h, i, true)) }));
  rows.forEach((r) => trs.push(new TableRow({ children: r.map((c, i) => mk(c, i, opts.firstColHead && i === 0)) })));
  return [new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w, rows: trs }), new Paragraph({ spacing: { after: 80 }, children: [] })];
}

function Code(lines) {
  const arr = Array.isArray(lines) ? lines : lines.split("\n");
  const longest = Math.max(...arr.map((l) => l.length));
  const size = longest > 104 ? 13 : longest > 90 ? 14 : longest > 80 ? 16 : 18; // keep ASCII diagrams on one line
  return arr.map((l, i) => new Paragraph({
    shading: { fill: C.code, type: ShadingType.CLEAR, color: "auto" },
    spacing: { before: i === 0 ? 80 : 0, after: i === arr.length - 1 ? 140 : 0, line: 240 },
    indent: { left: 200, right: 200 },
    children: [new TextRun({ text: l.length ? l : " ", font: MONO, size })],
  }));
}

const KIND = {
  synthetic: [C.amber, C.amberB, "SYNTHETIC DEVELOPMENT ONLY"],
  planned: [C.grey, C.greyB, "PLANNED / NOT YET IMPLEMENTED"],
  key: [C.green, C.greenB, "KEY IDEA"],
  warn: [C.red, C.redB, "IMPORTANT"],
  viva: [C.blue, C.blueB, "VIVA TIP"],
  example: [C.blue, C.blueB, "EXAMPLE"],
};
function Note(kind, text, label) {
  const [fill, bc, def] = KIND[kind];
  const texts = Array.isArray(text) ? text : [text];
  const bd = { style: BorderStyle.SINGLE, size: 18, color: bc, space: 6 };
  return texts.map((t, i) => new Paragraph({
    shading: { fill, type: ShadingType.CLEAR, color: "auto" },
    border: { left: bd },
    indent: { left: 160, right: 160 },
    spacing: { before: i === 0 ? 120 : 0, after: i === texts.length - 1 ? 160 : 40 },
    children: i === 0 ? [new TextRun({ text: (label || def) + ": ", bold: true, color: bc }), ...runs(t)] : runs(t),
  }));
}

// "What / Why / How / Contribution" block
function WWHC(what, why, how, contrib) {
  return T(null, [["**What is it?**", what], ["**Why do we use it?**", why], ["**How does it work?**", how],
    ["**What does it contribute?**", contrib]], [22, 78]);
}

const PB = () => new Paragraph({ children: [new PageBreak()] });

function build(title, subtitle, children, file) {
  const doc = new Document({
    creator: "Group 3", title,
    styles: {
      default: { document: { run: { font: FONT, size: 21 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: FONT, color: C.accent }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 26, bold: true, font: FONT, color: C.accent }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 22, bold: true, font: FONT, color: "333333" }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
      ],
    },
    numbering: { config: numberingConfigs },
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: title, size: 16, color: "808080" })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", size: 16, color: "808080" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080" })] })] }) },
      children: [
        new Paragraph({ spacing: { before: 600, after: 120 }, children: [new TextRun({ text: title, bold: true, size: 44, color: C.accent })] }),
        new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: subtitle, size: 24, color: "555555" })] }),
        new Paragraph({ spacing: { after: 300 }, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: C.accent, space: 4 } },
          children: [new TextRun({ text: "NLP Course Project 2026 · AI University Platform · Team J013, J015, J016, J017, J018", size: 20, color: "555555" })] }),
        ...children,
      ],
    }],
  });
  return Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(file, buf); console.log("wrote", file); });
}

module.exports = { H1, H2, H3, P, B, N, T, Code, Note, WWHC, PB, build };
