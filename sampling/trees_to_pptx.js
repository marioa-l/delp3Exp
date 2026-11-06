// Draw fitted decision trees as trees (not as cascades): one editable slide
// per tree, every box a native shape, questions as rounded boxes, leaves
// coloured by the recommended sampler, edges labelled yes / no.
//
// Internal nodes sit centred over their subtree; leaves are spaced evenly
// across the slide width, so a depth-5 tree with 16 leaves still fits.
//
//   node sampling/trees_to_pptx.js <out.pptx> <tree1.json> [<tree2.json> ...]

const fs = require("fs");
const pptxgen = require("pptxgenjs");

const INK      = "1B1D21";
const NODE_BG  = "F2F2F2";
const NODE_ED  = "333333";
const WORLDS   = "0072B2";
const PROGRAMS = "D55E00";
const MUTED    = "666666";
const BODY     = "Calibri";

const outPath = process.argv[2];
const paths = process.argv.slice(3);
if (!outPath || !paths.length) {
  console.error("usage: node sampling/trees_to_pptx.js <out.pptx> <tree.json>...");
  process.exit(1);
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";          // 13.33 x 7.5 in
const SLIDE_W = 13.33, SLIDE_H = 7.5;

function leaves(n) { return n.kind === "leaf" ? 1 : leaves(n.yes) + leaves(n.no); }
function depth(n)  { return n.kind === "leaf" ? 0 : 1 + Math.max(depth(n.yes), depth(n.no)); }

// Assign x (leaf slots, left to right) and y (level) to every node.
function layout(node, level, slot) {
  node.level = level;
  if (node.kind === "leaf") { node.slot = slot; return slot + 1; }
  const next = layout(node.yes, level + 1, slot);
  const end = layout(node.no, level + 1, next);
  node.slot = (node.yes.slot + node.no.slot) / 2;
  return end;
}

function drawTree(tree, title) {
  const s = pres.addSlide();
  const nLeaves = leaves(tree);
  const nLevels = depth(tree) + 1;
  layout(tree, 0, 0);

  const margin = 0.4;
  const colW = (SLIDE_W - 2 * margin) / nLeaves;          // one slot per leaf
  const top = 0.55, bottom = 0.35;
  const levelH = (SLIDE_H - top - bottom) / nLevels;
  const boxH = Math.min(0.62, levelH * 0.55);
  // Questions may span more than one slot; leaves keep to their slot.
  const qW = Math.min(2.7, Math.max(colW * 1.9, 1.3));
  const leafW = Math.min(2.4, Math.max(colW * 0.9, 0.62));
  const fontQ = nLeaves > 8 ? 8 : 11;
  const fontL = nLeaves > 8 ? 7.5 : 10.5;

  s.addText(title, {
    x: margin, y: 0.1, w: 6, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11, color: MUTED, objectName: "title",
  });

  const cx = n => margin + (n.slot + 0.5) * colW;
  const cy = n => top + n.level * levelH + boxH / 2;

  let k = 0;
  (function draw(n) {
    const x = cx(n), y = cy(n);
    if (n.kind === "node") {
      for (const b of ["yes", "no"]) {
        const c = n[b];
        const x2 = cx(c), y2 = cy(c);
        // Elbow: down from the question, across, down into the child.
        const ym = y + boxH / 2 + (y2 - boxH / 2 - (y + boxH / 2)) / 2;
        s.addShape(pres.ShapeType.line, { x, y: y + boxH / 2, w: 0, h: ym - (y + boxH / 2),
          line: { color: NODE_ED, width: 1 }, objectName: `e${k}-a` });
        s.addShape(pres.ShapeType.line, { x: Math.min(x, x2), y: ym, w: Math.abs(x2 - x), h: 0,
          line: { color: NODE_ED, width: 1 }, objectName: `e${k}-b` });
        s.addShape(pres.ShapeType.line, { x: x2, y: ym, w: 0, h: y2 - boxH / 2 - ym,
          line: { color: NODE_ED, width: 1 }, objectName: `e${k}-c` });
        // Label beside the drop into the child, not on top of the line.
        s.addText(b, {
          x: x2 - 0.62, y: ym + 0.02, w: 0.58, h: 0.2, isTextBox: true, margin: 0,
          fontFace: BODY, fontSize: fontL, italic: true, color: MUTED,
          align: "right", valign: "top", objectName: `e${k}-label`,
        });
        k++;
      }
      s.addShape(pres.ShapeType.roundRect, {
        x: x - qW / 2, y: y - boxH / 2, w: qW, h: boxH, rectRadius: 0.08,
        fill: { color: NODE_BG }, line: { color: NODE_ED, width: 1 }, objectName: `q${k}`,
      });
      s.addText(`${n.a} < ${n.b}`, {
        x: x - qW / 2 + 0.05, y: y - boxH / 2, w: qW - 0.1, h: boxH, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: fontQ, color: INK, align: "center", valign: "middle",
        objectName: `q${k}-text`,
      });
      draw(n.yes); draw(n.no);
      return;
    }
    const total = n.n_p + n.n_w;
    const purity = total ? Math.round(100 * Math.max(n.n_p, n.n_w) / total) : 0;
    const color = n.cls === "worlds" ? WORLDS : PROGRAMS;
    s.addShape(pres.ShapeType.roundRect, {
      x: x - leafW / 2, y: y - boxH / 2, w: leafW, h: boxH, rectRadius: 0.08,
      fill: { color }, line: { color: NODE_ED, width: 1 }, objectName: `leaf${k}`,
    });
    s.addText([
      { text: n.cls, options: { bold: true, breakLine: true } },
      { text: `n=${total}  ${purity}%` },
    ], {
      x: x - leafW / 2, y: y - boxH / 2, w: leafW, h: boxH, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: fontL, color: "FFFFFF", align: "center", valign: "middle",
      objectName: `leaf${k}-text`,
    });
    k++;
  })(tree);
}

for (const p of paths) {
  const data = JSON.parse(fs.readFileSync(p, "utf8"));
  const d = data.depth !== undefined ? data.depth : depth(data.tree);
  const acc = data.held_out_accuracy !== undefined
    ? `  held-out accuracy ${Number(data.held_out_accuracy).toFixed(4)}` : "";
  drawTree(data.tree, `depth ${d}${acc}`);
}

pres.writeFile({ fileName: outPath }).then(f => console.log("wrote", f));
