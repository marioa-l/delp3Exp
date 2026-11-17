// Draw fitted decision trees as trees (not as cascades): one editable slide
// per tree, every box a native shape, questions as rounded boxes, leaves
// coloured by the recommended sampler, edges labelled yes / no, with the
// "no" branch on the left and the "yes" branch on the right.
//
// Internal nodes sit centred over their subtree; leaves are spaced evenly
// across the slide width, so a depth-5 tree with 16 leaves still fits. The
// depth and accuracy go to the slide notes, not onto the slide: the figure
// title belongs in the caption.
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

// Assign x (leaf slots, left to right) and y (level) to every node; the
// "no" subtree takes the slots on the left, the "yes" subtree those on the right.
function layout(node, level, slot) {
  node.level = level;
  node.lo = slot;                                  // first leaf slot below
  if (node.kind === "leaf") { node.slot = node.hi = slot; return slot + 1; }
  const next = layout(node.no, level + 1, slot);
  const end = layout(node.yes, level + 1, next);
  node.slot = (node.no.slot + node.yes.slot) / 2;
  node.hi = end - 1;                               // last leaf slot below
  return end;
}

function drawTree(tree, notes) {
  const s = pres.addSlide();
  s.addNotes(notes);
  const nLeaves = leaves(tree);
  const nLevels = depth(tree) + 1;
  layout(tree, 0, 0);

  const big = nLeaves <= 8;
  const margin = 0.3;
  const colW = (SLIDE_W - 2 * margin) / nLeaves;          // one slot per leaf
  // Levels share the height, up to 1.9 in each; the drawing is centred.
  const levelH = Math.min(1.9, (SLIDE_H - 0.5) / nLevels);
  const top = (SLIDE_H - nLevels * levelH) / 2 + 0.1;
  const boxH = Math.min(big ? 0.95 : 0.8, levelH * 0.62);
  // A question stays within the leaf slots below it (so neighbours at the
  // same level never overlap): its half-width is the distance from its centre
  // to the nearer edge of that span. Leaves keep to their slot.
  const qWidth = n => Math.min(3.0,
    2 * 0.95 * colW * Math.min(n.slot - n.lo + 0.5, n.hi - n.slot + 0.5));
  const leafW = Math.min(2.4, Math.max(colW * 0.92, 0.7));
  const fontQ = big ? 18 : 10;
  const fontL = big ? 17 : 11;
  const LINE = big ? 2.5 : 1.75;                           // points

  const cx = n => margin + (n.slot + 0.5) * colW;
  const cy = n => top + n.level * levelH + boxH / 2;

  let k = 0;
  (function draw(n) {
    const x = cx(n), y = cy(n);
    if (n.kind === "node") {
      for (const b of ["no", "yes"]) {
        const c = n[b];
        const x2 = cx(c), y2 = cy(c);
        // Elbow: down from the question, across, down into the child, which
        // the arrow points at.
        const ym = y + boxH / 2 + (y2 - boxH / 2 - (y + boxH / 2)) / 2;
        s.addShape(pres.ShapeType.line, { x, y: y + boxH / 2, w: 0, h: ym - (y + boxH / 2),
          line: { color: NODE_ED, width: LINE }, objectName: `e${k}-a` });
        s.addShape(pres.ShapeType.line, { x: Math.min(x, x2), y: ym, w: Math.abs(x2 - x), h: 0,
          line: { color: NODE_ED, width: LINE }, objectName: `e${k}-b` });
        s.addShape(pres.ShapeType.line, { x: x2, y: ym, w: 0, h: y2 - boxH / 2 - ym,
          line: { color: NODE_ED, width: LINE, endArrowType: "triangle" },
          objectName: `e${k}-c` });
        // Label on the outer side of the drop: "no" to its left, "yes" to its right.
        const lw = big ? 0.6 : 0.42;
        s.addText(b, {
          x: b === "no" ? x2 - lw - 0.06 : x2 + 0.06, y: ym + 0.03, w: lw, h: big ? 0.32 : 0.22,
          isTextBox: true, margin: 0, fontFace: BODY, fontSize: fontL, italic: true,
          color: MUTED, align: b === "no" ? "right" : "left", valign: "top",
          objectName: `e${k}-label`,
        });
        k++;
      }
      const qW = qWidth(n);
      s.addShape(pres.ShapeType.roundRect, {
        x: x - qW / 2, y: y - boxH / 2, w: qW, h: boxH, rectRadius: 0.08,
        fill: { color: NODE_BG }, line: { color: NODE_ED, width: LINE }, objectName: `q${k}`,
      });
      s.addText([
        { text: `${n.a} <`, options: { breakLine: true } },
        { text: n.b },
      ], {
        x: x - qW / 2 + 0.04, y: y - boxH / 2, w: qW - 0.08, h: boxH, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: fontQ, color: INK, align: "center", valign: "middle",
        objectName: `q${k}-text`,
      });
      draw(n.no); draw(n.yes);
      return;
    }
    const total = n.n_p + n.n_w;
    const purity = total ? Math.round(100 * Math.max(n.n_p, n.n_w) / total) : 0;
    const color = n.cls === "worlds" ? WORLDS : PROGRAMS;
    s.addShape(pres.ShapeType.roundRect, {
      x: x - leafW / 2, y: y - boxH / 2, w: leafW, h: boxH, rectRadius: 0.08,
      fill: { color }, line: { color: NODE_ED, width: LINE }, objectName: `leaf${k}`,
    });
    s.addText([
      { text: n.cls, options: { bold: true, breakLine: true } },
      { text: big ? `n=${total}  ${purity}%` : `n=${total}`, options: big ? {} : { breakLine: true } },
      ...(big ? [] : [{ text: `${purity}%` }]),
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
    ? `, held-out accuracy ${Number(data.held_out_accuracy).toFixed(4)}` : "";
  drawTree(data.tree, `${p}: depth ${d}${acc}`);
}

pres.writeFile({ fileName: outPath }).then(f => console.log("wrote", f));
