// Render the validated decision tree as an editable PowerPoint slide.
//
// Reads results_tte/analysis/final_tree.json, so the slide always matches the
// tree that was actually fitted; nothing about the structure is hard-coded.
// Every box is a native shape, so the diagram can be rearranged in PowerPoint
// without regenerating anything.
//
//   node sampling/tree_to_pptx.js <tree.json> <out.pptx>

const fs = require("fs");
const pptxgen = require("pptxgenjs");

const INK      = "1B1D21";
const NODE_BG  = "F2F2F2";
const NODE_ED  = "333333";
const WORLDS   = "0072B2";
const PROGRAMS = "D55E00";
const MUTED    = "666666";
const BODY     = "Calibri";

const treePath = process.argv[2] || "results_tte/analysis/final_tree.json";
const outPath  = process.argv[3] || "images/final_tree.pptx";
const tree = JSON.parse(fs.readFileSync(treePath, "utf8")).tree;

// Flatten into the indented decision list the tree amounts to: a question
// whose branches mostly terminate reads better as a cascade, and the width
// then grows with the indent instead of with the number of leaves.
const rows = [];
(function walk(node, indent) {
  if (node.kind === "leaf") { rows.push({ indent, kind: "leaf", node }); return; }
  rows.push({ indent, kind: "node", node });
  for (const branch of ["yes", "no"]) {
    const child = node[branch];
    if (child.kind === "leaf") rows.push({ indent: indent + 1, kind: "leaf", node: child, branch });
    else { rows.push({ indent: indent + 1, kind: "label", branch }); walk(child, indent + 1); }
  }
})(tree, 0);

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";          // 13.3 x 7.5 in
pres.title = "Decision tree";
const s = pres.addSlide();

const X0 = 0.45, Y0 = 0.45;
const DX = 0.38;                      // indent step, inches
const ROW = 0.46;                     // row pitch, inches
const ROW_H = 0.36;

rows.forEach((r, i) => {
  const y = Y0 + i * ROW;
  const x = X0 + r.indent * DX;

  if (r.kind === "label") {
    s.addText(`${r.branch}:`, {
      x, y, w: 0.6, h: 0.26, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11, italic: true, color: MUTED,
      valign: "middle", objectName: `label-${i}`,
    });
    return;
  }

  if (r.kind === "node") {
    const label = `${r.node.a}  <  ${r.node.b} ?`;
    const w = Math.max(2.6, 0.105 * label.length + 0.45);
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w, h: ROW_H, rectRadius: 0.06,
      fill: { color: NODE_BG }, line: { color: NODE_ED, width: 1 },
      objectName: `q-${i}`,
    });
    s.addText(label, {
      x: x + 0.08, y, w: w - 0.16, h: ROW_H, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12, color: INK, valign: "middle",
      objectName: `q-text-${i}`,
    });
    return;
  }

  const total = r.node.n_p + r.node.n_w;
  const purity = total ? Math.max(r.node.n_p, r.node.n_w) / total : 0;
  const color = r.node.cls === "worlds" ? WORLDS : PROGRAMS;
  const prefix = r.branch ? `${r.branch}: ` : "";
  const label = `${prefix}sample ${r.node.cls.toUpperCase()}   (n=${total}, ${Math.round(purity * 100)}%)`;
  const w = Math.max(2.9, 0.098 * label.length + 0.4);
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h: ROW_H, rectRadius: 0.06,
    fill: { color }, line: { color: NODE_ED, width: 1 },
    objectName: `leaf-${i}`,
  });
  s.addText(label, {
    x: x + 0.08, y, w: w - 0.16, h: ROW_H, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, color: "FFFFFF", valign: "middle",
    objectName: `leaf-text-${i}`,
  });
});

pres.writeFile({ fileName: outPath }).then(f => console.log("wrote", f));
