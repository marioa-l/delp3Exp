// Figure: the two sampling approaches over the same scenario space.
//
// Left  — scenario-based (SamplEM): each draw is one scenario and contributes
//         its own probability.
// Right — subprogram-based (SamplAM): the space partitions into regions that
//         induce the same subprogram, so each draw contributes the summed
//         probability of a whole region.
//
// The asymmetry between the two is the point of the figure: a scenario induces
// exactly one subprogram, while a subprogram is induced by many scenarios.
//
// Every element is a native shape, so the diagram stays editable.
//
//   node sampling/approaches_to_pptx.js <out.pptx>

const pptxgen = require("pptxgenjs");

const INK     = "1B1D21";
const MUTED   = "5A5E66";
const SPACE   = "EDEFF2";   // the scenario space
const EDGE    = "333333";
const EM      = "0072B2";   // scenario sampling
const AM      = "D55E00";   // subprogram sampling
const DOT     = "FFFFFF";
const BODY    = "Calibri";

const out = process.argv[2] || "images/enfoques_new.pptx";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";          // 13.3 x 7.5 in
pres.title = "Sampling approaches";
const s = pres.addSlide();

const D = 3.9;                        // circle diameter
const CY = 1.52;                      // circle top
const LX = 0.95, RX = 8.1;            // circle left edges

// ---------------------------------------------------------------- helpers
const circle = (x, name) =>
  s.addShape(pres.ShapeType.ellipse, {
    x, y: CY, w: D, h: D,
    fill: { color: SPACE }, line: { color: EDGE, width: 2 },
    objectName: name,
  });

// a scenario, drawn as a small dot; `on` marks it as drawn by the sampler
const dot = (cx, cy, label, on, accent, name) => {
  const d = 0.46;
  s.addShape(pres.ShapeType.ellipse, {
    x: cx - d / 2, y: cy - d / 2, w: d, h: d,
    fill: { color: on ? accent : DOT },
    line: { color: on ? accent : "9AA0A6", width: on ? 2 : 1 },
    objectName: name,
  });
  s.addText(label, {
    x: cx - 0.33, y: cy - 0.16, w: 0.66, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10, color: on ? "FFFFFF" : INK,
    align: "center", valign: "middle", objectName: name + "-t",
  });
};

// Scenarios in polar coordinates about the centre, each placed inside one
// wedge and short of the rim, so the wedge labels can sit outside the circle
// without colliding with them. [angle in degrees, radius as a fraction of D]
const P = [
  [330, 0.20], [ 20, 0.30], [ 45, 0.19], [ 95, 0.30],
  [115, 0.20], [160, 0.29], [185, 0.17], [215, 0.30],
  [268, 0.26], [290, 0.17], [ 60, 0.32],
];
const at = (x0, p) => {
  const a = p[0] * Math.PI / 180;
  return [x0 + D / 2 + p[1] * D * Math.cos(a),
          CY + D / 2 + p[1] * D * Math.sin(a)];
};

// ============================================ left: scenario-based sampling
circle(LX, "space-left");
const drawnLeft = new Set([0, 5, 8]);
P.forEach((p, i) => {
  const [cx, cy] = at(LX, p);
  dot(cx, cy, `σ${sub(i + 1)}`, drawnLeft.has(i), EM, `l-dot-${i}`);
});

// ====================================== right: subprogram-based sampling
// Wedges partition the space: each is the set of scenarios inducing one
// subprogram. Drawing one wedge collects the probability of all of them.
const wedges = [
  { from: 300, to: 360, label: "𝒫¹" },
  { from: 0,   to:  65, label: "𝒫²" },
  { from: 65,  to: 130, label: "𝒫³" },
  { from: 130, to: 196, label: "𝒫⁴" },
  { from: 196, to: 250, label: "𝒫⁵" },
  { from: 250, to: 300, label: "𝒫⁶" },
];
const picked = 2;                      // the wedge the sampler drew
wedges.forEach((w, i) => {
  s.addShape(pres.ShapeType.pie, {
    x: RX, y: CY, w: D, h: D,
    angleRange: [w.from, w.to],
    fill: { color: i === picked ? AM : SPACE,
            transparency: i === picked ? 25 : 0 },
    line: { color: EDGE, width: 1.25 },
    objectName: `wedge-${i}`,
  });
  const mid = ((w.from + w.to) / 2) * Math.PI / 180;
  const r = D * 0.60;                  // outside the rim: no collision with dots
  const cx = RX + D / 2 + r * Math.cos(mid);
  const cy = CY + D / 2 + r * Math.sin(mid);
  s.addText(w.label, {
    x: cx - 0.35, y: cy - 0.18, w: 0.7, h: 0.36, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13, bold: true,
    color: i === picked ? AM : MUTED,
    align: "center", valign: "middle", objectName: `wedge-label-${i}`,
  });
});
// the same scenarios, now seen as members of the regions
P.forEach((p, i) => {
  const [cx, cy] = at(RX, p);
  dot(cx, cy, `σ${sub(i + 1)}`, false, AM, `r-dot-${i}`);
});

// ================================================================ labels
const panelLabel = (x, tag, name, rule, color, key) => {
  s.addText([{ text: `(${tag})  `, options: { bold: true } },
             { text: name, options: { bold: true, color } }], {
    x, y: CY + D + 0.46, w: D, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 14, color: INK, align: "center",
    objectName: key + "-name",
  });
  s.addText(rule, {
    x: x - 0.35, y: CY + D + 0.80, w: D + 0.7, h: 0.6, isTextBox: true,
    margin: 0, fontFace: BODY, fontSize: 12, color: MUTED, align: "center",
    objectName: key + "-rule",
  });
};

panelLabel(LX, "a", "scenario-based  (SamplEM)",
  "draw σ, build the induced 𝒫 σ,\nadd ℰ(σ) to the interval", EM, "left");
panelLabel(RX, "b", "subprogram-based  (SamplAM)",
  "draw a subprogram, add the summed\nmass Σ ℰ(σ) of every σ inducing it", AM, "right");

// the asymmetry, stated between the panels
s.addShape(pres.ShapeType.line, {
  x: LX + D + 0.42, y: CY + 0.5, w: 0, h: D - 1.0,
  line: { color: "C9CBCD", width: 1, dashType: "dash" },
  objectName: "divider",
});
s.addText("one scenario induces\nexactly one subprogram,\nbut a subprogram is\ninduced by many scenarios",
  { x: LX + D + 0.12, y: CY + 1.25, w: 2.6, h: 1.3, isTextBox: true,
    margin: 0, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTED,
    align: "center", valign: "middle", objectName: "asymmetry" });

// the space itself, named once
const spaceName = (x, key) => s.addText(
  [{ text: "scenario space  2", options: {} },
   { text: "E", options: { superscript: true } }],
  { x, y: CY - 0.92, w: D, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, color: MUTED, align: "center",
    objectName: key });
spaceName(LX, "space-name-left");
spaceName(RX, "space-name-right");

function sub(n) {
  const map = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅",
                6: "₆", 7: "₇", 8: "₈", 9: "₉" };
  return String(n).split("").map(c => map[c]).join("");
}

pres.writeFile({ fileName: out }).then(f => console.log("wrote", f));
