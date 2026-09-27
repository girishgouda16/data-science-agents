import type { Element, Root, RootContent } from "hast";

// A top-level paragraph holding a chart ("**Title** — caption:\n![alt](src)") becomes a card:
// chart on top, caption under it. Consecutive charts share one grid instead of stacking.
export function rehypeChartGrid() {
  return (tree: Root) => {
    const out: RootContent[] = [];
    let grid: Element | null = null;
    for (const node of tree.children) {
      const img =
        node.type === "element" && node.tagName === "p"
          ? node.children.find((c): c is Element => c.type === "element" && c.tagName === "img")
          : undefined;
      if (img && node.type === "element") {
        const caption = node.children.filter((c) => c !== img);
        const last = caption.at(-1);
        if (last?.type === "text") last.value = last.value.trimEnd().replace(/:$/, "");
        const hasCaption = caption.some((c) => c.type !== "text" || c.value.trim());
        const figcaption: Element = { type: "element", tagName: "figcaption", properties: {}, children: caption };
        if (!grid) {
          grid = { type: "element", tagName: "div", properties: { className: ["chart-grid"] }, children: [] };
          out.push(grid);
        }
        grid.children.push({
          type: "element",
          tagName: "figure",
          properties: { className: ["chart-card"] },
          children: hasCaption ? [img, figcaption] : [img],
        });
      } else if (!(grid && node.type === "text" && !node.value.trim())) {
        grid = null; // anything but the blank line between two charts ends the grid
        out.push(node);
      }
    }
    tree.children = out;
  };
}
