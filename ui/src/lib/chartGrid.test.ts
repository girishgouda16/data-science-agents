import type { Element, ElementContent, Root } from "hast";
import { expect, test } from "vitest";
import { rehypeChartGrid } from "./chartGrid";

const el = (tagName: string, children: ElementContent[] = [], properties = {}): Element =>
  ({ type: "element", tagName, properties, children });
const text = (value: string) => ({ type: "text" as const, value });
const chart = (title: string) =>
  el("p", [el("strong", [text(title)]), text(" — caption:\n"), el("img", [], { src: `/charts/${title}.png` })]);

test("consecutive chart paragraphs become cards in one grid; other blocks are untouched", () => {
  const tree: Root = { type: "root", children: [chart("ROC"), text("\n"), chart("PR"), text("\n"), el("p", [text("Read.")])] };
  rehypeChartGrid()(tree);

  const [grid, after] = tree.children as Element[];
  expect(grid.properties.className).toEqual(["chart-grid"]);
  expect(grid.children.map((c) => (c as Element).tagName)).toEqual(["figure", "figure"]);
  const [img, caption] = (grid.children[0] as Element).children as Element[];
  expect(img.tagName).toBe("img"); // chart first, caption under it
  expect(caption.tagName).toBe("figcaption");
  expect(caption.children.at(-1)).toEqual(text(" — caption")); // trailing ":" dropped
  expect(after.tagName).toBe("p");
});

test("a paragraph without an image stays a paragraph", () => {
  const tree: Root = { type: "root", children: [el("p", [text("No chart here:")])] };
  rehypeChartGrid()(tree);
  expect((tree.children[0] as Element).tagName).toBe("p");
});
