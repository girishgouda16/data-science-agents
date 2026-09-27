import { expect, test } from "vitest";
import { fileLabel, linkServerPaths } from "./files";

test("server paths become names, the user's own files become downloads", () => {
  const upload = "/srv/aml/data/uploads/alice/9298d38e-ab75-475d-a3ca-ec0d8155f089_housing_predictions.csv";
  expect(fileLabel(upload)).toBe("housing_predictions.csv");
  expect(linkServerPaths(`Output CSV: \`${upload}\`.`)).toBe(
    `Output CSV: [📄 housing_predictions.csv](#download=${encodeURIComponent(upload)}).`);
  expect(linkServerPaths("log: /srv/aml/data/artifacts/model-v1.pkl")).toBe("log: `model-v1.pkl`");
  expect(linkServerPaths("![chart](/charts/x.png) and https://example.com/data/a.csv")).toBe(
    "![chart](/charts/x.png) and https://example.com/data/a.csv");
});
