import { accessToken, apiBase } from "./auth";

// Server paths mean nothing to a user; a file's own name does. Uploads are
// stored as <uuid>_<name>, so the uuid comes off too.
export function fileLabel(path: string): string {
  return (path.split("/").pop() ?? path).replace(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_/, "");
}

// Absolute paths into the data folder, with the backticks agents wrap them in —
// starting where a path can start, so a URL's ".../data/x.csv" is left alone.
const DATA_PATH = /`?(?<![\w:/.])(\/[^\s`'"()<>]+\/data\/[^\s`'"()<>]+\.[A-Za-z0-9]+)`?/g;

/** Agent markdown with server paths replaced: the user's own files (uploads,
 * predictions) become download links, anything else just its name. */
export function linkServerPaths(markdown: string): string {
  return markdown.replace(DATA_PATH, (_match, path: string) =>
    /\/data\/(uploads|predictions)\//.test(path)
      ? `[📄 ${fileLabel(path)}](#download=${encodeURIComponent(path)})`
      : `\`${fileLabel(path)}\``,
  );
}

/** Download one of the user's files (a bearer token can't ride a plain <a href>). */
export async function saveFile(path: string): Promise<void> {
  const res = await fetch(`${apiBase()}/api/v1/download?path=${encodeURIComponent(path)}`, {
    headers: { Authorization: `Bearer ${await accessToken()}` },
  });
  if (!res.ok) throw new Error(res.status === 404 ? "That file is gone or isn't yours." : `${res.status}`);
  const url = URL.createObjectURL(await res.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: fileLabel(path) });
  a.click();
  URL.revokeObjectURL(url);
}
