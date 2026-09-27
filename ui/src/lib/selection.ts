import { useState } from "react";

/** Checkbox selection over a list of ids; ids that disappear drop out of it. */
export function useSelection(ids: string[]) {
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const selected = ids.filter((id) => picked.has(id));
  const allSelected = ids.length > 0 && selected.length === ids.length;
  return {
    selected,
    isSelected: (id: string) => picked.has(id),
    toggle: (id: string) =>
      setPicked((prev) => {
        const next = new Set(prev);
        if (!next.delete(id)) next.add(id);
        return next;
      }),
    allSelected,
    toggleAll: () => setPicked(allSelected ? new Set() : new Set(ids)),
    clear: () => setPicked(new Set()),
  };
}

/** One at a time (the registry's file store serializes writes anyway); returns what failed and why. */
export async function deleteEach(ids: string[], del: (id: string) => Promise<unknown>): Promise<string[]> {
  const failures: string[] = [];
  for (const id of ids) {
    try {
      await del(id);
    } catch (e) {
      failures.push(e instanceof Error ? e.message : String(e));
    }
  }
  return failures;
}
