export function SelectionBar({
  what,
  total,
  count,
  allSelected,
  onToggleAll,
  onDelete,
  busy,
}: {
  what: string;
  total: number;
  count: number;
  allSelected: boolean;
  onToggleAll: () => void;
  onDelete: () => void;
  busy: boolean;
}) {
  if (total === 0) return null;
  return (
    <div className="selection-bar">
      <label>
        <input type="checkbox" className="check" checked={allSelected} onChange={onToggleAll} />
        Select all
      </label>
      <button type="button" className="danger small" disabled={count === 0 || busy} onClick={onDelete}>
        {busy ? "Deleting…" : count ? `Delete ${count} ${what}` : `Delete ${what}`}
      </button>
    </div>
  );
}
