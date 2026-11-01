async function loadTurn() {
  const res = await fetch("/api/turn");
  if (!res.ok) {
    throw new Error(`Failed to load turn: ${res.status}`);
  }
  return res.json();
}

function chip(text, cls = "") {
  const el = document.createElement("span");
  el.className = `chip ${cls}`.trim();
  el.textContent = text;
  return el;
}

function render(payload) {
  const turn = payload.turn;
  const diff = turn.diff_summary || {};
  const meta = document.getElementById("meta");
  meta.textContent = [
    payload.path,
    `vendor=${turn.vendor}`,
    `visibility=${turn.visibility}`,
    turn.model ? `model=${turn.model}` : null,
  ]
    .filter(Boolean)
    .join("\n");

  document.getElementById("ui-text").textContent = turn.user_utterance_raw || "(empty)";

  const bar = document.getElementById("diff-bar");
  bar.innerHTML = "";
  bar.appendChild(
    chip(
      diff.utterance_unchanged ? "utterance kept" : "utterance not found verbatim",
      diff.utterance_unchanged ? "ok" : "accent",
    ),
  );
  if (diff.utterance_wrapped) {
    bar.appendChild(chip("wrapped in tags", "accent"));
  }
  bar.appendChild(chip(`${diff.bytes_ui || 0} B UI`));
  bar.appendChild(chip(`${diff.bytes_model || 0} B model input`, "accent"));
  bar.appendChild(chip(`${(diff.added_layers || []).length} added layers`));

  const layers = document.getElementById("layers");
  layers.innerHTML = "";
  for (const layer of turn.layers || []) {
    const details = document.createElement("details");
    details.className = `layer ${layer.kind}`;
    details.open =
      layer.kind === "user_utterance" ||
      layer.kind === "system" ||
      layer.kind === "project_memory" ||
      layer.kind === "rules_or_instructions";

    const summary = document.createElement("summary");
    const left = document.createElement("span");
    left.innerHTML = `<span class="kind">${layer.kind}</span> · ${layer.name}`;
    const right = document.createElement("span");
    right.textContent = `${layer.bytes} B`;
    summary.append(left, right);

    const pre = document.createElement("pre");
    pre.textContent = layer.text;

    details.append(summary, pre);
    layers.appendChild(details);
  }
}

loadTurn()
  .then(render)
  .catch((err) => {
    document.getElementById("ui-text").textContent = String(err);
  });
