"use strict";

const $ = (id) => document.getElementById(id);
const form = $("form");
const promptBox = $("prompt");
const submit = $("submit");
const errorBox = $("error");
const temperature = $("temperature");

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function optionalInt(input) {
  const raw = input.value.trim();
  return raw === "" ? null : Number(raw);
}

temperature.addEventListener("input", () => {
  $("temperature_value").textContent = Number(temperature.value).toFixed(2);
});

for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    promptBox.value = button.textContent;
    promptBox.focus();
  });
}

promptBox.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  if (!promptBox.value.trim()) {
    showError("Write the beginning of a story first.");
    return;
  }
  const body = {
    prompt: promptBox.value,
    max_new_tokens: Number($("max_new_tokens").value),
    temperature: Number(temperature.value),
    top_k: optionalInt($("top_k")),
    seed: optionalInt($("seed")),
  };

  submit.disabled = true;
  submit.textContent = "Writing…";
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      showError(data.error || `The server returned an error (${response.status}).`);
      return;
    }
    $("out_prompt").textContent = data.prompt;
    $("out_completion").textContent = data.completion;
    const stop = data.stop_reason === "eos"
      ? "stopped at the end of the story"
      : `stopped at the ${data.settings.max_new_tokens}-token limit`;
    const parts = [
      `${data.generated_tokens} tokens`,
      stop,
      `seed ${data.seed}`,
      `${(data.elapsed_ms / 1000).toFixed(2)} s`,
    ];
    if (data.prompt_unknown_tokens > 0) {
      parts.push(`${data.prompt_unknown_tokens} character(s) in your text were never seen in training`);
    }
    $("meta").textContent = parts.join(" · ");
    $("result").hidden = false;
  } catch (err) {
    showError("Could not reach the Eclipse server. Is it still running?");
  } finally {
    submit.disabled = false;
    submit.textContent = "Continue the story";
  }
});

async function loadInfo() {
  const footer = $("model_info");
  try {
    const response = await fetch("/api/info");
    const info = await response.json();
    const a = info.architecture;
    const provenance = {
      verified: "weights verified against the training run's recorded hash",
      unverifiable: "weights not verifiable against the research record",
      unavailable: "no provenance record available for these weights",
      skipped: "provenance check skipped",
    }[info.provenance.status] || info.provenance.status;
    footer.textContent =
      `Model: ${info.checkpoint} · step ${info.step == null ? "unknown" : info.step.toLocaleString()} · ` +
      `${info.parameters.toLocaleString()} parameters · ` +
      `${a.num_blocks} blocks, d_model ${a.d_model}, ${a.num_heads} heads · ` +
      `${a.context_length}-token context · ${a.vocab_size.toLocaleString()}-token BPE vocabulary · ` +
      `running on ${info.device} · ${provenance}. ` +
      "Everything runs locally; no external model or API is used.";
    promptBox.maxLength = info.limits.max_prompt_chars;
    $("max_new_tokens").max = info.limits.max_new_tokens;
    $("temperature").max = info.limits.max_temperature;
    $("top_k").max = a.vocab_size;
  } catch (err) {
    footer.textContent = "Could not load model information.";
  }
}

loadInfo();
