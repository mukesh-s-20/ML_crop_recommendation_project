/* ── predict.js ─────────────────────────────────── */

// ── Sync sliders <-> text inputs ─────────────────────
const pairs = [
  ["sl_n","inp_n"], ["sl_p","inp_p"], ["sl_k","inp_k"], ["sl_t","inp_t"]
];
pairs.forEach(([slId, inpId]) => {
  const sl = document.getElementById(slId);
  const inp = document.getElementById(inpId);
  if (!sl || !inp) return;
  inp.value = sl.value;
  sl.addEventListener("input", () => { inp.value = sl.value; });
  inp.addEventListener("input", () => {
    const v = parseFloat(inp.value);
    if (!isNaN(v)) sl.value = v;
  });
});

// ── Table row click ───────────────────────────────────
document.querySelectorAll("tr[data-example]").forEach(row => {
  row.addEventListener("click", () => {
    const vals = row.dataset.example.split(",").map(Number);
    const mapping = [
      ["inp_n","sl_n"], ["inp_p","sl_p"], ["inp_k","sl_k"], ["inp_t","sl_t"]
    ];
    mapping.forEach(([inpId, slId], i) => {
      const inp = document.getElementById(inpId);
      const sl  = document.getElementById(slId);
      if (inp) inp.value = vals[i];
      if (sl)  sl.value  = vals[i];
    });
    document.querySelectorAll("tr[data-example]").forEach(r => r.classList.remove("active-row"));
    row.classList.add("active-row");
  });
});

// ── Fill Example ──────────────────────────────────────
function fillExample() {
  const ex = { inp_n:90, inp_p:42, inp_k:43, inp_t:20.9 };
  const sm = { inp_n:"sl_n", inp_p:"sl_p", inp_k:"sl_k", inp_t:"sl_t" };
  Object.entries(ex).forEach(([id, val]) => {
    const el = document.getElementById(id); if (el) el.value = val;
    const sl = document.getElementById(sm[id]); if (sl) sl.value = val;
  });
}

// ── Clear Form ────────────────────────────────────────
function clearForm() {
  ["inp_n","inp_p","inp_k","inp_t"].forEach(id => {
    const el = document.getElementById(id); if (el) el.value = "";
  });
  document.getElementById("resultContent").style.display = "none";
  document.getElementById("resultPlaceholder").style.display = "block";
}

// ── Validate & Get Inputs ─────────────────────────────
function getInputs() {
  const fields = [
    { id:"inp_n",  label:"Nitrogen",    min:0,  max:140 },
    { id:"inp_p",  label:"Phosphorus",  min:5,  max:145 },
    { id:"inp_k",  label:"Potassium",   min:5,  max:205 },
    { id:"inp_t",  label:"Temperature", min:8,  max:44  },
  ];
  const vals = {};
  for (const f of fields) {
    const el = document.getElementById(f.id);
    const val = parseFloat(el.value);
    if (isNaN(val) || val < f.min || val > f.max) {
      alert(`⚠️ Please enter a valid ${f.label} (${f.min}–${f.max})`);
      el.focus(); return null;
    }
    vals[f.id] = val;
  }
  return vals;
}

// ── Run Prediction ────────────────────────────────────
async function runPrediction() {
  const vals = getInputs();
  if (!vals) return;

  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analysing…';

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nitrogen:    vals.inp_n,
        phosphorus:  vals.inp_p,
        potassium:   vals.inp_k,
        temperature: vals.inp_t,
      })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Prediction failed");
    showResult(data);
  } catch (err) {
    alert("❌ Error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "✨ Predict Crop";
  }
}

// ── Render Result ─────────────────────────────────────
function showResult(data) {
  document.getElementById("resultPlaceholder").style.display = "none";
  const rc = document.getElementById("resultContent");
  rc.style.display = "block";

  document.getElementById("reEmoji").textContent   = data.info.emoji || "🌾";
  document.getElementById("reCrop").textContent    = data.crop;
  document.getElementById("reConf").textContent    = `${data.confidence}% Confidence`;
  document.getElementById("reModel").textContent   = data.model;
  document.getElementById("reMlpCrop").textContent = data.mlp_crop;
  document.getElementById("reMlpConf").textContent = `${data.mlp_conf}%`;

  const info = data.info;
  document.getElementById("reInfo").innerHTML = `
    <strong>${info.emoji} ${data.crop}</strong><br/>
    ${info.desc}<br/>
    <span style="margin-top:.4rem;display:inline-flex;gap:.8rem;font-size:.8rem;">
      <span>📅 Season: <strong>${info.season}</strong></span>
      <span>💧 Water: <strong>${info.water}</strong></span>
    </span>
  `;

  document.getElementById("reTop3").innerHTML = data.top3.map((t, i) =>
    `<div class="top3-item">
      <span>${i + 1}.</span><span>${t.emoji}</span>
      <span>${t.crop}</span>
      <span class="top3-conf">${t.confidence}%</span>
    </div>`
  ).join("");

  document.getElementById("probBars").innerHTML = data.top3.map(t =>
    `<div class="prob-bar-item">
      <div class="pb-label"><span>${t.emoji} ${t.crop}</span><span>${t.confidence}%</span></div>
      <div class="pb-track"><div class="pb-fill" style="width:${t.confidence}%"></div></div>
    </div>`
  ).join("");

  rc.scrollIntoView({ behavior: "smooth", block: "start" });
}

document.addEventListener("keydown", e => {
  if (e.key === "Enter" && document.activeElement.classList.contains("form-input")) {
    runPrediction();
  }
});
