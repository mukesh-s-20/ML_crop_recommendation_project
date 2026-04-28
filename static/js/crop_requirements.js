/* ═══════════════════════════════════════════════════════
   crop_requirements.js
   Method 2 – Choose Crop → Show Soil Requirements
   ═══════════════════════════════════════════════════════ */

// ── Crop meta-info ────────────────────────────────────────
const CROP_INFO = {
  Apple:       { emoji:"🍎", desc:"Temperate fruit crop requiring cold winters and mild summers with good drainage.", season:"Perennial", water:"Moderate" },
  Banana:      { emoji:"🍌", desc:"Tropical fruit requiring high humidity, warm temperatures and well-drained loamy soil.", season:"Perennial", water:"High" },
  Blackgram:   { emoji:"🫘", desc:"Warm-season pulse crop rich in protein, grows well in tropical and sub-tropical regions.", season:"Kharif/Rabi", water:"Moderate" },
  ChickPea:    { emoji:"🫘", desc:"Cool-season legume that thrives in semi-arid conditions with low humidity.", season:"Rabi", water:"Low" },
  Coconut:     { emoji:"🥥", desc:"Coastal tropical crop requiring high humidity, warm climate and sandy loam soil.", season:"Perennial", water:"High" },
  Coffee:      { emoji:"☕", desc:"Grows best in tropical highlands with moderate temperature and high humidity.", season:"Perennial", water:"Moderate" },
  Cotton:      { emoji:"☁️", desc:"Fibre crop needing long warm seasons, moderate rainfall and deep black soil.", season:"Kharif", water:"Moderate" },
  Grapes:      { emoji:"🍇", desc:"Grown in temperate to subtropical climates; requires dry summers and well-drained soil.", season:"Perennial", water:"Low" },
  Jute:        { emoji:"🌿", desc:"Bast fibre crop requiring hot humid climate with well-distributed rainfall.", season:"Kharif", water:"High" },
  KidneyBeans: { emoji:"🫘", desc:"Warm-season legume requiring moderate moisture and fertile, well-drained soil.", season:"Kharif", water:"Moderate" },
  Lentil:      { emoji:"🫘", desc:"Cool-season pulse; drought-tolerant once established. Fixes nitrogen in soil.", season:"Rabi", water:"Low" },
  Maize:       { emoji:"🌽", desc:"Versatile cereal crop needing warm weather, good drainage and moderate rainfall.", season:"Kharif/Rabi", water:"Moderate" },
  Mango:       { emoji:"🥭", desc:"Tropical fruit tree preferring a well-defined dry season and deep fertile soil.", season:"Perennial", water:"Low" },
  MothBeans:   { emoji:"🌱", desc:"Extremely drought-resistant legume, best suited for hot arid and semi-arid regions.", season:"Kharif", water:"Very Low" },
  MungBean:    { emoji:"🫘", desc:"Fast-growing legume adapted to warm and humid conditions with short crop duration.", season:"Kharif", water:"Moderate" },
  Muskmelon:   { emoji:"🍈", desc:"Warm-season cucurbit requiring sandy loam soil, warm days and good sunshine.", season:"Kharif", water:"Moderate" },
  Orange:      { emoji:"🍊", desc:"Subtropical citrus requiring warm days, cool nights and well-drained fertile soil.", season:"Perennial", water:"Moderate" },
  Papaya:      { emoji:"🍈", desc:"Fast-growing tropical fruit sensitive to waterlogging and frost.", season:"Perennial", water:"Moderate" },
  PigeonPeas:  { emoji:"🌿", desc:"Drought-tolerant legume that thrives in warm tropical climates and poor soils.", season:"Kharif", water:"Low" },
  Pomegranate: { emoji:"🍎", desc:"Drought-tolerant fruit crop suited to hot dry summers and mild winters.", season:"Perennial", water:"Low" },
  Rice:        { emoji:"🌾", desc:"Staple cereal requiring high humidity, warm temperatures and standing water.", season:"Kharif", water:"High" },
  Watermelon:  { emoji:"🍉", desc:"Warm-season vine crop needing sandy loam soil, abundant sunshine and heat.", season:"Kharif", water:"Moderate" },
};

const PARAM_META = {
  Nitrogen:    { unit:"kg/ha", icon:"🧪", label:"Nitrogen (N)",    desc:"Primary nutrient for leaf and stem growth" },
  Phosphorus:  { unit:"kg/ha", icon:"🧪", label:"Phosphorus (P)", desc:"Essential for root development and flowering" },
  Potassium:   { unit:"kg/ha", icon:"🧪", label:"Potassium (K)",  desc:"Regulates water uptake and disease resistance" },
  Temperature: { unit:"°C",    icon:"🌡️", label:"Temperature",    desc:"Optimal growing temperature range" },
  Humidity:    { unit:"%",     icon:"💧", label:"Humidity",       desc:"Relative air moisture required" },
  pH_Value:    { unit:"pH",    icon:"⚗️", label:"pH Value",       desc:"Soil acidity/alkalinity level" },
  Rainfall:    { unit:"mm",    icon:"🌧️", label:"Rainfall",       desc:"Annual rainfall or irrigation equivalent" },
};

const PARAM_ORDER = ["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","pH_Value","Rainfall"];

// ── State ─────────────────────────────────────────────────
let currentCropData   = null;
let currentCropName   = null;

// ── Tab Switching ─────────────────────────────────────────
function switchTab(n) {
  document.getElementById("method1Panel").style.display = n === 1 ? "block" : "none";
  document.getElementById("method2Panel").style.display = n === 2 ? "block" : "none";
  document.getElementById("tab1").classList.toggle("active", n === 1);
  document.getElementById("tab2").classList.toggle("active", n === 2);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ── Filter crop buttons ────────────────────────────────────
function filterCrops(query) {
  const q = query.toLowerCase().trim();
  document.querySelectorAll(".crop-pick-btn").forEach(btn => {
    const name = btn.dataset.crop.toLowerCase();
    btn.style.display = !q || name.includes(q) ? "" : "none";
  });
}

// ── Select crop ───────────────────────────────────────────
function selectCrop(cropName, emoji) {
  // Highlight selected
  document.querySelectorAll(".crop-pick-btn").forEach(b => b.classList.remove("selected"));
  const btn = document.querySelector(`.crop-pick-btn[data-crop="${cropName}"]`);
  if (btn) btn.classList.add("selected");

  currentCropName = cropName;
  fetchCropRequirements(cropName, emoji || (CROP_INFO[cropName] ? CROP_INFO[cropName].emoji : "🌱"));
}

// ── Fetch from API ────────────────────────────────────────
async function fetchCropRequirements(cropName, emoji) {
  try {
    const res  = await fetch(`/api/crop_requirements/${encodeURIComponent(cropName)}`);
    const data = await res.json();
    if (!data.success) throw new Error(data.error);
    currentCropData = data;
    renderCropRequirements(cropName, emoji, data);
  } catch (e) {
    alert("Failed to load crop data: " + e.message);
  }
}

// ── Render soil requirements ──────────────────────────────
function renderCropRequirements(cropName, emoji, data) {
  document.getElementById("soilPlaceholder").style.display  = "none";
  document.getElementById("soilReqContent").style.display   = "block";

  const info = CROP_INFO[cropName] || { desc:"—", season:"—", water:"—", emoji };

  document.getElementById("soilEmoji").textContent    = info.emoji || emoji;
  document.getElementById("soilCropName").textContent = cropName;
  document.getElementById("soilCropDesc").textContent = info.desc;
  document.getElementById("soilSeason").textContent   = "📅 " + info.season;
  document.getElementById("soilWater").textContent    = "💧 Water: " + info.water;

  // Parameter cards
  const grid = document.getElementById("soilParamsGrid");
  grid.innerHTML = "";
  PARAM_ORDER.forEach(param => {
    const s    = data.requirements[param];
    const m    = PARAM_META[param];
    const pct  = (val, lo, hi) => Math.max(0, Math.min(100, ((val - lo) / (hi - lo)) * 100));
    const lo   = s.min, hi = s.max;

    const card = document.createElement("div");
    card.className = "soil-param-card";
    card.innerHTML = `
      <div class="spc-header">
        <span class="spc-icon">${m.icon}</span>
        <div>
          <div class="spc-label">${m.label}</div>
          <div class="spc-desc">${m.desc}</div>
        </div>
        <span class="spc-unit">${m.unit}</span>
      </div>
      <div class="spc-range-visual">
        <div class="spc-track">
          <!-- Acceptable band (min–max) -->
          <div class="spc-band acceptable"
               style="left:0%;width:100%"
               title="Acceptable: ${s.min}–${s.max}"></div>
          <!-- Ideal band (Q1–Q3) -->
          <div class="spc-band ideal"
               style="left:${pct(s.q25,lo,hi)}%;width:${pct(s.q75,lo,hi)-pct(s.q25,lo,hi)}%"
               title="Ideal: ${s.q25}–${s.q75}"></div>
          <!-- Mean marker -->
          <div class="spc-mean-marker" style="left:${pct(s.mean,lo,hi)}%"
               title="Mean: ${s.mean}"></div>
        </div>
        <div class="spc-labels">
          <span>${s.min}</span>
          <span class="spc-ideal-label">Ideal: ${s.q25} – ${s.q75}</span>
          <span>${s.max}</span>
        </div>
      </div>
      <div class="spc-values">
        <div class="spc-val-item">
          <span class="spc-dot amber"></span>
          <span class="spc-val-label">Min</span>
          <strong>${s.min} ${m.unit}</strong>
        </div>
        <div class="spc-val-item">
          <span class="spc-dot green"></span>
          <span class="spc-val-label">Ideal Low (Q1)</span>
          <strong>${s.q25} ${m.unit}</strong>
        </div>
        <div class="spc-val-item">
          <span class="spc-dot blue"></span>
          <span class="spc-val-label">Mean</span>
          <strong>${s.mean} ${m.unit}</strong>
        </div>
        <div class="spc-val-item">
          <span class="spc-dot green"></span>
          <span class="spc-val-label">Ideal High (Q3)</span>
          <strong>${s.q75} ${m.unit}</strong>
        </div>
        <div class="spc-val-item">
          <span class="spc-dot amber"></span>
          <span class="spc-val-label">Max</span>
          <strong>${s.max} ${m.unit}</strong>
        </div>
      </div>`;
    grid.appendChild(card);
  });

  // Reference table
  const tbody = document.getElementById("soilRefBody");
  tbody.innerHTML = "";
  PARAM_ORDER.forEach(param => {
    const s = data.requirements[param];
    const m = PARAM_META[param];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${m.icon} ${m.label}</td>
      <td>${s.min}</td>
      <td class="ideal-cell">${s.q25}</td>
      <td><strong>${s.mean}</strong></td>
      <td class="ideal-cell">${s.q75}</td>
      <td>${s.max}</td>
      <td style="color:var(--text-muted);font-size:.8rem">${m.unit}</td>`;
    tbody.appendChild(tr);
  });

  // Scroll to panel
  document.getElementById("soilReqCard").scrollIntoView({ behavior:"smooth", block:"start" });
}

// ── Pre-fill Method 1 from ideal (mean) values ────────────
function prefillFromCrop() {
  if (!currentCropData) return;
  const r = currentCropData.requirements;
  const map = {
    inp_n: r.Nitrogen.mean,    sl_n: r.Nitrogen.mean,
    inp_p: r.Phosphorus.mean,  sl_p: r.Phosphorus.mean,
    inp_k: r.Potassium.mean,   sl_k: r.Potassium.mean,
    inp_t: r.Temperature.mean, sl_t: r.Temperature.mean,
    inp_h: r.Humidity.mean,    sl_h: r.Humidity.mean,
    inp_ph:r.pH_Value.mean,    sl_ph:r.pH_Value.mean,
    inp_r: r.Rainfall.mean,    sl_r: r.Rainfall.mean,
  };
  Object.entries(map).forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el) el.value = val;
  });
  switchTab(1);
  // Auto-trigger prediction after fill
  setTimeout(runPrediction, 400);
}

// ── "View Soil Requirements" button from Method 1 result ──
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("viewSoilBtn");
  if (btn) {
    btn.addEventListener("click", () => {
      const cropName = document.getElementById("reCrop").textContent.trim();
      if (!cropName || cropName === "—") return;
      const info = CROP_INFO[cropName];
      switchTab(2);
      selectCrop(cropName, info ? info.emoji : "🌱");
    });
  }
});
