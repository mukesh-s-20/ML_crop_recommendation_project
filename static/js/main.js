/* ── main.js ─────────────────────────────────────── */

// Mobile nav toggle
const navToggle = document.getElementById("navToggle");
const navLinks  = document.querySelector(".nav-links");
if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    navLinks.classList.toggle("open");
    navLinks.style.display = navLinks.classList.contains("open") ? "flex" : "";
  });
}

// Example table row click (predict page)
document.querySelectorAll("tr[data-example]").forEach(row => {
  row.addEventListener("click", () => {
    const vals = row.dataset.example.split(",").map(Number);
    const ids  = ["inp_n","inp_p","inp_k","inp_t","inp_h","inp_ph","inp_r"];
    const slds = ["sl_n","sl_p","sl_k","sl_t","sl_h","sl_ph","sl_r"];
    ids.forEach((id, i) => {
      const el = document.getElementById(id);
      if (el) el.value = vals[i];
    });
    slds.forEach((id, i) => {
      const el = document.getElementById(id);
      if (el) el.value = vals[i];
    });
    // Highlight
    document.querySelectorAll("tr[data-example]").forEach(r => r.classList.remove("active-row"));
    row.classList.add("active-row");
  });
});

// Smooth reveal on scroll
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity = "1";
      e.target.style.transform = "translateY(0)";
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll(".feat-card, .step, .dl-metric-card").forEach(el => {
  el.style.opacity = "0";
  el.style.transform = "translateY(20px)";
  el.style.transition = "opacity .4s ease, transform .4s ease";
  observer.observe(el);
});
