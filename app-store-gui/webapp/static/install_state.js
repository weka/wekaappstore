/*
 * Shared blueprint install-state controller.
 *
 * Included by every blueprint page. Reads window.WARP_INSTALL = {crName, appName,
 * defaultNamespace}. When the blueprint's WekaAppStore CR already exists it:
 *   - shows a status badge near the page title ("Installed" in WEKA purple, or
 *     "Install failed" in red, or "Installing…" in amber),
 *   - populates the form fields with the originally-submitted variables and locks them,
 *   - hides Deploy and shows an Uninstall button that deletes the CR via the operator.
 * When the CR does not exist it leaves the normal Deploy UI untouched.
 */
(function () {
  "use strict";
  var cfg = window.WARP_INSTALL || {};
  if (!cfg.crName) return; // blueprint has no WekaAppStore CR mapping — nothing to manage

  var form = document.getElementById("deploy-form");
  if (!form) return;

  var PURPLE = "background:rgba(107,47,179,0.20);border:1px solid rgba(107,47,179,0.55);color:#d7c2f0;";
  var RED = "background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.45);color:#fca5a5;";
  var AMBER = "background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.45);color:#fde68a;";

  function deployBtn() {
    return document.getElementById("deploy-btn") || form.querySelector('button[type="submit"]');
  }
  function nsEl() {
    return document.getElementById("field-namespace");
  }
  function resEl() {
    return document.getElementById("deploy-result");
  }
  function fieldNamespace() {
    var el = nsEl();
    return (el && el.value.trim()) || cfg.defaultNamespace || "default";
  }
  // Namespace where the CR was actually found installed (used for Uninstall).
  var installedNs = null;

  var badge = null;
  function ensureBadge() {
    if (badge) return badge;
    badge = document.getElementById("install-status-badge");
    if (badge) return badge;
    badge = document.createElement("span");
    badge.id = "install-status-badge";
    badge.className = "badge";
    badge.style.marginLeft = "6px";
    var existing = document.querySelector("main .badge") || document.querySelector(".badge");
    var h1 = document.querySelector("main h1") || document.querySelector("h1");
    if (existing && existing.parentNode) existing.parentNode.appendChild(badge);
    else if (h1) h1.insertAdjacentElement("afterend", badge);
    else form.insertBefore(badge, form.firstChild);
    return badge;
  }

  var uninstallBtn = null;
  function ensureUninstallBtn() {
    if (uninstallBtn) return uninstallBtn;
    uninstallBtn = document.createElement("button");
    uninstallBtn.type = "button";
    uninstallBtn.id = "uninstall-btn";
    uninstallBtn.className = "w-full px-4 py-2 rounded-md text-sm font-medium";
    uninstallBtn.style.cssText = "background:#b91c1c;color:#fff;";
    uninstallBtn.textContent = "Uninstall";
    uninstallBtn.addEventListener("click", onUninstall);
    var db = deployBtn();
    if (db && db.parentNode) db.insertAdjacentElement("afterend", uninstallBtn);
    else form.appendChild(uninstallBtn);
    return uninstallBtn;
  }

  function setFieldsDisabled(disabled) {
    form.querySelectorAll("input, select, textarea").forEach(function (el) {
      el.disabled = disabled;
    });
  }

  function fillFields(vars) {
    vars = vars || {};
    Object.keys(vars).forEach(function (k) {
      var el = form.querySelector('[name="' + (window.CSS && CSS.escape ? CSS.escape(k) : k) + '"]') ||
               document.getElementById("field-" + k);
      if (!el) return;
      var val = vars[k] == null ? "" : String(vars[k]);
      if (el.tagName === "SELECT") {
        var match = Array.prototype.find.call(el.options, function (o) { return o.value === val; });
        if (!match && val) el.add(new Option(val, val));
        el.value = val;
      } else {
        el.value = val;
      }
    });
  }

  function showInstalled(phase, vars) {
    var p = (phase || "").toLowerCase();
    var b = ensureBadge();
    if (p === "failed" || p === "error") {
      b.setAttribute("style", RED + "margin-left:6px;");
      b.textContent = "Install failed";
    } else if (p === "ready" || p === "healthy" || p === "deployed") {
      b.setAttribute("style", PURPLE + "margin-left:6px;");
      b.textContent = "Installed";
    } else {
      b.setAttribute("style", AMBER + "margin-left:6px;");
      b.textContent = "Installing…";
    }
    b.style.display = "";
    fillFields(vars);
    setFieldsDisabled(true);
    var db = deployBtn();
    if (db) db.style.display = "none";
    ensureUninstallBtn().style.display = "";
  }

  function showNotInstalled() {
    if (badge) badge.style.display = "none";
    setFieldsDisabled(false);
    var db = deployBtn();
    if (db) db.style.display = "";
    if (uninstallBtn) uninstallBtn.style.display = "none";
  }

  function refresh(ns) {
    ns = ns || fieldNamespace();
    return fetch("/api/wekaappstore-exists?name=" + encodeURIComponent(cfg.crName) +
                 "&namespace=" + encodeURIComponent(ns))
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j && j.ok && j.exists) { installedNs = ns; showInstalled(j.phase, j.variables); }
        else { installedNs = null; showNotInstalled(); }
      })
      .catch(function () { /* leave Deploy UI as-is on transient errors */ });
  }

  function onUninstall() {
    var ns = installedNs || fieldNamespace();
    if (!window.confirm('Uninstall "' + cfg.crName + '" from namespace "' + ns +
                        '"? This deletes the deployment via the operator.')) return;
    uninstallBtn.disabled = true;
    uninstallBtn.textContent = "Uninstalling…";
    var res = resEl();
    if (res) res.textContent = "Uninstalling…";
    fetch("/api/blueprints/" + encodeURIComponent(ns) + "/" + encodeURIComponent(cfg.crName),
          { method: "DELETE" })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        uninstallBtn.disabled = false;
        uninstallBtn.textContent = "Uninstall";
        if (j && j.ok) {
          if (res) res.textContent = "Uninstalled.";
          showNotInstalled();
          setTimeout(function () { refresh(ns); }, 1500);
        } else if (res) {
          res.textContent = "Uninstall failed: " + ((j && j.error) || "unknown error");
        }
      })
      .catch(function (e) {
        uninstallBtn.disabled = false;
        uninstallBtn.textContent = "Uninstall";
        if (res) res.textContent = "Uninstall error: " + e;
      });
  }

  // After a deploy is submitted, poll briefly so the page flips to the installed
  // state without a manual reload.
  function startPostDeployPolling() {
    var ns = fieldNamespace(); // namespace the deploy was just submitted against
    var ticks = 0;
    var iv = setInterval(function () {
      ticks += 1;
      if (ticks > 45) { clearInterval(iv); return; } // ~3 min cap
      refresh(ns);
    }, 4000);
  }

  function init() {
    // Initial check uses the blueprint's canonical namespace; the change handler
    // re-checks whatever namespace the user selects.
    refresh(cfg.defaultNamespace || fieldNamespace());
    var el = nsEl();
    if (el) el.addEventListener("change", function () { refresh(fieldNamespace()); });
    form.addEventListener("submit", function () { setTimeout(startPostDeployPolling, 1000); });
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
