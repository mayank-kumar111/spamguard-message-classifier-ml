// Front-end for SpamGuard: theme toggle, background particles, the live
// prediction gauge, batch upload and the Chart.js charts.
(function () {
    "use strict";

    const $ = (s, r = document) => r.querySelector(s);
    const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function cssVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }
    function hexToRgba(hex, a) {
        hex = hex.replace("#", "");
        if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
        const n = parseInt(hex, 16);
        return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
    }
    function escapeHtml(s) {
        return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
            .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }
    function debounce(fn, ms) {
        let t;
        return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
    }
    async function apiPredict(text) {
        const r = await fetch("/api/predict", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        return r.json();
    }
    // green -> amber -> red interpolation for a spam probability in [0,1]
    function riskColor(p) {
        const stops = [[52, 211, 153], [251, 191, 36], [251, 83, 120]];
        const seg = p < 0.5 ? 0 : 1;
        const t = p < 0.5 ? p / 0.5 : (p - 0.5) / 0.5;
        const a = stops[seg], b = stops[seg + 1];
        const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
        return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
    }

    // Build a chart but don't let one bad chart take down the rest of the page.
    function safeNew(el, cfg) {
        var C = window.Chart;
        try { return new C(el, cfg); }
        catch (e) { console.error("[chart] failed to render:", e); return null; }
    }

    // Keeps a record of what was classified this session (single scans and
    // batch rows) so the page can show running stats. Not saved anywhere, so it
    // clears on reload. Only submitted messages count, not live-typing previews.
    const SESSION = (function () {
        const records = [];        // { text, p, label, chars, words, source }
        const tokens = new Map();  // token -> { count, direction }
        const wordsByClass = { spam: new Map(), ham: new Map() };
        const subs = [];

        function bump(token, direction, n) {
            const cur = tokens.get(token) || { count: 0, direction: direction };
            cur.count += n;
            cur.direction = direction;
            tokens.set(token, cur);
        }
        function bumpWord(cls, word, n) {
            const m = wordsByClass[cls];
            if (!m) return;
            m.set(word, (m.get(word) || 0) + n);
        }
        function wordCount(t) {
            const s = (t || "").trim();
            return s ? s.split(/\s+/).length : 0;
        }
        function snapshot() {
            const total = records.length;
            const spam = records.reduce((n, r) => n + (r.label === "spam" ? 1 : 0), 0);
            const sumP = records.reduce((n, r) => n + r.p, 0);
            const sumC = records.reduce((n, r) => n + r.chars, 0);
            return {
                records: records,
                tokens: tokens,
                wordsSpam: wordsByClass.spam,
                wordsHam: wordsByClass.ham,
                total: total,
                spam: spam,
                ham: total - spam,
                avgP: total ? sumP / total : 0,
                avgChars: total ? sumC / total : 0,
            };
        }
        function emit() {
            const s = snapshot();
            subs.forEach((fn) => fn(s));
        }

        return {
            addSingle(text, d) {
                records.push({
                    text: (text || "").slice(0, 200),
                    p: d.prob_spam,
                    label: d.label,
                    chars: (text || "").length,
                    words: wordCount(text),
                    source: "single",
                });
                (d.top_words || []).forEach((w) => bump(w.token, w.direction, 1));
                (d.words || []).forEach((w) => bumpWord(d.label, w, 1));
                emit();
            },
            addBatch(d) {
                (d.results || []).forEach((r) => {
                    records.push({
                        text: (r.text || "").slice(0, 200),
                        p: r.prob_spam,
                        label: r.label,
                        chars: typeof r.chars === "number" ? r.chars : (r.text || "").length,
                        words: wordCount(r.text),
                        source: "batch",
                    });
                });
                (d.token_summary || []).forEach((t) => bump(t.token, t.direction, t.count));
                const wf = d.word_freq || {};
                ["spam", "ham"].forEach((cls) => {
                    (wf[cls] || []).forEach((x) => bumpWord(cls, x.word, x.count));
                });
                emit();
            },
            clear() {
                records.length = 0;
                tokens.clear();
                wordsByClass.spam.clear();
                wordsByClass.ham.clear();
                emit();
            },
            subscribe(fn) { subs.push(fn); fn(snapshot()); },
            snapshot: snapshot,
        };
    })();

    // Theme toggle
    const themeToggle = $("#themeToggle");
    function themeIcon() {
        const t = document.documentElement.getAttribute("data-theme");
        if (themeToggle) themeToggle.innerHTML = t === "dark"
            ? '<i class="bi bi-moon-stars"></i>' : '<i class="bi bi-sun"></i>';
    }
    themeIcon();
    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", next);
            try { localStorage.setItem("sg-theme", next); } catch (e) {}
            themeIcon();
            document.dispatchEvent(new CustomEvent("sg-theme"));
        });
    }

    // Background particle network
    (function particles() {
        const cvs = $("#particles");
        if (!cvs) return;
        const ctx = cvs.getContext("2d");
        let w, h, nodes, raf;
        const COUNT = window.innerWidth < 768 ? 34 : 66;

        function resize() {
            w = cvs.width = window.innerWidth;
            h = cvs.height = window.innerHeight;
        }
        function init() {
            nodes = Array.from({ length: COUNT }, () => ({
                x: Math.random() * w, y: Math.random() * h,
                vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35,
            }));
        }
        function draw() {
            const light = document.documentElement.getAttribute("data-theme") === "light";
            const base = light ? "37, 99, 235" : "34, 211, 238";
            ctx.clearRect(0, 0, w, h);
            for (const n of nodes) {
                n.x += n.vx; n.y += n.vy;
                if (n.x < 0 || n.x > w) n.vx *= -1;
                if (n.y < 0 || n.y > h) n.vy *= -1;
            }
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
                    const d = Math.hypot(dx, dy);
                    if (d < 130) {
                        ctx.strokeStyle = `rgba(${base}, ${(1 - d / 130) * (light ? 0.16 : 0.22)})`;
                        ctx.lineWidth = 1;
                        ctx.beginPath(); ctx.moveTo(nodes[i].x, nodes[i].y); ctx.lineTo(nodes[j].x, nodes[j].y); ctx.stroke();
                    }
                }
            }
            for (const n of nodes) {
                ctx.fillStyle = `rgba(${base}, ${light ? 0.5 : 0.7})`;
                ctx.beginPath(); ctx.arc(n.x, n.y, 1.6, 0, Math.PI * 2); ctx.fill();
            }
            raf = requestAnimationFrame(draw);
        }
        resize(); init();
        if (REDUCED) { draw(); cancelAnimationFrame(raf); } else { draw(); }
        window.addEventListener("resize", debounce(() => { resize(); init(); }, 200));
    })();

    // Number count-up and scroll reveals
    function fmt(v, dec) { return dec ? v.toFixed(dec) : Math.round(v).toLocaleString(); }
    function countUp(el) {
        const target = parseFloat(el.dataset.count);
        if (isNaN(target)) return;
        const dec = parseInt(el.dataset.decimals || "0", 10);
        const suffix = el.querySelector(".suffix");
        const suffixHTML = suffix ? suffix.outerHTML : "";
        if (REDUCED) { el.innerHTML = fmt(target, dec) + suffixHTML; return; }
        const dur = 1500, start = performance.now();
        (function tick(now) {
            const p = Math.min((now - start) / dur, 1);
            const val = target * (1 - Math.pow(1 - p, 3));
            el.innerHTML = fmt(val, dec) + suffixHTML;
            if (p < 1) requestAnimationFrame(tick);
            else el.innerHTML = fmt(target, dec) + suffixHTML;
        })(start);
    }
    if ("IntersectionObserver" in window) {
        const co = new IntersectionObserver((entries, obs) => {
            entries.forEach((e) => { if (e.isIntersecting) { countUp(e.target); obs.unobserve(e.target); } });
        }, { threshold: 0.4 });
        $$("[data-count]").forEach((el) => co.observe(el));

        const ro = new IntersectionObserver((entries, obs) => {
            entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("visible"); obs.unobserve(e.target); } });
        }, { threshold: 0.12 });
        $$(".reveal").forEach((el) => ro.observe(el));
    } else {
        $$("[data-count]").forEach(countUp);
        $$(".reveal").forEach((el) => el.classList.add("visible"));
    }

    // Scrolling example ticker
    (function ticker() {
        const el = $("#ticker");
        if (!el) return;
        const items = [
            { t: "Claim your FREE £500 voucher now!", s: true },
            { t: "Are we still on for lunch today?", s: false },
            { t: "URGENT: your account will be suspended", s: true },
            { t: "Thanks for the ride yesterday", s: false },
            { t: "You've WON! Txt WIN to 80086", s: true },
            { t: "Can you send the report by 5pm?", s: false },
            { t: "Pre-approved loan, reply YES now", s: true },
            { t: "Happy birthday! See you tonight", s: false },
        ];
        const html = items.concat(items).map((it) =>
            `<span class="ticker-item"><span class="badge-dot" style="background:${it.s ? cssVar("--spam") : cssVar("--ham")};box-shadow:0 0 10px ${it.s ? cssVar("--spam") : cssVar("--ham")}"></span>${escapeHtml(it.t)} <span class="${it.s ? "text-spam" : "text-ham"}">· ${it.s ? "SPAM" : "HAM"}</span></span>`
        ).join("");
        el.innerHTML = html;
    })();

    // Hero live demo
    (function heroDemo() {
        const msgEl = $("#demoMsg");
        if (!msgEl) return;
        const tag = $("#demoTag"), meter = $("#demoMeter"), meterVal = $("#demoMeterVal"), line = $("#demoVerdict");
        const samples = [
            { t: "Congratulations! You've won a £1,000 gift card. Click the link to claim now!", label: "spam", p: 0.99 },
            { t: "Hey, are we still meeting for coffee tomorrow at 10?", label: "ham", p: 0.02 },
            { t: "URGENT! Your account is locked. Verify at http://secure-login.co now", label: "spam", p: 0.98 },
            { t: "Don't forget to pick up milk on your way home, thanks!", label: "ham", p: 0.01 },
        ];
        let i = 0;

        async function typeOut(text) {
            for (let k = 0; k <= text.length; k++) {
                msgEl.innerHTML = escapeHtml(text.slice(0, k)) + '<span class="cursor"></span>';
                await sleep(REDUCED ? 0 : 22);
            }
        }
        function showVerdict(label, p) {
            const spam = label === "spam";
            tag.textContent = spam ? "SPAM" : "HAM";
            tag.className = "verdict-tag " + (spam ? "spam" : "ham");
            meter.className = spam ? "spam" : "ham";
            meter.style.width = Math.round((spam ? p : 1 - p) * 100) + "%";
            meterVal.textContent = Math.round(p * 100) + "% spam";
            line.classList.add("show");
        }
        async function loop() {
            while (true) {
                const s = samples[i % samples.length];
                line.classList.remove("show");
                await typeOut(s.t);
                let label = s.label, p = s.p;
                try { const d = await apiPredict(s.t); if (d && d.label) { label = d.label; p = d.prob_spam; } } catch (e) {}
                showVerdict(label, p);
                await sleep(2600);
                i++;
                if (REDUCED) await sleep(1500);
            }
        }
        loop();
    })();

    // Prediction page: gauge, probabilities, trigger words
    (function predictPage() {
        const input = $("#messageInput");
        if (!input) return;

        const CIRC = 2 * Math.PI * 94;
        const arc = $("#gaugeArc"), gPct = $("#gaugePct"), gVerdict = $("#gaugeVerdict");
        const probHam = $("#probHam"), probSpam = $("#probSpam"), topWords = $("#topWords");
        const note = $("#resultNote"), charCount = $("#charCount");
        const classifyBtn = $("#classifyBtn"), clearBtn = $("#clearBtn");
        const liveToggle = $("#liveToggle"), liveBadge = $("#liveBadge");

        function updateCount() {
            const t = input.value;
            const words = t.trim() ? t.trim().split(/\s+/).length : 0;
            charCount.textContent = `${t.length} chars · ${words} words`;
        }
        function resetResult() {
            arc.style.strokeDashoffset = CIRC;
            arc.setAttribute("stroke", cssVar("--ham"));
            gPct.textContent = "—"; gPct.style.color = "";
            gVerdict.textContent = "awaiting input"; gVerdict.style.color = "var(--muted)";
            probHam.textContent = "—"; probSpam.textContent = "—";
            topWords.innerHTML = '<span class="result-empty-state">Signal words will appear here as you type.</span>';
            note.classList.add("d-none");
        }
        function render(d) {
            const p = d.prob_spam;
            const col = riskColor(p);
            arc.style.strokeDashoffset = CIRC * (1 - p);
            arc.setAttribute("stroke", col);
            gPct.textContent = Math.round(p * 100) + "%";
            gPct.style.color = col;
            const spam = d.label === "spam";
            gVerdict.textContent = spam ? "▲ SPAM" : "✓ HAM";
            gVerdict.style.color = spam ? cssVar("--spam") : cssVar("--ham");
            probHam.textContent = (d.prob_ham * 100).toFixed(1) + "%";
            probSpam.textContent = (d.prob_spam * 100).toFixed(1) + "%";

            topWords.innerHTML = "";
            if (d.top_words && d.top_words.length) {
                const maxW = Math.max(...d.top_words.map((w) => w.weight), 0.0001);
                d.top_words.forEach((w) => {
                    const chip = document.createElement("span");
                    chip.className = "signal-chip " + w.direction;
                    const bw = 10 + (w.weight / maxW) * 34;
                    chip.innerHTML = `${escapeHtml(w.token)}<span class="bar" style="width:${bw}px"></span>`;
                    topWords.appendChild(chip);
                });
            } else {
                topWords.innerHTML = '<span class="result-empty-state">No strong signal words detected.</span>';
            }
            if (d.note) { note.textContent = d.note; note.classList.remove("d-none"); }
            else { note.classList.add("d-none"); }
        }

        let busy = false;
        async function run(text, force) {
            if (!text) { resetResult(); return; }
            if (busy && !force) return;
            busy = true;
            try {
                const d = await apiPredict(text);
                if (d.error) {
                    resetResult();
                } else {
                    render(d);
                    // Only record when the user submits (force). Live typing
                    // fires on every keystroke and would spam the history.
                    if (force) SESSION.addSingle(text, d);
                }
            }
            catch (e) { /* silent for live */ }
            finally { busy = false; }
        }
        const liveRun = debounce(() => {
            if (!liveToggle.checked) return;
            const t = input.value.trim();
            if (t.split(/\s+/).filter(Boolean).length >= 2) run(t, false);
            else resetResult();
        }, 420);

        input.addEventListener("input", () => { updateCount(); liveRun(); });
        classifyBtn.addEventListener("click", async () => {
            const t = input.value.trim();
            if (!t) { input.focus(); return; }
            classifyBtn.disabled = true;
            classifyBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Scanning...';
            await run(t, true);
            classifyBtn.disabled = false;
            classifyBtn.innerHTML = '<i class="bi bi-search me-1"></i>Classify';
        });
        input.addEventListener("keydown", (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") classifyBtn.click();
        });
        clearBtn.addEventListener("click", () => { input.value = ""; updateCount(); resetResult(); input.focus(); });
        liveToggle.addEventListener("change", () => { liveBadge.style.display = liveToggle.checked ? "" : "none"; });

        const SAMPLES = {
            spam: "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt the word CLAIM to 81010. To stop txts reply STOP. www.dbuk.net",
            ham: "Hey, are we still on for dinner tonight? Thinking we could try that new Italian place near the office around 7. Let me know!",
        };
        $$(".sample-btn").forEach((b) => b.addEventListener("click", () => {
            input.value = SAMPLES[b.dataset.type]; updateCount(); run(input.value, true);
        }));

        updateCount(); resetResult();
    })();

    // Batch CSV upload
    (function batch() {
        const batchBtn = $("#batchBtn");
        if (!batchBtn) return;
        const csvInput = $("#csvInput");
        let last = [];

        batchBtn.addEventListener("click", async () => {
            if (!csvInput.files.length) { csvInput.focus(); return; }
            batchBtn.disabled = true;
            batchBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Scanning...';
            const fd = new FormData(); fd.append("file", csvInput.files[0]);
            try {
                const r = await fetch("/api/predict_batch", { method: "POST", body: fd });
                const d = await r.json();
                if (!r.ok) throw new Error(d.error || "Batch failed.");
                last = d.results; render(d); SESSION.addBatch(d);
            } catch (e) { alert(e.message); }
            finally { batchBtn.disabled = false; batchBtn.innerHTML = '<i class="bi bi-upload me-1"></i>Scan file'; }
        });

        function render(d) {
            $("#batchSummary").classList.remove("d-none");
            $("#bTotal").textContent = d.total; $("#bSpam").textContent = d.spam; $("#bHam").textContent = d.ham;
            const note = $("#batchNote");
            let m = `Detected message column <code>${escapeHtml(d.text_column)}</code>.`;
            if (d.truncated) m += ` Showing the first ${d.max_rows.toLocaleString()} rows.`;
            note.innerHTML = m; note.classList.remove("d-none");
            const body = $("#batchTableBody"); body.innerHTML = "";
            d.results.forEach((r) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td class="small">${escapeHtml(r.text)}</td>`
                    + `<td class="${r.label === "spam" ? "pill-spam" : "pill-ham"}">${r.label === "spam" ? "▲ Spam" : "✓ Ham"}</td>`
                    + `<td class="mono">${(r.prob_spam * 100).toFixed(1)}%</td>`;
                body.appendChild(tr);
            });
            $("#batchResultWrap").classList.remove("d-none");
        }
        $("#downloadBtn").addEventListener("click", () => {
            if (!last.length) return;
            const rows = last.map((r) => `"${(r.text || "").replace(/"/g, '""')}",${r.prediction},${r.prob_spam}`).join("\n");
            const blob = new Blob(["text,prediction,prob_spam\n" + rows], { type: "text/csv" });
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob); a.download = "spam_predictions.csv"; a.click();
            URL.revokeObjectURL(a.href);
        });
    })();

    // Session analytics on the predict page: running stats and four charts
    // that redraw when the theme changes so the colors stay in sync.
    (function sessionAnalytics() {
        const panel = $("#sessionPanel");
        if (!panel) return;

        const empty = $("#sessEmpty"), body = $("#sessBody");
        const btnClear = $("#sessClear"), btnExport = $("#sessExport");
        let charts = [];

        function pal() {
            return {
                muted: cssVar("--muted"), grid: hexToRgba(cssVar("--cyan"), 0.08),
                cyan: cssVar("--cyan"), ham: cssVar("--ham"), spam: cssVar("--spam"),
            };
        }
        function destroyCharts() { charts.forEach((c) => c && c.destroy()); charts = []; }

        function stats(s) {
            $("#sTotal").textContent = s.total.toLocaleString();
            $("#sSpam").textContent = s.spam.toLocaleString();
            $("#sHam").textContent = s.ham.toLocaleString();
            $("#sAvgRisk").textContent = Math.round(s.avgP * 100) + "%";
            $("#sAvgLen").textContent = Math.round(s.avgChars).toLocaleString();
        }

        function build(s) {
            destroyCharts();
            if (typeof Chart === "undefined") return;
            const p = pal();
            Chart.defaults.color = p.muted;
            Chart.defaults.font.family = "'Inter', sans-serif";
            const noLegend = { legend: { display: false } };

            const donut = $("#sessDonut");
            if (donut) charts.push(safeNew(donut, {
                type: "doughnut",
                data: {
                    labels: ["Ham", "Spam"],
                    datasets: [{
                        data: [s.ham, s.spam],
                        backgroundColor: [hexToRgba(p.ham, .85), hexToRgba(p.spam, .85)],
                        borderColor: [p.ham, p.spam], borderWidth: 2, hoverOffset: 8,
                    }],
                },
                options: {
                    responsive: true, maintainAspectRatio: false, cutout: "62%",
                    plugins: { legend: { position: "bottom", labels: { usePointStyle: true, padding: 14 } } },
                },
            }));

            const buckets = [0, 0, 0, 0, 0];
            s.records.forEach((r) => { buckets[Math.min(4, Math.floor(r.p * 5))]++; });
            const risk = $("#sessRisk");
            if (risk) charts.push(safeNew(risk, {
                type: "bar",
                data: {
                    labels: ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
                    datasets: [{
                        data: buckets, borderRadius: 6, borderWidth: 0,
                        backgroundColor: buckets.map((_, i) => riskColor((i + 0.5) / 5)),
                    }],
                },
                options: {
                    responsive: true, maintainAspectRatio: false, plugins: noLegend,
                    scales: {
                        x: { grid: { display: false }, title: { display: true, text: "spam probability" } },
                        y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: p.grid } },
                    },
                },
            }));

            const tl = $("#sessTimeline");
            if (tl) charts.push(safeNew(tl, {
                type: "line",
                data: {
                    labels: s.records.map((_, i) => i + 1),
                    datasets: [
                        {
                            data: s.records.map((r) => +(r.p * 100).toFixed(1)),
                            borderColor: p.cyan, backgroundColor: hexToRgba(p.cyan, .12),
                            fill: true, tension: .3, borderWidth: 2,
                            pointRadius: s.total > 60 ? 0 : 3,
                            pointBackgroundColor: s.records.map((r) => riskColor(r.p)),
                            pointBorderColor: "transparent",
                        },
                        {
                            data: s.records.map(() => 50),
                            borderColor: hexToRgba(p.spam, .5),
                            borderDash: [6, 6], borderWidth: 1, pointRadius: 0, fill: false,
                        },
                    ],
                },
                options: {
                    responsive: true, maintainAspectRatio: false, plugins: noLegend,
                    scales: {
                        x: { grid: { display: false }, title: { display: true, text: "scan order" } },
                        y: { min: 0, max: 100, grid: { color: p.grid }, ticks: { callback: (v) => v + "%" } },
                    },
                },
            }));

            // Message length distribution, spam vs ham, over the inserted messages
            const lbEdges = [0, 40, 80, 120, 160, 200, Infinity];
            const lbLabels = ["0-40", "40-80", "80-120", "120-160", "160-200", "200+"];
            const lenHam = lbLabels.map(() => 0), lenSpam = lbLabels.map(() => 0);
            s.records.forEach((r) => {
                let bi = lbEdges.findIndex((e, i) => r.chars >= e && r.chars < lbEdges[i + 1]);
                if (bi < 0) bi = lbLabels.length - 1;
                (r.label === "spam" ? lenSpam : lenHam)[bi]++;
            });
            const len = $("#sessLength");
            if (len) charts.push(safeNew(len, {
                type: "bar",
                data: {
                    labels: lbLabels,
                    datasets: [
                        { label: "Ham", data: lenHam, backgroundColor: hexToRgba(p.ham, .75), borderColor: p.ham, borderWidth: 1, borderRadius: 4 },
                        { label: "Spam", data: lenSpam, backgroundColor: hexToRgba(p.spam, .75), borderColor: p.spam, borderWidth: 1, borderRadius: 4 },
                    ],
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { position: "bottom", labels: { usePointStyle: true, padding: 12 } } },
                    scales: {
                        x: { grid: { display: false }, title: { display: true, text: "message length (characters)" } },
                        y: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: p.grid } },
                    },
                },
            }));

            // Top words per class, taken from the text and files you classify
            function wordChart(sel, map, color) {
                const el = $(sel); if (!el) return;
                const items = Array.from(map.entries()).sort((a, b) => b[1] - a[1]).slice(0, 10).reverse();
                charts.push(safeNew(el, {
                    type: "bar",
                    data: {
                        labels: items.map((w) => w[0]),
                        datasets: [{ data: items.map((w) => w[1]), backgroundColor: hexToRgba(color, .78), borderColor: color, borderWidth: 1, borderRadius: 5 }],
                    },
                    options: {
                        indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: noLegend,
                        scales: {
                            x: { beginAtZero: true, ticks: { precision: 0 }, grid: { color: p.grid } },
                            y: { grid: { display: false }, ticks: { font: { family: "'JetBrains Mono', monospace" } } },
                        },
                    },
                }));
            }
            wordChart("#sessWordsSpam", s.wordsSpam, p.spam);
            wordChart("#sessWordsHam", s.wordsHam, p.ham);
        }

        function render(s) {
            btnClear.disabled = !s.total;
            btnExport.disabled = !s.total;
            if (!s.total) {
                empty.classList.remove("d-none");
                body.classList.add("d-none");
                destroyCharts();
                return;
            }
            empty.classList.add("d-none");
            body.classList.remove("d-none");  // un-hide first so canvases have a size
            stats(s);
            build(s);
        }

        btnClear.addEventListener("click", () => SESSION.clear());
        btnExport.addEventListener("click", () => {
            const s = SESSION.snapshot();
            if (!s.total) return;
            const rows = s.records.map((r) =>
                `"${(r.text || "").replace(/"/g, '""')}",${r.label},${r.p},${r.chars},${r.words},${r.source}`
            ).join("\n");
            const blob = new Blob(["text,label,prob_spam,chars,words,source\n" + rows], { type: "text/csv" });
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "spamguard_session.csv";
            a.click();
            URL.revokeObjectURL(a.href);
        });

        SESSION.subscribe(render);
        document.addEventListener("sg-theme", () => render(SESSION.snapshot()));
    })();

    // Copy buttons on the developer card
    $$(".dev-copy").forEach((b) => b.addEventListener("click", async () => {
        const val = b.dataset.copy || "";
        try {
            await navigator.clipboard.writeText(val);
        } catch (e) {
            const t = document.createElement("textarea");
            t.value = val;
            document.body.appendChild(t);
            t.select();
            try { document.execCommand("copy"); } catch (e2) {}
            t.remove();
        }
        const old = b.innerHTML;
        b.classList.add("copied");
        b.innerHTML = '<i class="bi bi-check2"></i>';
        setTimeout(() => { b.innerHTML = old; b.classList.remove("copied"); }, 1400);
    }));

    // Analytics page charts (Chart.js)
    (function analytics() {
        const dataEl = $("#chart-data");
        if (!dataEl) return;
        if (typeof Chart === "undefined") {
            $$(".chart-holder").forEach((h) => {
                h.innerHTML = '<p class="chart-fallback">Chart library did not load. Check your connection and refresh.</p>';
            });
            return;
        }
        const CH = JSON.parse(dataEl.textContent);
        const MODEL_KEYS = Object.keys(CH.models);
        let charts = [];

        function P() {
            return {
                text: cssVar("--text"), muted: cssVar("--muted"),
                grid: hexToRgba(cssVar("--cyan"), 0.08),
                cyan: cssVar("--cyan"), violet: cssVar("--violet"),
                ham: cssVar("--ham"), spam: cssVar("--spam"), amber: "#f59e0b",
            };
        }
        function modelColors() { const p = P(); return [p.cyan, p.violet, p.amber, p.spam]; }

        function buildAll(modelMode) {
            charts.forEach((c) => c && c.destroy());
            charts = [];
            const p = P();
            Chart.defaults.color = p.muted;
            Chart.defaults.font.family = "'Inter', sans-serif";
            Chart.defaults.borderColor = p.grid;

            // Class distribution doughnut
            const cls = $("#chartClass");
            if (cls) {
                charts.push(safeNew(cls, {
                    type: "doughnut",
                    data: {
                        labels: ["Ham", "Spam"],
                        datasets: [{
                            data: [window.__M.ham, window.__M.spam],
                            backgroundColor: [hexToRgba(p.ham, 0.85), hexToRgba(p.spam, 0.85)],
                            borderColor: [p.ham, p.spam], borderWidth: 2, hoverOffset: 8,
                        }],
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false, cutout: "64%",
                        plugins: { legend: { position: "bottom", labels: { usePointStyle: true, padding: 18 } } },
                    },
                    plugins: [{
                        id: "ct",
                        afterDraw(c) {
                            const { ctx, chartArea: { left, right, top, bottom } } = c;
                            const x = (left + right) / 2, y = (top + bottom) / 2;
                            ctx.save();
                            ctx.textAlign = "center"; ctx.textBaseline = "middle";
                            ctx.fillStyle = p.text; ctx.font = "700 26px 'Space Grotesk', sans-serif";
                            ctx.fillText((window.__M.ham + window.__M.spam).toLocaleString(), x, y - 8);
                            ctx.fillStyle = p.muted; ctx.font = "500 12px 'JetBrains Mono', monospace";
                            ctx.fillText("messages", x, y + 14);
                            ctx.restore();
                        },
                    }],
                }));
            }

            // Model comparison radar / bar
            const mc = $("#chartModels");
            if (mc) {
                const metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"];
                const labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"];
                const cols = modelColors();
                const ds = MODEL_KEYS.map((name, i) => ({
                    label: name,
                    data: metrics.map((m) => CH.models[name][m]),
                    borderColor: cols[i],
                    backgroundColor: hexToRgba(cols[i].startsWith("#") ? cols[i] : "#22d3ee", modelMode === "radar" ? 0.12 : 0.7),
                    borderWidth: 2, pointBackgroundColor: cols[i], tension: 0.1,
                }));
                const common = { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { usePointStyle: true, padding: 14 } } } };
                if (modelMode === "radar") {
                    charts.push(safeNew(mc, {
                        type: "radar", data: { labels, datasets: ds },
                        options: Object.assign({}, common, {
                            scales: { r: { min: 0.8, max: 1, ticks: { stepSize: 0.05, backdropColor: "transparent", color: p.muted, font: { size: 9 } }, grid: { color: p.grid }, angleLines: { color: p.grid }, pointLabels: { color: p.text, font: { size: 12, weight: "600" } } } },
                        }),
                    }));
                } else {
                    charts.push(safeNew(mc, {
                        type: "bar", data: { labels, datasets: ds },
                        options: Object.assign({}, common, {
                            scales: { y: { min: 0.8, max: 1, grid: { color: p.grid } }, x: { grid: { display: false } } },
                        }),
                    }));
                }
            }

            // ROC
            const roc = $("#chartRoc");
            if (roc) {
                const cols = modelColors();
                const ds = MODEL_KEYS.map((name, i) => ({
                    label: `${name} (${CH.roc[name].auc})`,
                    data: CH.roc[name].fpr.map((f, k) => ({ x: f, y: CH.roc[name].tpr[k] })),
                    borderColor: cols[i], backgroundColor: "transparent",
                    borderWidth: 2, pointRadius: 0, tension: 0,
                }));
                ds.push({ label: "chance", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: p.muted, borderDash: [6, 6], borderWidth: 1, pointRadius: 0 });
                charts.push(safeNew(roc, {
                    type: "line", data: { datasets: ds },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        plugins: { legend: { position: "bottom", labels: { usePointStyle: true, padding: 12, filter: (i) => i.text !== "chance" } } },
                        scales: {
                            x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate", color: p.muted }, grid: { color: p.grid } },
                            y: { min: 0, max: 1, title: { display: true, text: "True Positive Rate", color: p.muted }, grid: { color: p.grid } },
                        },
                    },
                }));
            }

            // Length distribution
            const len = $("#chartLength");
            if (len) {
                charts.push(safeNew(len, {
                    type: "line",
                    data: {
                        labels: CH.length.labels,
                        datasets: [
                            { label: "Ham", data: CH.length.ham, borderColor: p.ham, backgroundColor: hexToRgba(p.ham, 0.18), fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2 },
                            { label: "Spam", data: CH.length.spam, borderColor: p.spam, backgroundColor: hexToRgba(p.spam, 0.18), fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2 },
                        ],
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        plugins: { legend: { position: "bottom", labels: { usePointStyle: true, padding: 14 } } },
                        scales: { x: { title: { display: true, text: "Message length (characters)", color: p.muted }, grid: { display: false } }, y: { title: { display: true, text: "Messages", color: p.muted }, grid: { color: p.grid } } },
                    },
                }));
            }

            // Frequent words
            function wordChart(sel, key, color) {
                const el = $(sel); if (!el) return;
                const items = CH.frequent_words[key];
                charts.push(safeNew(el, {
                    type: "bar",
                    data: {
                        labels: items.map((w) => w.word),
                        datasets: [{ data: items.map((w) => w.count), backgroundColor: hexToRgba(color, 0.75), borderColor: color, borderWidth: 1, borderRadius: 5 }],
                    },
                    options: {
                        indexAxis: "y", responsive: true, maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: { x: { grid: { color: p.grid } }, y: { grid: { display: false }, ticks: { font: { family: "'JetBrains Mono', monospace" } } } },
                    },
                }));
            }
            wordChart("#chartWordsSpam", "spam", p.spam);
            wordChart("#chartWordsHam", "ham", p.ham);
        }

        // dataset stats for the doughnut center (from metrics)
        const mEl = $("#metrics-data");
        const M = mEl ? JSON.parse(mEl.textContent) : {};
        window.__M = { ham: M.dataset ? M.dataset.ham : 0, spam: M.dataset ? M.dataset.spam : 0 };

        let mode = "radar";
        buildAll(mode);

        const toggle = $("#modelChartToggle");
        if (toggle) {
            $$("button", toggle).forEach((b) => b.addEventListener("click", () => {
                $$("button", toggle).forEach((x) => x.classList.remove("active"));
                b.classList.add("active");
                mode = b.dataset.mode;
                buildAll(mode);
            }));
        }
        document.addEventListener("sg-theme", () => buildAll(mode));
    })();
})();
