import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Info, Message, Mode, Passage, Session, SourceId, StageEvent } from "./api";
import { api, sendMessage } from "./api";

// ---------- constants ----------
const SOURCES: { id: SourceId; chip: string; label: string; type: string }[] = [
  { id: "dailymed", chip: "DailyMed", label: "DailyMed drug labels", type: "Drug label" },
  { id: "pmc_oa", chip: "PMC", label: "PubMed Central articles", type: "Literature" },
  { id: "nih_livertox", chip: "LiverTox", label: "NIH LiverTox", type: "LiverTox" },
  { id: "fda_drugsatfda", chip: "FDA letters", label: "FDA approval letters", type: "FDA letter" },
];
const TYPE = Object.fromEntries(SOURCES.map((s) => [s.id, s]));
const SUGGESTIONS = [
  { cat: "Dosing", q: "How should gabapentin be dosed in patients with renal impairment?" },
  { cat: "Interactions", q: "Can warfarin be taken with aspirin or NSAIDs such as naproxen?" },
  { cat: "Safety", q: "What is the boxed warning for Synthroid (levothyroxine)?" },
  { cat: "Liver injury", q: "How soon after starting atorvastatin can liver injury appear?" },
];
const STAGES: { id: StageEvent["stage"]; label: string }[] = [
  { id: "understand", label: "Understanding the question" },
  { id: "search", label: "Searching knowledge bases" },
  { id: "compose", label: "Composing cited answer" },
];
const SIZES = { Small: "16px", Default: "17.5px", Large: "19.5px" } as const;
type Size = keyof typeof SIZES;
type Theme = "light" | "dark" | "system";
type Screen = "home" | "chat" | "settings";
type Tab = "sources" | "graph" | "notes";

interface Prefs { theme: Theme; size: Size; k: number; mode: Mode; sources: SourceId[] }
const DEFAULT_PREFS: Prefs = { theme: "light", size: "Default", k: 8, mode: "hybrid", sources: SOURCES.map((s) => s.id) };

interface Pending { question: string; stages: Record<string, StageEvent>; passages: Passage[]; error?: string }
interface Note { key: string; tag: string; text: string }

// ---------- small helpers ----------
function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
  } catch {
    return fallback;
  }
}
function save(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}
function loadList<T>(key: string): T[] {
  try { return JSON.parse(localStorage.getItem(key) || "[]"); } catch { return []; }
}
const sNum = (sid: string) => Number(sid.replace(/^S/, ""));
const fmtS = (ms?: number) => (ms ? `${(ms / 1000).toFixed(1)}s` : "");
function useWidth() {
  const [w, setW] = useState(window.innerWidth);
  useEffect(() => {
    const on = () => setW(window.innerWidth);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return w;
}

const Mark = ({ small }: { small?: boolean }) => (
  <div className={"mark" + (small ? " sm" : "")}><div className="h" /><div className="v" />{!small && <div className="dot" />}</div>
);
const Word = ({ size }: { size?: number }) => (
  <div className="word" style={size ? { fontSize: size } : undefined}><span>Beyond</span><span className="d">Doc</span></div>
);
const Arrow = ({ s = 16 }: { s?: number }) => (
  <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
);
const Check = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12.5l4.5 4.5L19 7" /></svg>
);

// ---------- app ----------
export default function App() {
  const vw = useWidth();
  const narrow = vw < 1240;
  const [prefs, setPrefsState] = useState<Prefs>(() => load("beyonddoc-prefs", DEFAULT_PREFS));
  const setPrefs = (p: Partial<Prefs>) => setPrefsState((cur) => { const next = { ...cur, ...p }; save("beyonddoc-prefs", next); return next; });
  const [info, setInfo] = useState<Info | null>(null);
  const [me, setMe] = useState<{ user: string; retention_days: number } | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [screen, setScreen] = useState<Screen>("home");
  const [active, setActive] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [pending, setPending] = useState<Pending | null>(null);
  const [draft, setDraft] = useState("");
  const [follow, setFollow] = useState("");
  const [tab, setTab] = useState<Tab>("sources");
  const [panelOpen, setPanelOpen] = useState(false);
  const [focusMsg, setFocusMsg] = useState<string | null>(null);
  const [activeSrc, setActiveSrc] = useState<number | null>(null);
  const [traceOpen, setTraceOpen] = useState<Set<string>>(new Set());
  const [notes, setNotes] = useState<Note[]>([]);
  const [noteText, setNoteText] = useState("");
  const [copied, setCopied] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  // theme
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      document.documentElement.dataset.theme = prefs.theme === "system" ? (mq.matches ? "dark" : "light") : prefs.theme;
      document.documentElement.style.setProperty("--ans", SIZES[prefs.size]);
    };
    apply();
    mq.addEventListener?.("change", apply);
    return () => mq.removeEventListener?.("change", apply);
  }, [prefs.theme, prefs.size]);

  // initial load
  useEffect(() => {
    api.info().then(setInfo).catch(() => setInfo(null));
    api.me().then(setMe).catch(() => setMe(null));
    api.sessions().then(setSessions).catch((e) => setLoadError(String(e.message || e)));
  }, []);

  // notes per chat (browser only)
  useEffect(() => {
    if (!active) { setNotes([]); setNoteText(""); return; }
    setNotes(loadList<Note>(`beyonddoc-notes-${active.session_id}`));
    try { setNoteText(localStorage.getItem(`beyonddoc-notetext-${active.session_id}`) || ""); } catch { setNoteText(""); }
  }, [active?.session_id]);
  const saveNotes = (list: Note[]) => { setNotes(list); if (active) save(`beyonddoc-notes-${active.session_id}`, list); };
  const pin = (tag: string, text: string) => { saveNotes([...notes, { key: crypto.randomUUID(), tag, text }]); setTab("notes"); if (narrow) setPanelOpen(true); };

  // ⌘K / Ctrl+K: new consultation
  useEffect(() => {
    const on = (e: KeyboardEvent) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); newChat(); } };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  });

  // auto-scroll
  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" }); }, [messages.length, pending?.passages.length, pending?.stages.compose?.status]);

  const newChat = () => { setScreen("home"); setActive(null); setMessages([]); setPending(null); setDraft(""); setFocusMsg(null); setActiveSrc(null); };

  const openSession = async (s: Session) => {
    if (pending) return;
    setScreen("chat"); setActive(s); setMessages([]); setFocusMsg(null); setActiveSrc(null); setTab("sources");
    try {
      const d = await api.session(s.session_id);
      setMessages(d.messages);
    } catch (e) {
      setLoadError(String((e as Error).message));
    }
  };

  const ask = useCallback(async (session: Session, question: string) => {
    const q = question.trim();
    if (q.length < 3) return;
    setPending({ question: q, stages: {}, passages: [] });
    setFocusMsg(null); setActiveSrc(null);
    await sendMessage(
      session.session_id,
      { content: q, k: prefs.k, mode: prefs.mode, sources: prefs.sources.length === SOURCES.length ? null : prefs.sources },
      {
        onUser: (m) => setMessages((cur) => [...cur, m]),
        onStage: (st) => setPending((p) => (p ? { ...p, stages: { ...p.stages, [st.stage]: st } } : p)),
        onSources: (d) => setPending((p) => (p ? { ...p, passages: d.passages } : p)),
        onAnswer: (d) => {
          setMessages((cur) => [...cur, d.message]);
          setPending(null);
          setActive(d.session);
          setSessions((cur) => [d.session, ...cur.filter((x) => x.session_id !== d.session.session_id)]);
        },
        onError: (detail) => setPending((p) => (p ? { ...p, error: detail } : { question: q, stages: {}, passages: [], error: detail })),
      },
    ).catch((e) => setPending((p) => (p ? { ...p, error: String(e.message || e) } : p)));
  }, [prefs.k, prefs.mode, prefs.sources]);

  const startChat = async (question: string) => {
    const q = question.trim();
    if (q.length < 3 || pending) return;
    try {
      const s = await api.createSession();
      setSessions((cur) => [s, ...cur]);
      setActive(s); setMessages([]); setScreen("chat"); setDraft(""); setTab("sources");
      await ask(s, q);
    } catch (e) {
      setLoadError(String((e as Error).message));
    }
  };

  const sendFollow = () => { if (active && !pending && follow.trim().length >= 3) { const q = follow; setFollow(""); ask(active, q); } };

  const removeSession = async (s: Session) => {
    if (!confirm(`Delete “${s.title}”?`)) return;
    await api.remove(s.session_id).catch(() => undefined);
    setSessions((cur) => cur.filter((x) => x.session_id !== s.session_id));
    if (active?.session_id === s.session_id) newChat();
  };
  const renameSession = async () => {
    if (!active) return;
    const title = prompt("Rename consultation", active.title)?.trim();
    if (!title) return;
    const s = await api.rename(active.session_id, title);
    setActive(s); setSessions((cur) => cur.map((x) => (x.session_id === s.session_id ? s : x)));
  };

  const copy = (key: string, text: string) => {
    navigator.clipboard?.writeText(text).catch(() => undefined);
    setCopied(key); setTimeout(() => setCopied(null), 1400);
  };

  // which assistant message the evidence panel shows: focused, else the latest
  const assistants = messages.filter((m) => m.role === "assistant");
  const panelMsg = assistants.find((m) => m.message_id === focusMsg) || assistants[assistants.length - 1];
  const panelPassages = pending ? pending.passages : panelMsg?.sources || [];
  const panelCited = pending ? [] : panelMsg?.cited || [];

  const citeClick = (m: Message, n: number) => {
    setFocusMsg(m.message_id); setActiveSrc(n); setTab("sources"); if (narrow) setPanelOpen(true);
  };

  // sidebar history groups
  const groups = useMemo(() => {
    const day = 86400000, nowMs = Date.now();
    const start = new Date(); start.setHours(0, 0, 0, 0);
    const g: Record<string, Session[]> = { Today: [], "Previous 7 days": [], Older: [] };
    for (const s of sessions) {
      const t = new Date(s.updated_at).getTime();
      (t >= start.getTime() ? g.Today : t >= nowMs - 7 * day ? g["Previous 7 days"] : g.Older).push(s);
    }
    return g;
  }, [sessions]);

  const kbOn = prefs.sources.length;
  const sideW = vw < 1000 ? 224 : 268;
  const chatCols = narrow ? "minmax(0,1fr)" : `minmax(0,1fr) ${vw < 1400 ? 344 : 392}px`;
  const toggleSource = (id: SourceId) => {
    const on = prefs.sources.includes(id);
    if (on && prefs.sources.length === 1) return; // keep at least one source
    setPrefs({ sources: on ? prefs.sources.filter((x) => x !== id) : [...prefs.sources, id] });
  };

  return (
    <div className="app" style={{ gridTemplateColumns: `${sideW}px minmax(0,1fr)` }}>
      {/* ---------------- sidebar ---------------- */}
      <aside className="side">
        <div className="brand"><Mark /><Word /></div>
        <div style={{ padding: "6px 12px 10px" }}>
          <button className="newchat" onClick={newChat}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
            New consultation<kbd>Ctrl K</kbd>
          </button>
        </div>
        <div className="hist">
          {sessions.length === 0 && <div className="hist-empty">{loadError ? `History unavailable: ${loadError}` : "Your consultations will appear here."}</div>}
          {Object.entries(groups).map(([label, list]) => list.length > 0 && (
            <div className="hist-group" key={label}>
              <div className="mono-label">{label}</div>
              {list.map((s) => (
                <div key={s.session_id} className={"hist-item" + (screen === "chat" && active?.session_id === s.session_id ? " on" : "")}>
                  <button className="t" title={s.title} onClick={() => openSession(s)}>{s.title}</button>
                  <button className="x" aria-label="Delete consultation" onClick={() => removeSession(s)}>×</button>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className="side-foot">
          <button className={"navbtn" + (screen === "settings" ? " on" : "")} onClick={() => setScreen("settings")}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M4 7h9M17 7h3M4 17h3M11 17h9" /><circle cx="15" cy="7" r="2" /><circle cx="9" cy="17" r="2" /></svg>
            Settings
          </button>
          <div className="who">
            <div className="avatar">{(me?.user || "··").slice(0, 2).toUpperCase()}</div>
            <div style={{ minWidth: 0 }}>
              <div className="n">Guest clinician</div>
              <div className="r">Local demo · {me ? `id ${me.user}` : "offline"}</div>
            </div>
          </div>
        </div>
      </aside>

      <main>
        {/* ---------------- home ---------------- */}
        {screen === "home" && (
          <div className="home">
            <div className="home-inner">
              <div className="home-col">
                <div className="hero">
                  <div className="kbpill"><span className="d" />{kbOn} of 4 sources connected{info?.chunks ? ` · ${info.chunks.toLocaleString()} passages` : ""}</div>
                  <h1><span>Beyond</span><span className="d">Doc</span></h1>
                  <p>Evidence-grounded answers from public drug labels, biomedical literature, LiverTox and FDA letters — every claim traced to its source.</p>
                </div>
                <div className="composer">
                  <textarea rows={3} value={draft} placeholder="Ask about a medicine — dosing, interactions, warnings, liver injury…"
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); startChat(draft); } }} />
                  <div className="row">
                    {SOURCES.map((s) => (
                      <button key={s.id} className={"chip" + (prefs.sources.includes(s.id) ? " on" : "")} onClick={() => toggleSource(s.id)}>
                        <span className="d" />{s.chip}
                      </button>
                    ))}
                    <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12 }}>
                      <span className="tiny-mono">{info && !info.llm_configured ? "answers off · no API key" : "informational only"}</span>
                      <button className="send" aria-label="Send" disabled={draft.trim().length < 3 || !!pending} onClick={() => startChat(draft)}><Arrow /></button>
                    </div>
                  </div>
                </div>
                <div className="sugg">
                  {SUGGESTIONS.map((s) => (
                    <button key={s.q} onClick={() => startChat(s.q)}>
                      <div className="cat"><span>{s.cat}</span>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink3)" strokeWidth="2" strokeLinecap="round"><path d="M7 17L17 7M8 7h9v9" /></svg>
                      </div>
                      <span className="q">{s.q}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="foot-note">{info?.disclaimer || "Informational demo over public documents. Not medical advice; not a diagnosis or treatment system."}</div>
          </div>
        )}

        {/* ---------------- chat ---------------- */}
        {screen === "chat" && active && (
          <div className="chat" style={{ gridTemplateColumns: chatCols }}>
            <div className="chat-main">
              <div className="chat-head">
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="tt" title={active.title}>{active.title}</div>
                  <div className="st">Evidence mode · {prefs.mode} · k={prefs.k} · {kbOn} of 4 sources</div>
                </div>
                <div style={{ marginLeft: "auto", display: "flex", gap: 8, flex: "none" }}>
                  {narrow && <button className={"btn" + (panelOpen ? " on" : "")} onClick={() => setPanelOpen(!panelOpen)}>Evidence<span className="tiny-mono" style={{ color: "var(--accent)" }}>{panelCited.length || panelPassages.length}</span></button>}
                  <button className="btn" onClick={renameSession}>Rename</button>
                  <button className="btn dark" onClick={() => copy("chat", transcript(active, messages))}>{copied === "chat" ? "Copied" : "Copy chat"}</button>
                </div>
              </div>

              <div className="scroll" ref={scroller}>
                <div className="thread">
                  {messages.map((m) => m.role === "user" ? (
                    <div className="ubub" key={m.message_id}><div>{m.content}</div></div>
                  ) : (
                    <BotMessage key={m.message_id} m={m} kbOn={kbOn} chunks={info?.chunks}
                      activeSrc={panelMsg?.message_id === m.message_id ? activeSrc : null}
                      trace={traceOpen.has(m.message_id)}
                      onTrace={() => setTraceOpen((cur) => { const n = new Set(cur); n.has(m.message_id) ? n.delete(m.message_id) : n.add(m.message_id); return n; })}
                      onCite={(n) => citeClick(m, n)}
                      copied={copied === m.message_id}
                      onCopy={() => copy(m.message_id, m.content)}
                      onPin={() => pin(`Answer · ${(m.cited || []).map((c) => `[${sNum(c)}]`).join("")}`, m.content)}
                      onRetrieve={() => { const prev = messages[messages.indexOf(m) - 1]; if (prev && !pending) ask(active, prev.content); }}
                      onFocus={() => setFocusMsg(m.message_id)} />
                  ))}
                  {pending && (
                    <>
                      {!messages.some((m) => m.role === "user" && m.content === pending.question && messages.indexOf(m) === messages.length - 1) && (
                        <div className="ubub"><div>{pending.question}</div></div>
                      )}
                      <div className="bot">
                        <div className="bot-head"><Mark small /><Word size={17} /></div>
                        <Pipeline stages={pending.stages} />
                        {pending.error && <div className="err">{pending.error} <button className="act" onClick={() => { const q = pending.question; setPending(null); ask(active, q); }}>Retry</button></div>}
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="follow-wrap">
                <div className="follow">
                  <input value={follow} placeholder={pending ? "Answering…" : "Ask a follow-up…"} disabled={!!pending}
                    onChange={(e) => setFollow(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") sendFollow(); }} />
                  <span className="tiny-mono">k={prefs.k}</span>
                  <button className="send sm" aria-label="Send" disabled={!!pending || follow.trim().length < 3} onClick={sendFollow}><Arrow s={15} /></button>
                </div>
                <div className="follow-note">Informational demo over public documents — not medical advice. Verify against official labeling.</div>
              </div>
            </div>

            {(!narrow || panelOpen) && (
              <aside className={"panel" + (narrow ? " float" : "")}>
                <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--line)" }}>
                  <div className="tabs">
                    {([["sources", "Sources", String(panelPassages.length || "")], ["graph", "Network", ""], ["notes", "Notes", notes.length ? String(notes.length) : ""]] as const).map(([id, label, count]) => (
                      <button key={id} className={tab === id ? "on" : ""} onClick={() => setTab(id)}>{label}<span>{count}</span></button>
                    ))}
                  </div>
                </div>
                <div className="panel-body">
                  {tab === "sources" && <SourcesPanel passages={panelPassages} cited={panelCited} k={prefs.k} activeSrc={activeSrc} setActiveSrc={setActiveSrc} onPin={(p, n) => pin(`Source [${n}] · ${p.title}`, p.text.slice(0, 600))} />}
                  {tab === "graph" && <NetworkPanel passages={panelPassages} cited={panelCited} activeSrc={activeSrc} setActiveSrc={setActiveSrc} />}
                  {tab === "notes" && (
                    <div className="pad" style={{ gap: 12, minHeight: "100%" }}>
                      <div><div style={{ font: "500 14px/1.2 var(--sans)" }}>Notes</div><div style={{ font: "400 12.5px/1.5 var(--sans)", color: "var(--ink3)", marginTop: 5 }}>Kept in this browser for this consultation. Pinned evidence keeps its citation.</div></div>
                      {notes.length === 0 && <div className="empty-box">Use “Add to note” on an answer or source to pin it here.</div>}
                      {notes.map((n) => (
                        <div className="card" key={n.key}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}><span className="tg">{n.tag}</span>
                            <button aria-label="Remove" onClick={() => saveNotes(notes.filter((x) => x.key !== n.key))} style={{ marginLeft: "auto", border: 0, background: "transparent", color: "var(--ink3)", cursor: "pointer", fontSize: 16 }}>×</button></div>
                          <div className="tx">{n.text}</div>
                        </div>
                      ))}
                      <textarea className="note-area" rows={6} placeholder="Your notes…" value={noteText}
                        onChange={(e) => { setNoteText(e.target.value); try { localStorage.setItem(`beyonddoc-notetext-${active.session_id}`, e.target.value); } catch { /* ignore */ } }} />
                      <div style={{ display: "flex", gap: 8, marginTop: "auto" }}>
                        <button className="btn" style={{ flex: 1, justifyContent: "center" }} onClick={() => copy("note", [...notes.map((n) => `${n.tag}\n${n.text}`), noteText].filter(Boolean).join("\n\n"))}>{copied === "note" ? "Copied" : "Copy note"}</button>
                      </div>
                    </div>
                  )}
                </div>
              </aside>
            )}
          </div>
        )}

        {/* ---------------- settings ---------------- */}
        {screen === "settings" && <Settings prefs={prefs} setPrefs={setPrefs} info={info} me={me} toggleSource={toggleSource} />}
      </main>
    </div>
  );
}

function transcript(s: Session, messages: Message[]): string {
  const lines = [`# ${s.title}`, ""];
  for (const m of messages) {
    lines.push(m.role === "user" ? `**Question:** ${m.content}` : `**Answer:** ${m.content}`);
    if (m.role === "assistant") {
      for (const sid of m.cited || []) {
        const p = m.sources?.[sNum(sid) - 1];
        if (p) lines.push(`  [${sNum(sid)}] ${p.title} — ${p.section_path || ""} ${p.source_reference}`);
      }
    }
    lines.push("");
  }
  lines.push("Informational demo over public documents. Not medical advice.");
  return lines.join("\n");
}

// ---------- pipeline ----------
function Pipeline({ stages, detailOverride }: { stages: Record<string, StageEvent>; detailOverride?: Record<string, string> }) {
  const activeIdx = STAGES.findIndex((s) => stages[s.id]?.status !== "done");
  return (
    <div className="pipe">
      {STAGES.map((s, i) => {
        const st = stages[s.id];
        const done = st?.status === "done";
        const isActive = !done && (st?.status === "active" || i === activeIdx);
        return (
          <div key={s.id} className={"stage" + (!done && !isActive ? " pending" : "")}>
            <div className="ic">{done ? <Check /> : isActive ? <span className="pulse" /> : <span className="pend" />}</div>
            <span className="lb">{s.label}</span>
            <span className="dt">{detailOverride?.[s.id] ?? st?.detail ?? ""}</span>
          </div>
        );
      })}
    </div>
  );
}

// ---------- assistant message ----------
function BotMessage(props: {
  m: Message; kbOn: number; chunks?: number | null; activeSrc: number | null; trace: boolean;
  onTrace: () => void; onCite: (n: number) => void; copied: boolean; onCopy: () => void; onPin: () => void; onRetrieve: () => void; onFocus: () => void;
}) {
  const { m } = props;
  const t = m.timings_ms || {};
  const cited = m.cited || [];
  const summary = `${props.kbOn} sources · ${(props.chunks || 0).toLocaleString()} passages · ${m.sources?.length ?? 0} retrieved · ${cited.length} cited · ${fmtS(t.total_ms)}`;
  const parts = m.content.split(/(\[S\d+\])/g);
  const done: Record<string, StageEvent> = Object.fromEntries(STAGES.map((s) => [s.id, { stage: s.id, status: "done", detail: "" }]));
  const details = {
    understand: `${m.query && m.query !== "" ? "query: " + (m.query.length > 42 ? m.query.slice(0, 41) + "…" : m.query) : ""} · ${fmtS(t.understand_ms) || "0.0s"}`,
    search: `${m.sources?.length ?? 0} passages · ${m.mode} · ${fmtS(t.retrieval_total_ms)}`,
    compose: `${m.abstained ? "abstained" : cited.length + " cited"} · ${fmtS(t.llm_ms)}`,
  };
  return (
    <div className="bot" onMouseEnter={props.onFocus}>
      <div className="bot-head">
        <Mark small /><Word size={17} />
        <button className="trace" onClick={props.onTrace}><span>{summary}</span><b>{props.trace ? "Hide" : "Trace"}</b></button>
      </div>
      {props.trace && <Pipeline stages={done} detailOverride={details} />}
      {m.abstained ? (
        <>
          <div className="tags"><span className="tag warn">Not enough evidence in the indexed documents</span></div>
          <p className="answer" style={{ color: "var(--ink2)" }}>{m.content}</p>
        </>
      ) : (
        <>
          <p className="answer">
            {parts.map((part, i) => {
              const hit = /^\[S(\d+)\]$/.exec(part);
              if (!hit) return <Fragment key={i}>{part}</Fragment>;
              const n = Number(hit[1]);
              return <button key={i} className={"cite" + (props.activeSrc === n ? " on" : "")} onClick={() => props.onCite(n)}>{n}</button>;
            })}
          </p>
          <div className="tags">
            <span className="tag accent"><span className="bars"><i /><i /><i /></span>{cited.length} cited passage{cited.length === 1 ? "" : "s"}</span>
            <span className="tag">{[...new Set((m.sources || []).filter((_, i) => cited.includes(`S${i + 1}`)).map((p) => TYPE[p.source]?.chip || p.source))].join(" · ") || "—"}</span>
            <span className="tag">{m.model}</span>
          </div>
        </>
      )}
      <div className="acts">
        <button className="act" onClick={props.onCopy}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a1 1 0 011-1h9" /></svg>{props.copied ? "Copied" : "Copy"}
        </button>
        <button className="act" onClick={props.onPin}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 4l6 6-4 1-4 4 1 4-2 1-4-4-4 4M9 11l4 4" /></svg>Add to note
        </button>
        <button className="act" onClick={props.onRetrieve}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 11a8 8 0 10-2.3 5.7M20 5v6h-6" /></svg>Re-retrieve
        </button>
        <span className="hint">Verify against official labeling</span>
      </div>
    </div>
  );
}

// ---------- sources ----------
function SourcesPanel({ passages, cited, k, activeSrc, setActiveSrc, onPin }: {
  passages: Passage[]; cited: string[]; k: number; activeSrc: number | null; setActiveSrc: (n: number | null) => void; onPin: (p: Passage, n: number) => void;
}) {
  if (!passages.length) return <div className="pad"><div className="empty-box">Sources for the selected answer appear here.</div></div>;
  const max = Math.max(...passages.map((p) => p.score)) || 1;
  return (
    <div className="pad">
      <div className="mono-label" style={{ display: "flex", justifyContent: "space-between", padding: "2px 2px 4px" }}>
        <span>{cited.length} cited · {passages.length} retrieved (k={k})</span><span>Relevance</span>
      </div>
      {passages.map((p, i) => {
        const n = i + 1, on = activeSrc === n, isCited = cited.includes(`S${n}`);
        const rel = p.score / max;
        return (
          <div key={p.chunk_id} className={"src" + (on ? " on" : "") + (cited.length && !isCited ? " uncited" : "")} onClick={() => setActiveSrc(on ? null : n)}>
            <div className="top">
              <span className="num">{n}</span>
              <span className="mono-label" style={{ letterSpacing: ".07em" }}>{TYPE[p.source]?.type || p.source}{isCited ? " · cited" : ""}</span>
              <span className="rel"><span className="bar"><i style={{ width: `${Math.round(rel * 100)}%` }} /></span><span className="sc">{rel.toFixed(2)}</span></span>
            </div>
            <div className="ti">{p.title}</div>
            <div className="pub">{p.section_path || "—"}</div>
            {on && (
              <div className="more" onClick={(e) => e.stopPropagation()}>
                <div className="ex">“{p.text.length > 520 ? p.text.slice(0, 520).trimEnd() + "…" : p.text}”</div>
                <div className="pub">{Object.entries(p.component_ranks).map(([r, v]) => `${r} #${v}`).join(" · ")}{p.token_count ? ` · ${p.token_count} tokens` : ""}</div>
                <div style={{ display: "flex", gap: 6 }}>
                  <button className="sbtn" onClick={() => onPin(p, n)}>Add to note</button>
                  <a className="sbtn" href={p.source_reference} target="_blank" rel="noopener noreferrer" style={{ textDecoration: "none" }}>Open source ↗</a>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------- network: question → medicines → cited passages (from real metadata) ----------
function NetworkPanel({ passages, cited, activeSrc, setActiveSrc }: {
  passages: Passage[]; cited: string[]; activeSrc: number | null; setActiveSrc: (n: number | null) => void;
}) {
  const [concept, setConcept] = useState<string | null>(null);
  const nodes = passages.map((p, i) => ({ n: i + 1, p })).filter(({ n }) => !cited.length || cited.includes(`S${n}`)).slice(0, 7);
  const meds: string[] = [];
  for (const { p } of nodes) for (const name of p.medicine_names.slice(0, 2)) {
    const key = name.toLowerCase();
    if (!meds.some((m) => m.toLowerCase() === key) && meds.length < 5) meds.push(name);
  }
  if (!nodes.length) return <div className="pad"><div className="empty-box">The evidence network appears after an answer.</div></div>;
  const C = 180, place = (i: number, total: number, r: number, off = -Math.PI / 2) => {
    const a = off + (2 * Math.PI * i) / Math.max(total, 1);
    return [C + r * Math.cos(a), C + r * Math.sin(a)];
  };
  const medPos = meds.map((_, i) => place(i, meds.length, 82));
  const srcPos = nodes.map((_, i) => place(i, nodes.length, 150, -Math.PI / 2 + Math.PI / nodes.length));
  const links = (p: Passage, m: string) => p.medicine_names.some((x) => x.toLowerCase() === m.toLowerCase()) || p.title.toLowerCase().includes(m.toLowerCase());
  const hlSrc = (n: number) => activeSrc === n || (concept !== null && links(nodes.find((x) => x.n === n)!.p, concept));
  const hlMed = (m: string) => concept === m || (activeSrc !== null && !!nodes.find((x) => x.n === activeSrc) && links(nodes.find((x) => x.n === activeSrc)!.p, m));
  const any = activeSrc !== null || concept !== null;
  const focus = activeSrc !== null ? nodes.find((x) => x.n === activeSrc) : null;
  return (
    <div className="pad" style={{ gap: 14 }}>
      <div><div style={{ font: "500 14px/1.2 var(--sans)" }}>Evidence network</div>
        <div style={{ font: "400 12.5px/1.5 var(--sans)", color: "var(--ink3)", marginTop: 5 }}>Medicines named in the cited passages, linked to the passages that mention them. Select a node to trace its evidence.</div></div>
      <div className="graph">
        <svg viewBox="0 0 360 360">
          {meds.map((m, i) => <line key={"q" + m} x1={C} y1={C} x2={medPos[i][0]} y2={medPos[i][1]} stroke={hlMed(m) ? "var(--accent)" : "var(--line2)"} strokeWidth={hlMed(m) ? 1.6 : 1} opacity={!any || hlMed(m) ? 1 : 0.35} />)}
          {meds.map((m, i) => nodes.map(({ n, p }, j) => links(p, m) && (
            <line key={m + n} x1={medPos[i][0]} y1={medPos[i][1]} x2={srcPos[j][0]} y2={srcPos[j][1]} stroke={hlMed(m) && hlSrc(n) ? "var(--accent)" : "var(--line2)"} strokeWidth={hlMed(m) && hlSrc(n) ? 1.6 : 1} opacity={!any || (hlMed(m) && hlSrc(n)) ? 1 : 0.35} />
          )))}
          {nodes.map(({ n }, j) => (
            <g key={"s" + n} style={{ cursor: "pointer" }} onClick={() => { setConcept(null); setActiveSrc(activeSrc === n ? null : n); }} opacity={!any || hlSrc(n) ? 1 : 0.3}>
              <rect x={srcPos[j][0] - 11} y={srcPos[j][1] - 11} width="22" height="22" rx="6" fill={hlSrc(n) ? "var(--accent)" : "var(--bg2)"} stroke="var(--line2)" />
              <text x={srcPos[j][0]} y={srcPos[j][1] + 3.5} textAnchor="middle" style={{ font: "600 10px var(--mono)", fill: hlSrc(n) ? "var(--accentInk)" : "var(--ink)" }}>{n}</text>
            </g>
          ))}
          {meds.map((m, i) => (
            <g key={"m" + m} style={{ cursor: "pointer" }} onClick={() => { setActiveSrc(null); setConcept(concept === m ? null : m); }} opacity={!any || hlMed(m) ? 1 : 0.3}>
              <circle cx={medPos[i][0]} cy={medPos[i][1]} r="14" fill="transparent" />
              <circle cx={medPos[i][0]} cy={medPos[i][1]} r="6.5" fill={hlMed(m) ? "var(--accent)" : "var(--bg)"} stroke="var(--ink2)" strokeWidth="1.5" />
              <text x={medPos[i][0]} y={medPos[i][1] - 12} textAnchor="middle" style={{ font: "500 11px var(--sans)", fill: "var(--ink)" }}>{m.length > 22 ? m.slice(0, 21) + "…" : m}</text>
            </g>
          ))}
          <g style={{ cursor: "pointer" }} onClick={() => { setConcept(null); setActiveSrc(null); }}>
            <circle cx={C} cy={C} r="26" fill="var(--accentSoft)" /><circle cx={C} cy={C} r="16" fill="var(--ink)" />
            <text x={C} y={C + 3.5} textAnchor="middle" style={{ font: "600 10px var(--mono)", fill: "var(--bg)" }}>Q</text>
          </g>
        </svg>
      </div>
      <div className="legend">
        <span><i style={{ width: 10, height: 10, borderRadius: "50%", background: "var(--ink)", display: "inline-block" }} />Question</span>
        <span><i style={{ width: 9, height: 9, borderRadius: "50%", border: "1.5px solid var(--ink2)", display: "inline-block" }} />Medicine</span>
        <span><i style={{ width: 10, height: 10, borderRadius: 3, border: "1px solid var(--line2)", background: "var(--bg2)", display: "inline-block" }} />Source passage</span>
      </div>
      <div className="card">
        <div className="mono-label">{focus ? `Source ${focus.n} · ${TYPE[focus.p.source]?.type}` : concept ? "Medicine" : "Overview"}</div>
        <div style={{ font: "500 14px/1.35 var(--sans)" }}>{focus ? focus.p.title : concept ?? `${meds.length} medicines · ${nodes.length} passages`}</div>
        <div className="tx">{focus ? focus.p.section_path : concept ? `Mentioned in passages ${nodes.filter(({ p }) => links(p, concept)).map(({ n }) => `[${n}]`).join(" ")}` : "Built from the metadata of the passages cited in this answer."}</div>
      </div>
    </div>
  );
}

// ---------- settings ----------
function Settings({ prefs, setPrefs, info, me, toggleSource }: {
  prefs: Prefs; setPrefs: (p: Partial<Prefs>) => void; info: Info | null; me: { user: string; retention_days: number } | null; toggleSource: (id: SourceId) => void;
}) {
  const [t, setT] = useState<"appearance" | "retrieval" | "privacy">("appearance");
  const Seg = <T extends string>({ opts, cur, set }: { opts: T[]; cur: T; set: (v: T) => void }) => (
    <div className="seg">{opts.map((o) => <button key={o} className={o === cur ? "on" : ""} onClick={() => set(o)}>{o}</button>)}</div>
  );
  const preview = (bg: string, side: string, ink: string, accent: string, line: string) => (
    <div className="pv" style={{ background: bg, gridTemplateColumns: "36px 1fr" }}>
      <div style={{ background: side, borderRight: `1px solid ${line}`, padding: "12px 7px", display: "flex", flexDirection: "column", gap: 6 }}>
        {[0.2, 0.1, 0.1].map((o, i) => <div key={i} style={{ height: 4, borderRadius: 2, background: ink, opacity: o * 2 }} />)}</div>
      <div style={{ padding: "16px 14px 10px", display: "flex", flexDirection: "column", gap: 7 }}>
        <div style={{ font: "400 17px/1 var(--serif)", color: ink }}>Beyond<span style={{ color: accent, fontStyle: "italic" }}>Doc</span></div>
        <div style={{ width: "80%", height: 4, borderRadius: 2, background: line }} /><div style={{ width: "56%", height: 4, borderRadius: 2, background: line }} />
        <div style={{ marginTop: "auto", height: 16, borderRadius: 5, border: `1px solid ${line}` }} /></div>
    </div>
  );
  return (
    <div className="settings"><div className="settings-inner">
      <div><h2>Settings</h2><p className="sub">Preferences are saved in this browser.</p></div>
      <div className="stabs">{(["appearance", "retrieval", "privacy"] as const).map((id) => <button key={id} className={t === id ? "on" : ""} onClick={() => setT(id)}>{id === "privacy" ? "Privacy & data" : id[0].toUpperCase() + id.slice(1)}</button>)}</div>

      {t === "appearance" && (
        <div>
          <div style={{ paddingBottom: 14 }}><div className="srow" style={{ border: 0, padding: 0, display: "block" }}><div className="k">Theme</div><div className="d">White is the default; Black is tuned for low-light wards.</div></div></div>
          <div className="themes">
            {([["light", "White", preview("#ffffff", "#f4f4f2", "#0c0c0e", "oklch(0.5 0.115 195)", "rgba(0,0,0,.12)")],
               ["dark", "Black", preview("#000000", "#0d0d0e", "#f5f5f6", "oklch(0.8 0.11 195)", "rgba(255,255,255,.16)")],
               ["system", "Match system", preview("linear-gradient(90deg,#fff 50%,#000 50%)", "#f4f4f2", "#0c0c0e", "oklch(0.5 0.115 195)", "rgba(127,127,127,.3)")]] as const).map(([id, label, pv]) => (
              <button key={id} className={"theme" + (prefs.theme === id ? " on" : "")} onClick={() => setPrefs({ theme: id })}>{pv}<div className="lb"><span className="ring"><i /></span>{label}</div></button>
            ))}
          </div>
          <div className="srow" style={{ marginTop: 18, borderTop: "1px solid var(--line)" }}>
            <div><div className="k">Answer text size</div><div className="d">Applies to answers in the chat.</div></div>
            <Seg opts={["Small", "Default", "Large"] as Size[]} cur={prefs.size} set={(v) => setPrefs({ size: v })} />
          </div>
        </div>
      )}

      {t === "retrieval" && (
        <div>
          <div className="mono-label" style={{ paddingBottom: 6 }}>Knowledge bases</div>
          {SOURCES.map((s) => (
            <div className="srow" key={s.id}>
              <div><div className="k">{s.label}</div><div className="d mono">{(info?.chunks_by_source[s.id] ?? 0).toLocaleString()} passages · Batch 1</div></div>
              <button className={"toggle" + (prefs.sources.includes(s.id) ? " on" : "")} aria-label={`Toggle ${s.label}`} onClick={() => toggleSource(s.id)}><i /></button>
            </div>
          ))}
          <div className="mono-label" style={{ padding: "32px 0 6px" }}>Answering</div>
          <div className="srow" style={{ gridTemplateColumns: "minmax(0,1fr) 260px" }}>
            <div><div className="k">Retrieval depth</div><div className="d">Passages sent to the answer model. Higher is more thorough, slower.</div></div>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <input type="range" min={4} max={20} value={prefs.k} onChange={(e) => setPrefs({ k: Number(e.target.value) })} style={{ flex: 1, accentColor: "var(--accent)" }} />
              <span style={{ width: 44, textAlign: "right", font: "500 12.5px/1 var(--mono)" }}>k={prefs.k}</span>
            </div>
          </div>
          <div className="srow">
            <div><div className="k">Retrieval method</div><div className="d">Hybrid fuses BM25 and dense search (RRF). All three are provisional demo options.</div></div>
            <Seg opts={["hybrid", "dense", "bm25"] as Mode[]} cur={prefs.mode} set={(v) => setPrefs({ mode: v })} />
          </div>
          <div className="srow">
            <div><div className="k">Models</div><div className="d mono">answers {info?.llm_model ?? "?"} · embeddings {info?.embedding_model ?? "?"}</div></div><span />
          </div>
        </div>
      )}

      {t === "privacy" && (
        <div>
          <div className="srow"><div><div className="k">Conversation retention</div><div className="d">Chats are stored on this demo server (ClickHouse) and deleted automatically after this period.</div></div>
            <span className="tiny-mono" style={{ fontSize: 13 }}>{me ? `${me.retention_days} days` : "—"}</span></div>
          <div className="srow"><div><div className="k">Short-term memory</div><div className="d">The last few turns of each chat are cached (Redis, 24 h) to understand follow-up questions.</div></div><span /></div>
          <div className="srow"><div><div className="k">Identity</div><div className="d">An anonymous id in a browser cookie; no account, no personal data. Do not enter patient information.</div></div>
            <span className="tiny-mono">{me ? `id ${me.user}` : ""}</span></div>
          <div className="notice"><b>DEMO</b><span>Informational demo over public documents (DailyMed, PMC, LiverTox, FDA). Not medical advice; not a diagnosis or treatment system.</span></div>
        </div>
      )}
    </div></div>
  );
}
