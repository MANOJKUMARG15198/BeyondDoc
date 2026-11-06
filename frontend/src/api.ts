// Typed client for the FastAPI chat endpoints, including the SSE message stream.

export type SourceId = "dailymed" | "pmc_oa" | "nih_livertox" | "fda_drugsatfda";
export type Mode = "hybrid" | "dense" | "bm25";

export interface Passage {
  rank: number;
  chunk_id: string;
  document_id: string;
  title: string;
  section_path: string | null;
  section_role: string | null;
  source: SourceId;
  source_reference: string;
  medicine_names: string[];
  kind: string | null;
  token_count: number | null;
  score: number;
  component_ranks: Record<string, number>;
  text: string;
}

export interface Message {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  abstained?: boolean;
  abstain_reason?: string | null;
  cited?: string[];
  sources?: Passage[];
  query?: string;
  mode?: Mode;
  k?: number;
  model?: string;
  timings_ms?: Record<string, number>;
}

export interface Session {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface Info {
  chunks: number | null;
  chunks_by_source: Record<string, number>;
  embedding_model: string;
  llm_configured: boolean;
  llm_model: string;
  disclaimer: string;
}

export interface StageEvent {
  stage: "understand" | "search" | "compose";
  status: "active" | "done";
  detail: string;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { credentials: "same-origin", ...init });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : r.statusText);
  }
  return r.status === 204 ? (undefined as T) : r.json();
}

const post = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  info: () => json<Info>("info"),
  me: () => json<{ user: string; retention_days: number }>("api/me"),
  sessions: () => json<{ sessions: Session[] }>("api/sessions").then((r) => r.sessions),
  createSession: () => json<Session>("api/sessions", post({})),
  session: (id: string) => json<{ session: Session; messages: Message[] }>(`api/sessions/${id}`),
  rename: (id: string, title: string) =>
    json<Session>(`api/sessions/${id}`, { ...post({ title }), method: "PATCH" }),
  remove: (id: string) => json<void>(`api/sessions/${id}`, { method: "DELETE" }),
};

export interface StreamHandlers {
  onUser?: (m: Message) => void;
  onStage?: (s: StageEvent) => void;
  onSources?: (d: { query: string; passages: Passage[] }) => void;
  onAnswer?: (d: { message: Message; session: Session }) => void;
  onError?: (detail: string) => void;
}

export async function sendMessage(
  sessionId: string,
  body: { content: string; k: number; mode: Mode; sources: SourceId[] | null },
  h: StreamHandlers,
): Promise<void> {
  const r = await fetch(`api/sessions/${sessionId}/messages`, { credentials: "same-origin", ...post(body) });
  if (!r.ok || !r.body) {
    const b = await r.json().catch(() => ({}));
    h.onError?.(typeof b.detail === "string" ? b.detail : `Request failed (${r.status})`);
    return;
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      const payload = data ? JSON.parse(data) : {};
      if (event === "user_message") h.onUser?.(payload);
      else if (event === "stage") h.onStage?.(payload);
      else if (event === "sources") h.onSources?.(payload);
      else if (event === "answer") h.onAnswer?.(payload);
      else if (event === "error") h.onError?.(payload.detail);
    }
  }
}
