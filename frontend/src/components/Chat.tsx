import { useEffect, useRef, useState } from "react";
import { ask } from "../api";
import type { ChatMessage } from "../types";
import Message from "./Message";

const STARTERS = [
  "What does Admire do at Microsoft?",
  "What are his projects?",
  "Where did he study?",
  "What's he interested in outside work?",
  "Can I book a call with him?",
];

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    setInput("");
    const history = messages; // prior completed turns, for follow-up context
    // The empty assistant message fills in as the reply streams
    setMessages((m) => [...m, { role: "user", content: question }, { role: "assistant", content: "" }]);
    setBusy(true);
    const updateReply = (change: (reply: ChatMessage) => ChatMessage) =>
      setMessages((m) => [...m.slice(0, -1), change(m[m.length - 1])]);
    try {
      const { booking, sources } = await ask(question, history, (text) =>
        updateReply((r) => ({ ...r, content: r.content + text })),
      );
      updateReply((r) => ({ ...r, content: r.content.trim(), booking, sources }));
    } catch (e) {
      updateReply((r) => ({ ...r, content: `${r.content}\n\n⚠️ ${String(e)}`.trim() }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex h-full w-full max-w-2xl flex-col">
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="mt-10 space-y-5 text-center">
            <p className="text-slate-400">
              Ask me anything about Admire — his work, projects, background, or interests.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {STARTERS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-full border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-300 hover:border-sky-500 hover:text-sky-300"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) =>
          // Hold the empty reply back until its first words arrive
          m.content ? <Message key={i} message={m} /> : null,
        )}
        {busy && !messages[messages.length - 1]?.content && (
          <div className="pl-1 text-sm text-slate-500">Thinking…</div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-slate-800 p-3">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send(input)}
            placeholder="Ask about Admire…"
            className="flex-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-slate-100 placeholder-slate-500 focus:border-sky-500 focus:outline-none"
          />
          <button
            onClick={() => send(input)}
            disabled={busy}
            className="rounded-lg bg-sky-600 px-4 py-2 font-medium text-white hover:bg-sky-500 disabled:opacity-50"
          >
            {busy ? "…" : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}
