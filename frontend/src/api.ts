import type { ChatMessage } from "./types";

export interface Done {
  booking: boolean;
  sources: string[];
}

// Ask one question. Sends the recent turns so follow-ups keep their referent.
// The reply streams back as newline-delimited JSON events: onText gets each
// piece of the answer as it's written, and the promise resolves with the final
// "done" event.
export async function ask(
  message: string,
  history: ChatMessage[],
  onText: (text: string) => void,
): Promise<Done> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // A network chunk can end partway through a line, so the last piece waits
    // for the rest of it
    const lines = buf.split("\n");
    buf = lines.pop()!;
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "delta") onText(event.text);
      else if (event.type === "done") return { booking: Boolean(event.booking), sources: event.sources ?? [] };
      else if (event.type === "error") throw new Error(event.error);
    }
  }
  throw new Error("The reply ended early");
}
