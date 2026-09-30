"use client";
import { useState, useEffect, useRef } from "react";
import styles from "./page.module.css";

// SVG Icons
const SendIcon = () => (
  <svg viewBox="0 0 24 24" className={styles.sendIcon}>
    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
  </svg>
);
const FileIcon = () => (
  <svg viewBox="0 0 24 24" className={styles.corpusIcon}>
    <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" />
  </svg>
);
const ResetIcon = () => (
  <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
    <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" />
  </svg>
);
const ChevronDown = () => (
  <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
    <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z" />
  </svg>
);
const ChevronUp = () => (
  <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
    <path d="M7.41 15.41L12 10.83l4.59 4.58L18 14l-6-6-6 6 1.41 1.41z" />
  </svg>
);

const PRESETS = [
  "Troubleshoot CrashLoopBackOff",
  "Difference between DaemonSet and Deployment",
  "Pod restartPolicy explained",
  "NSA Container Hardening best practices"
];

const CORPUS_FILES = [
  "01_Overview.pdf",
  "02_Architecture.pdf",
  "03_Containers.pdf",
  "04_Workloads.pdf",
  "05_Networking.pdf",
  "06_Storage.pdf",
  "07_Configuration.pdf",
  "08_Security.pdf",
  "09_Policies.pdf",
  "10_Scheduling.pdf",
  "11_Administration.pdf",
  "12_Windows_Nodes.pdf",
  "13_Extensions.pdf",
  "CIS_Kubernetes_V1.20_Benchmark_v1.0.0_PDF.pdf",
  "CNCF_Operator_WhitePaper.pdf",
  "CNCF_cloud-native-security-whitepaper-May2022-v2.pdf",
  "NSA_Kubernetes_Hardening_Guidance_1.0.pdf",
  "platforms-def-v1.0.pdf"
];

export default function Home() {
  const [loadingApp, setLoadingApp] = useState(true);
  const [messages, setMessages] = useState<{ role: "user" | "bot", text: string, data?: any }[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [inspectorData, setInspectorData] = useState<any>(null);
  const [expandedChunks, setExpandedChunks] = useState<number[]>([]);
  const [backendReady, setBackendReady] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Show beautiful loading screen for 3 seconds
    const timer = setTimeout(() => {
      setLoadingApp(false);
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const checkHealth = async () => {
      // Don't poll if we already know it's ready (though interval is cleared, this is a safeguard)
      if (backendReady) return;

      try {
        const res = await fetch("http://127.0.0.1:8000/health");
        if (res.ok) {
          setBackendReady(true);
          if (intervalId) clearInterval(intervalId);
        } else {
          setBackendReady(false);
        }
      } catch (e) {
        setBackendReady(false);
      }
    };

    // Initial check
    checkHealth();

    // Set polling every 5 seconds
    intervalId = setInterval(() => {
      checkHealth();
    }, 5000);

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [backendReady]);

  const handleSend = async (overrideQuery?: string) => {
    const query = overrideQuery || input;
    if (!query.trim()) return;

    setMessages((prev) => [...prev, { role: "user", text: query }]);
    if (!overrideQuery) setInput("");
    setIsTyping(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/v1/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          top_n: 5,
          verify_grounding: true
        }),
      });

      if (!res.ok) throw new Error("Network response was not ok");

      const data = await res.json();
      setMessages((prev) => [...prev, { role: "bot", text: data.answer, data }]);
      setInspectorData(data); // Auto-focus the latest response
      setExpandedChunks([]); // Reset accordions
    } catch (error) {
      setMessages((prev) => [...prev, { role: "bot", text: "Error connecting to backend." }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleReset = () => {
    setMessages([]);
    setInspectorData(null);
  };

  const toggleChunk = (idx: number) => {
    setExpandedChunks(prev =>
      prev.includes(idx) ? prev.filter(i => i !== idx) : [...prev, idx]
    );
  };

  const getSourceFilename = (sourcePath: string) => {
    let name = sourcePath;
    if (name.includes("\\")) name = name.split("\\").pop() || name;
    if (name.includes("/")) name = name.split("/").pop() || name;
    return name;
  };

  if (loadingApp) {
    return (
      <div className={styles.container}>
        <div className={styles.loadingScreen}>
          <div className={styles.loaderShape}></div>
          <div className={styles.loadingTextContainer}>
            <span>L</span><span>O</span><span>A</span><span>D</span><span>I</span><span>N</span><span>G</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Left Panel */}
      <div className={styles.leftPanel}>
        <h2 className={styles.panelTitle}>Quick Tools & Reference</h2>

        <h3 className={styles.panelTitle} style={{ marginTop: "16px" }}>Preset Prompt Library</h3>
        {PRESETS.map((p, i) => (
          <div key={i} className={styles.presetChip} onClick={() => setInput(p)}>
            {p}
          </div>
        ))}

        <h3 className={styles.panelTitle} style={{ marginTop: "24px" }}>Corpus Scope</h3>
        <div className={styles.corpusList}>
          {CORPUS_FILES.map((f, i) => (
            <div key={i} className={styles.corpusItem}>
              <FileIcon />
              <span>{f}</span>
            </div>
          ))}
        </div>

        <button className={styles.resetBtn} onClick={handleReset}>
          <ResetIcon /> Reset View
        </button>
      </div>

      {/* Center Panel (Chat) */}
      <div className={styles.chatContainer}>
        <div className={styles.header}>
          <h1 className={styles.headerTitle}>
            K8s Assistant
            {!backendReady && (
              <span style={{ fontSize: "14px", color: "var(--error)", marginLeft: "12px", fontWeight: "normal", verticalAlign: "middle" }}>
                ● Backend Initializing...
              </span>
            )}
          </h1>
        </div>

        <div className={styles.chatHistory}>
          {messages.length === 0 && (
            <div style={{ textAlign: "center", color: "var(--text-secondary)", marginTop: "40px" }}>
              <p>Welcome to Kubernetes Technical Assistant.</p>
              <p style={{ fontSize: "14px", marginTop: "10px" }}>Select a prompt from the left or type below.</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`${styles.messageRow} ${msg.role === "user" ? styles.userRow : styles.botRow}`}>
              <div
                className={`${styles.messageBubble} ${msg.role === "user" ? styles.userMessage : styles.botMessage}`}
                onClick={() => msg.role === "bot" && msg.data && setInspectorData(msg.data)}
                style={msg.role === "bot" && msg.data ? { cursor: "pointer" } : {}}
              >
                {msg.text}
              </div>
            </div>
          ))}

          {isTyping && (
            <div className={`${styles.messageRow} ${styles.botRow}`}>
              <div className={styles.chatSpinnerContainer}>
                <div className={styles.dotPulse}>
                  <div></div><div></div><div></div>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className={styles.inputContainer}>
          <input
            type="text"
            className={styles.inputField}
            placeholder={backendReady ? "Ask about Kubernetes..." : "Waiting for backend to be ready..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSend(); }}
            disabled={isTyping || !backendReady}
          />
          <button className={styles.sendButton} onClick={() => handleSend()} disabled={isTyping || !input.trim() || !backendReady}>
            <SendIcon />
          </button>
        </div>
      </div>

      {/* Right Panel (Live Pipeline Inspector) */}
      <div className={styles.rightPanel}>
        <h2 className={styles.panelTitle}>Live Pipeline Inspector</h2>

        {!inspectorData ? (
          <div className={styles.inspectorPlaceholder}>
            <svg viewBox="0 0 24 24" className={styles.inspectorPlaceholderIcon} fill="currentColor">
              <path d="M19.5 3.5L18 2l-1.5 1.5L15 2l-1.5 1.5L12 2l-1.5 1.5L9 2 7.5 3.5 6 2v14H3v3c0 1.66 1.34 3 3 3h12c1.66 0 3-1.34 3-3V2l-1.5 1.5zM19 19c0 .55-.45 1-1 1H6c-.55 0-1-.45-1-1v-1h14v1zm0-3H5V4.08L6 5.1l1.5-1.5L9 5.1l1.5-1.5L12 5.1l1.5-1.5L15 5.1l1.5-1.5L18 5.1l1-1.02V16z" />
            </svg>
            <p>Select a bot response to inspect its RAG pipeline metrics.</p>
          </div>
        ) : (
          <>
            {/* 1. Verification & Safety */}
            {inspectorData.verification && (
              <div className={styles.inspectorCard}>
                <div className={styles.inspectorCardHeader}>VERIFICATION & SAFETY</div>
                <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                  <div className={`${styles.badge} ${inspectorData.verification.is_grounded ? styles.badgeGrounded : styles.badgeUngrounded}`}>
                    {inspectorData.verification.is_grounded ? "✓ Grounded" : "⚠ Not Fully Grounded"}
                  </div>
                  <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                    Hallucination Score: {inspectorData.verification.hallucination_score}
                  </span>
                </div>
                <div className={styles.verifDesc}>
                  "{inspectorData.verification.explanation}"
                </div>
              </div>
            )}

            {/* 2. Performance Breakdown */}
            {inspectorData.metrics && (
              <div className={styles.inspectorCard}>
                <div className={styles.inspectorCardHeader}>PERFORMANCE BREAKDOWN</div>

                <div className={styles.perfRow}>
                  <span>Retrieval & Rerank</span>
                  <span>{(inspectorData.metrics.retrieval_rerank_ms / 1000).toFixed(2)}s</span>
                </div>
                <div className={styles.perfRow}>
                  <span>Generation</span>
                  <span>{(inspectorData.metrics.generation_ms / 1000).toFixed(2)}s</span>
                </div>
                <div className={styles.perfRow}>
                  <span>Verification</span>
                  <span>{(inspectorData.metrics.verification_ms / 1000).toFixed(2)}s</span>
                </div>
                <div className={styles.perfRow} style={{ fontWeight: 700, marginTop: "8px", paddingTop: "8px", borderTop: "1px solid var(--border-color)" }}>
                  <span style={{ color: "var(--text-primary)" }}>Total Latency</span>
                  <span>{(inspectorData.metrics.total_latency_ms / 1000).toFixed(2)}s</span>
                </div>

                {/* Segmented Bar */}
                {(() => {
                  const m = inspectorData.metrics;
                  const total = m.total_latency_ms || 1;
                  const retPct = (m.retrieval_rerank_ms / total) * 100;
                  const genPct = (m.generation_ms / total) * 100;
                  const verPct = (m.verification_ms / total) * 100;
                  return (
                    <div className={styles.perfBarContainer}>
                      <div className={`${styles.perfSegment} ${styles.perfSegRetrieve}`} style={{ width: `${retPct}%` }}></div>
                      <div className={`${styles.perfSegment} ${styles.perfSegGenerate}`} style={{ width: `${genPct}%` }}></div>
                      <div className={`${styles.perfSegment} ${styles.perfSegVerify}`} style={{ width: `${verPct}%` }}></div>
                    </div>
                  );
                })()}

                <div className={styles.perfLegend}>
                  <div className={styles.legendItem}><div className={`${styles.legendDot} ${styles.perfSegRetrieve}`}></div> Retrieval</div>
                  <div className={styles.legendItem}><div className={`${styles.legendDot} ${styles.perfSegGenerate}`}></div> Gen</div>
                  <div className={styles.legendItem}><div className={`${styles.legendDot} ${styles.perfSegVerify}`}></div> Verif</div>
                </div>
              </div>
            )}

            {/* 3. Retrieved Context */}
            {inspectorData.retrieved_chunks && inspectorData.retrieved_chunks.length > 0 && (
              <div className={styles.inspectorCard} style={{ marginBottom: 0 }}>
                <div className={styles.inspectorCardHeader}>
                  RETRIEVED CONTEXT ({inspectorData.retrieved_chunks.length} Chunks)
                </div>
                <div style={{ display: "flex", flexDirection: "column" }}>
                  {inspectorData.retrieved_chunks.map((chunk: any, i: number) => {
                    const isExpanded = expandedChunks.includes(i);
                    const filename = getSourceFilename(chunk.source);
                    return (
                      <div key={i} className={styles.chunkAccordion}>
                        <div className={styles.chunkHeader} onClick={() => toggleChunk(i)}>
                          <div style={{ display: "flex", alignItems: "center", gap: "8px", overflow: "hidden" }}>
                            <span className={styles.chunkSource}>[{filename}]</span>
                            <span style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", flex: 1 }}>
                              {chunk.section || "Excerpt"}
                            </span>
                          </div>
                          {isExpanded ? <ChevronUp /> : <ChevronDown />}
                        </div>
                        {isExpanded && (
                          <div className={styles.chunkContent}>
                            {chunk.content_preview}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
