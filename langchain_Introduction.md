# LangChain — Deep‑Dive Notes (2025‑07)

> **Definition** LangChain is an open‑source Python/TypeScript framework that lets you compose Large‑Language‑Model (LLM) calls into reliable, production‑grade applications. It abstracts away vendor differences, adds orchestration primitives (chains, agents, memory), and integrates with dozens of data sources and orchestration tools.

---

## 1. Why LangChain?

| Need                       | How LangChain Helps                                                                                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLM vendor portability** | Uniform `LLM` and `ChatModel` interfaces over OpenAI, Anthropic, Mistral, Cohere, Ollama, Hugging Face, Vertex AI, Azure OpenAI, etc. Swap via one environment variable or config change. |
| **Tooling ecosystem**      | Built‑in connectors: SQL, NoSQL, Pinecone, Weaviate, Redis, Elasticsearch, Supabase, Airbyte, Apache Airflow, Streamlit, FastAPI, Ray, Dask, BentoML, AWS Lambda, …                       |
| **Complex orchestration**  | Chains, routers, tool‑calling agents, multi‑step workflows, and guardrails remove the need to write brittle glue code.                                                                    |
| **State & memory**         | Conversation history, summary memory, vector memory, key‑value memory. Lets stateless LLM APIs feel stateful.                                                                             |
| **Research patterns**      | Templates for RAG (Retrieval‑Augmented Generation), hierarchical topic‑summarisation, map‑reduce QA, self‑reflection, tool‑usage, etc.                                                    |

---

## 2. Semantic Search (Vector‑Based Retrieval)

1. **Document Loading**

   ```python
   from langchain.document_loaders import PyPDFLoader
   docs = PyPDFLoader("handbook.pdf").load()        # paginated Document objects
   ```

2. **Text Splitting** (keeps chunk size + overlap tuned for embedding model token limits)

   ```python
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   splitter = RecursiveCharacterTextSplitter(chunk_size=1_024, chunk_overlap=128)
   chunks = splitter.split_documents(docs)
   ```

3. **Embedding → Vector DB**

   ```python
   from langchain.embeddings import OpenAIEmbeddings
   from langchain.vectorstores import Pinecone
   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
   vectordb = Pinecone.from_documents(chunks, embeddings, index_name="handbook")
   ```

4. **Query‑time Retrieval**

   ```python
   query = "How do I reset a pod in Kubernetes?"
   docs = vectordb.similarity_search(query, k=5)         # returns top‑k Document objects
   ```

> **Challenge** Production orchestration: chunking strategy tuning, batch embeddings, rate‑limiting, partial re‑indexing, and hybrid search (BM25 + vectors) – LangChain’s Index, Retriever, and Asynchronous wrappers simplify these.

---

## 3. Core Components

### 3.1 Models

* **`LLM`** for completion‑style (GPT‑4o, Claude 3, PaLM 2).
* **`ChatModel`** for message‑based (ChatGPT, Gemini).
* **`Embedding`** for vector space projection.
* **Vendor‑agnostic signature:**

  ```python
  from langchain.chat_models import ChatOpenAI
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
  response = llm.invoke("Why do rockets stage‑separate?")
  ```

### 3.2 Prompts

* **Templates** with Jinja‑like variables:

  ```python
  from langchain.prompts import PromptTemplate
  tpl = PromptTemplate.from_template(
      "You are a Kubernetes SRE. Answer in bullet points:\n{question}"
  )
  ```
* **Few‑Shot** example injection: `FewShotPromptTemplate`.
* **Role‑based** system/user/assistant separation.
* **Partial application** lets you freeze certain variables.

### 3.3 Chains

> Deterministic DAGs where each node can be an LLM call, a function, or another chain.

* **`LLMChain`** – single prompt → LLM → parse.
* **`SequentialChain`** – linear pipelines.
* **`RouterChain` / `MultiPromptChain`** – conditionally branch on tool, language, topic.
* **`ParallelChain`** – fan‑out/fan‑in (e.g. ask multiple models then ensemble).

Example: *Title → Outline → Blog* pipeline:

```python
from langchain.chains import LLMChain, SequentialChain
title_chain   = LLMChain(llm=llm, prompt=tpl_title)
outline_chain = LLMChain(llm=llm, prompt=tpl_outline)
blog_chain    = LLMChain(llm=llm, prompt=tpl_blog)
pipeline = SequentialChain(chains=[title_chain, outline_chain, blog_chain],
                           input_variables=["topic"],
                           output_variables=["blog_post"])
```

### 3.4 Indexes & Retrievers

* **`VectorStoreIndexWrapper`**: unify vector DBs.
* **Hybrid retrievers**: `SelfQueryRetriever`, `MultiVectorRetriever`.
* **Doc Loaders**: PDF, DOCX, Notion, Confluence, S3, GCS, generic URL.
* **Text Splitters**: recursive, markdown‑aware, latex‑aware.

### 3.5 Memory

| Class                            | Description                                                       | Use‑Case                          |
| -------------------------------- | ----------------------------------------------------------------- | --------------------------------- |
| `ConversationBufferMemory`       | Raw message log                                                   | Small chats, debugging            |
| `ConversationBufferWindowMemory` | Sliding window of last *k* messages                               | Resource‑constrained environments |
| `ConversationSummaryMemory`      | Full history summarised by LLM after *n* turns                    | Long sessions                     |
| `VectorStoreRetrieverMemory`     | Stores past messages as vectors; retrieval by semantic similarity | Thematic recall                   |

Attach memory to any chain/agent:

```python
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm, timeout=3600)
chat = ConversationalRetrievalChain(llm=llm, retriever=vectordb.as_retriever(),
                                    memory=memory)
```

### 3.6 Agents

* **Architecture**: `Agent` (planner / decider) + **Tools** (actions) + optional `Memory`.
* **Built‑ins**: `CSVAgent`, `SQLAgent`, `PythonAgent`, `OpenAIFunctionsAgent`.
* **Reasoning pattern**: *“Thought → Action → Observation → … → Final Answer”* (aka ReAct).
* **Tool specification** via Pydantic schemas → converted to OpenAI function‑calling automatically.

Minimal example: query a Postgres DB then answer in natural language.

```python
from langchain.tools.sql_database import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
toolkit = SQLDatabaseToolkit(db=my_db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit,
                         agent_type="openai-tools",
                         verbose=True)
agent.invoke("Total cargo weight flown BOM -> DXB last month?")
```

---

## 4. What You Can Build

| Pattern                      | Real‑World Deliverable                                    |
| ---------------------------- | --------------------------------------------------------- |
| **Conversational RAG**       | Chat support trained on your knowledge base.              |
| **AI Knowledge Assistant**   | Internal dev‑ops copilot indexing runbooks, Grafana, SQL. |
| **Autonomous Agents**        | Multi‑step ticket triage / incident resolution.           |
| **Workflow Automation**      | LLM‑mediated ETL, report generation, cron triggers.       |
| **Summarization / Research** | 100‑PDF synthesis, quarterly trend digests.               |

---

## 5. Operational Best Practices

1. **Token / cost budgeting** – Use streaming, summarised memory, windowed retrieval.
2. **Determinism** – Fix `temperature=0`, seed random‑sampling models if available.
3. **Observability** – Enable LangSmith tracing or OpenTelemetry exporters.
4. **Eval** – LangChain Benchmarks, synthetic QA pairs, golden‑answer diffing.
5. **Security** – PII scrubbing in log middleware; use SOC2‑compliant vendors for embeddings.
6. **Concurrency** – Async wrappers (`.ainvoke`, `.abatch`, `asyncio.gather`) plus rate‑limit backoff.
7. **Deployment** – Package as FastAPI + Uvicorn inside Docker; horizontally scale stateless pods; mount vector DB as service.

---

### Quick Project Skeleton

```text
app/
 ├── main.py          ← FastAPI entrypoint
 ├── chains.py        ← build_chains(), build_agent()
 ├── data/
 │    ├── loaders.py  ← PDF/SQL loaders
 │    └── vectordb.py ← Pinecone/Re🡒dis config
 ├── prompts/
 │    ├── system.txt
 │    └── blog.tpl
 └── tests/
      └── eval.ipynb  ← automated eval harness
```

---

## 6. Key APIs Cheat‑Sheet

```python
# Single call (sync)
llm.invoke("Explain CRDTs in 2 sentences")

# Batch calls
llm.batch(["a", "b", "c"])

# Async
await llm.ainvoke("…")
await llm.abatch(list_of_prompts)

# Streaming
for chunk in llm.stream("Tell me a joke"): print(chunk.content, end="", flush=True)

# Guardrails (Pydantic Output Parser)
from langchain.output_parsers import PydanticOutputParser
class Answer(BaseModel):
    answer: str
    sources: list[str]
parser = PydanticOutputParser(pydantic_object=Answer)
prompt = tpl + parser.get_format_instructions()
resp = llm.invoke(prompt)
result = parser.parse(resp.content)
```

---

### Memory Selection Matrix

| Scenario                         | Recommended Memory                     |
| -------------------------------- | -------------------------------------- |
| ≤ 10 turns, low latency critical | `ConversationBufferWindowMemory(k=10)` |
| Chatbot w/ unlimited context     | `ConversationSummaryMemory`            |
| Multi‑session, thematic recall   | `VectorStoreRetrieverMemory`           |

---

## 7. Common Pitfalls & Fixes

| Pitfall                        | Symptom                            | Fix                                                                                 |
| ------------------------------ | ---------------------------------- | ----------------------------------------------------------------------------------- |
| Embedding chunk too large      | Incomplete vector index, high cost | Reduce chunk\_size (< model context) & add chunk\_overlap (10–15 %).                |
| Tool recursion / infinite loop | Agent repeats calls                | Add `max_iterations`, enable `callback_manager` warnings.                           |
| Hallucinated citations in RAG  | Fake URLs or IDs                   | Use `Refine` chain or rerank retrieved docs; filter answer tokens by doc citations. |
| API throttling                 | `RateLimitError`                   | Exponential backoff wrapper (`Retrying`, `tenacity`).                               |
| Memory bloat                   | Cost spikes over time              | Summarise old messages, evict tokens > N, or roll your own long‑term DB memory.     |

---

## 8. Further Reading / Repos

* **LangChain Docs** (Python & TS): [https://python.langchain.com](https://python.langchain.com)
* **LangSmith** (tracing + eval): [https://smith.langchain.com](https://smith.langchain.com)
* **Guides**: RAG Cookbook, Agent Cookbook, PatternHub.
* **Reference Implementations**:

  * `langchain-ai/langchain-templates` (Streamlit, FastAPI)
  * `hwchase17/langchain-hub` (prompt & chain library)
  * `imartinez/privateGPT` (local/offline RAG).

---

> **Takeaway:** Think of LangChain as *infrastructure glue* for LLM apps. Treat the chain‑graph as code, test each node, monitor tokens like CPU, and remember that retrieval quality usually matters more than model size.
