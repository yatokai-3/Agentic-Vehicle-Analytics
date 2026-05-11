# 🚗 VahanGraph
### Agentic Vehicle Registration Analytics using LangGraph

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/LangGraph-Agentic_Workflow-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LLM-Powered-black?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active_Development-success?style=for-the-badge"/>
</p>

---

> ⚡ An AI-powered analytics workflow that converts natural language questions into executable Python analysis over large-scale Indian vehicle registration datasets.

---

# ✨ Overview

VahanGraph is a **human-in-the-loop analytics agent** built using:

- 🧠 LangGraph
- 🐼 Pandas
- 🤖 LLM-powered code generation
- 📊 Dynamic visualization pipelines

Instead of manually writing analysis scripts, users can simply ask questions like:

```text
What was the trend of EV registrations in Delhi from 2020-2024?
```

or

```text
Compare diesel vehicle registrations between Bihar and Odisha.
```

The system will:

✅ Understand the query intent  
✅ Extract metadata automatically  
✅ Route to the correct analytical workflow  
✅ Generate Python analysis code dynamically  
✅ Ask for human approval/review  
✅ Iteratively improve generated code using feedback  
✅ Execute approved analysis  
✅ Summarize insights in natural language  

---

# 🧠 Core Features

## 🔍 Natural Language Analytics

Ask analytical questions in plain English.

### Examples

```text
What was the trend of EV registrations in Delhi from 2020-2024?
```

```text
Compare EV registrations between Delhi and Bihar.
```

```text
Show fuel-wise distribution of registrations in Odisha.
```

---

## 🔄 Agentic Workflow with LangGraph

The system uses graph-based orchestration for:

- Metadata extraction
- Intelligent routing
- Iterative code generation
- Human approval loops
- Execution pipeline management

---

## 👨‍💻 Human-in-the-Loop Refinement

Generated code is **never blindly executed**.

Users can:

- ✅ Approve generated code
- ❌ Reject generated code
- ✍️ Provide feedback
- 🔁 Iteratively refine analysis logic

### Example Feedback

```text
Use exact ELECTRIC(BOV) instead of broad ELECTRIC matching.
```

The agent regenerates improved code using accumulated human feedback history.

---

## ⚙️ Dynamic Python Code Generation

The LLM dynamically generates:

- Pandas analysis logic
- Aggregations
- Charts & visualizations
- Statistical summaries

while operating directly on an already-loaded DataFrame.

---

## 🗂️ Multi-State Vehicle Registration Analytics

Supports analysis across:

- Delhi
- Bihar
- Odisha
- Additional states via scalable folder-based ingestion

---

# 🏗️ System Architecture

```text
User Query
    ↓
Metadata Extraction
    ↓
Intent Routing
    ↓
Code Generation
    ↓
Human Approval Loop
    ↓
Code Execution
    ↓
Result Summarization
```

---

# 📊 Supported Analysis Types

---

## 📈 Trend Analysis

Time-series analysis over vehicle registration data.

### Example

```text
What was the trend of EV registrations in Delhi from 2020-2024?
```

---

## ⚖️ Comparison Analysis

Entity-vs-entity comparison workflows.

### Example

```text
Compare EV registrations between Delhi and Bihar.
```

---

## 🥧 Breakdown Analysis

Composition/distribution analysis.

### Example

```text
Show fuel-wise distribution of registrations in Odisha.
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangGraph | Workflow orchestration |
| Pandas | Data analysis |
| Matplotlib | Visualization |
| Groq API | LLM inference |
| Pydantic | Structured outputs |
| TypedDict | Graph state management |

---

# 🧩 Repository Structure

```text
project/
│
├── data/
│   ├── Delhi/
│   ├── Bihar/
│   └── Odisha/
│
├── notebooks/
│
├── master_vehicle_data.csv
│
├── main.py
│
└── README.md
```

---

# 🎯 Project Goals

This project explores:

- Agentic analytics systems
- Human-supervised code generation
- LLM-based workflow orchestration
- Interactive analytical reasoning
- Safe execution pipelines

---

# 💡 Why This Project?

Most analytics systems still require:

- SQL expertise
- Manual scripting
- Static dashboards
- Repetitive workflows

VahanGraph attempts to bridge that gap by combining:

- Natural language understanding
- Dynamic code generation
- Graph-based reasoning
- Human oversight

into a single intelligent analytical workflow.

---

# ⭐ If you found this interesting...

Consider starring the repository and contributing ideas 🚀
