# VahanGraph
### Agentic Vehicle Registration Analytics using LangGraph

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/LangGraph-Agentic_Workflow-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LLM-Powered-black?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active_Development-success?style=for-the-badge"/>
</p>

---

> An AI-powered analytics workflow that converts natural language questions into executable Python analysis over large-scale Indian vehicle registration datasets.

---


TRY THE APPLICATION HERE -> https://agentic-vehicle-analytics-jpszmntgvh5n72n8wmct9f.streamlit.app/
#  Overview

VahanGraph is a **human-in-the-loop analytics agent** built using:

- LangGraph
- Pandas
- LLM-powered code generation
- Dynamic visualization pipelines

Instead of manually writing analysis scripts, users can simply ask questions like:

```text
What was the trend of EV registrations in Delhi from 2020-2024?
```

or

```text
Compare diesel vehicle registrations between Bihar and Odisha.
```

The system will:

 - Understand the query intent  
 - Extract metadata automatically  
 - Route to the correct analytical workflow  
 - Generate Python analysis code dynamically  
 - Ask for human approval/review  
 - Iteratively improve generated code using feedback  
 - Execute approved analysis  
 - Summarize insights in natural language  

---

# System Architecture

<img width="3972" height="3184" alt="image" src="https://github.com/user-attachments/assets/ad19c343-cf0c-4ae8-acfa-0981b220062a" />


---

# Supported Analysis Types

---

## Trend Analysis

Time-series analysis over vehicle registration data.

### Example

```text
What was the trend of EV registrations in Delhi from 2020-2024?
```

---

## Comparison Analysis

Entity-vs-entity comparison workflows.

### Example

```text
Compare EV registrations between Delhi and Bihar.
```

---

## Breakdown Analysis

Composition/distribution analysis.

### Example

```text
Show fuel-wise distribution of registrations in Odisha.
```

---

# Repository Structure

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

# Project Goals

This project explores:

- Agentic analytics systems
- Human-supervised code generation
- LLM-based workflow orchestration
- Interactive analytical reasoning
- Safe execution pipelines

---

# If you found this interesting...

Consider starring the repository and contributing ideas 
