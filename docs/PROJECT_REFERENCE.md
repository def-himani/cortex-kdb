# Physical AI kDB

**Consolidated project reference \| NLP / Applied AI**

A self-implemented Pydantic AI agent that compiles vision-language-action (VLA) research papers into a grounded Knowledge DB (kDB).

## 1. Project purpose

The assignment is to build a Python agent that reads a fixed library of 30 research papers and produces a persistent, structured knowledge database. A 31st paper is reserved for a later incremental update. The system must make factual claims traceable to exact sentences or table values in the original PDFs, with highlighted evidence that a human can inspect.

The kDB is designed for both humans and other agents. At grading time, the course's question-answering agent will use the submitted kDB and lake alone, not the raw papers or external knowledge. It should answer questions requiring multiple papers, distinguish incomparable experimental conditions, preserve disagreements, and decline questions whose answers are absent.

## Why this is different from ordinary RAG

| **Conventional retrieval-augmented generation (RAG)**                        | **Knowledge DB approach**                                                                |
|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Retrieve relevant passages at question time.                                 | Compile verified knowledge when a paper arrives.                                         |
| Recombine the source passages for each new question.                         | Query persistent entities, relationships and results.                                    |
| Citations may identify passages without proving that they support an answer. | Store exact evidence locations and annotations alongside claims; validate the locations. |
| Repeated retrieval may consume substantial context.                          | Reuse structured knowledge and retrieve only the needed rows.                            |

The assignment draws on Karpathy's LLM Wiki idea but adds machine-checkable grounding, structured conditions and auditable updates. The objective is to test whether compilation can outperform retrieval on correctness and token use. Grounding tools verify that evidence exists at the specified location; they cannot establish semantic entailment, so human review remains important.

## Project at a glance

- Input: 30 specified VLA PDFs, followed by LIBERO-Plus as paper 31.

- Output: a Python CLI, an OKF bundle in Git, a DuckLake catalog and media, annotated PDFs, and a question history.

- Required evaluation: grounding, incremental changes, question-answering quality and token efficiency, plus a live demo.

- Team size: one or two students. No model training or physical robot is required.

*Reference scope: consolidated from the five supplied course documentation sections and the starter README. This is a study/reference document, not a replacement for the official assignment. No implementation decisions are assumed to have been made.*

# 2. Required architecture and constraints

The student writes the agent and its kDB engine. The model supplies structured extraction and reasoning; Python controls validation, entity merging and persistence.

| **Component**  | **Required or permitted approach**                                                                                                        |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Agent harness  | Python with Pydantic AI; typed Python tools and Pydantic inputs/outputs; reject invalid tool calls and allow retries.                     |
| Model server   | A local OpenAI-compatible server (Ollama, llama.cpp, vLLM, SGLang or LM Studio), or an approved OpenRouter model.                         |
| Configuration  | Read base_url, model name and the API-key environment-variable name from configuration; grading changes the server without changing code. |
| PDF parsing    | Open-source parser preserving page positions (e.g., Docling or MinerU); direct PDF reading permitted.                                     |
| PDF annotation | Highlight with comments on a copy; kdb_tools.annotate.highlight is supplied; PdfCraft or pypdf also accepted.                             |
| Persistence    | Student-written kDB engine calling logged actions; local DuckLake by default, optional locally hosted RustFS/S3-compatible storage.       |
| CLI            | One program exposing ingest, ask, history and ask --save, plus the starter's lake init and export.                                        |

## Model and service restrictions

- Use pretrained models only. No fine-tuning is expected.

- Allowed model providers: local open-weight model; OpenRouter model with ID ending in :free; or paid OpenRouter with a strict credit limit on the API key set before use.

- No other hosted model APIs; do not send document contents, page images or extracted text to other hosted model APIs.

- No commercial agent application (ChatGPT, Claude, Gemini, NotebookLM), hosted agent platform, AI browser, alphaXiv, or framework that runs its own agent to do the submitted agent's work. No other paid service is allowed.

- Do not modify original PDFs. Keep API keys in environment variables, never in committed files.

## Cost and compute expectations

The default storage is local and needs no cloud credentials. Local inference and OpenRouter free models avoid API charges, but hardware, runtime and free-tier quotas may limit feasibility. The course reports approximately 3.6M input tokens and 0.8M output tokens for one 31-paper build. Reference local runs: Gemma 4 12B QAT on a 16 GB GPU took about 19 minutes per paper (nearly 10 hours for the library); a 31B QAT setup is suggested for 24 GB GPU memory. These are reference measurements, not required hardware or guaranteed performance. Runtime is not graded, but one-paper ingest must finish in a reasonable time on the student's hardware.

# 3. Provided infrastructure and student responsibilities

## What the course supplies

| **Tool / resource**      | **Purpose**                                                                                                                       |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| starter/                 | CLI scaffolding, two Pydantic AI agents, configuration, context guard, local-lake setup and offline tests using a scripted model. |
| kdb_tools                | Python functions for validation, grounding, PDF location/annotation, typed actions, configuration and toolsets.                   |
| okf-validate             | Checks OKF structure, schema/row rules, ontology rules and logged page edits.                                                     |
| okf-ground               | Checks quotes and table values, bboxes, matching PDF highlights, registered media and hashes.                                     |
| okf-diff                 | Compares Git/directory versions or lake snapshots; detects unlogged changes and unnecessary page rewrites.                        |
| okf-seed                 | Writes the seed ontologies using cited scope notes and logged actions.                                                            |
| okf-export / okf-restore | Exports the lake and media with a hash manifest; restores only when hashes check.                                                 |
| okf-mcp                  | Exposes checks and actions over MCP; the Python agent may instead use supplied functions directly.                                |
| Seed + VLA schemas       | Core/VLA classes and properties, robotics, datasets and vla-datasets ontologies.                                                  |

## What the student must implement or improve

- Implement write_rows (initially raises NotImplementedError): map extracted facts into correct, deduplicated rows; build evidence, highlights, and typed action calls.

- Improve extraction of facts, training relationships, table results and experimental conditions; handle chunking, context limits, aliases and contradictions.

- Improve the ask agent's search tools to retrieve relevant rows, include narrower ontology terms, and avoid excessive context.

- Implement citation printing, --show, --save, and SQLite question history, including filtering by paper.

- Preserve unchanged rows/pages during incremental ingestion; test and validate the output, export both submission states, and prepare the findings report and demo.

## Starter setup order

The starter README describes copying starter/ into the student's repository, copying kdb.toml.example to kdb.toml, running uv sync and uv run pytest, initializing the lake, then seeding the ontology before ingest. The first offline test run may download the DuckLake extension. Use the assigned team ID (lowercase letters, digits and hyphens). Choose local or RustFS storage before the first lake initialization. Keep the runtime model configuration portable.

# 4. Data architecture and table schema

The kDB itself describes types; the lake holds instances. The kDB is an Open Knowledge Format (OKF) v0.2 bundle of Markdown files with YAML frontmatter, tracked in Git. Individual papers, models, datasets and results are rows, not Markdown pages.

## Expected directory layout

kdb.toml \# model and lake configuration  
compose.yml \# optional local RustFS storage  
kdb/  
index.md, log.md \# page index and action log  
ontology/  
core/, vla/ \# classes and properties  
robotics/, datasets/, vla-datasets/ \# ontology terms  
papers/ \# original PDFs  
annotated/ \# highlighted PDF copies  
lake/  
catalog.ducklake \# catalog, snapshots, actions  
data/ \# data and registered media

## Lake tables

Every table also has iri, type and team columns. Classes and properties define datatypes and required fields.

| **Table**    | **Domain-specific columns**                                                                                      |
|--------------|------------------------------------------------------------------------------------------------------------------|
| papers       | title, arxiv_id, version, media                                                                                  |
| models       | title, aliases, description, model_class, evidence                                                               |
| datasets     | title, aliases, description, collection, embodiments, tasks, modalities, evidence                                |
| trained_on   | model, dataset, role, evidence                                                                                   |
| results      | title, model, task, benchmark, suite, embodiment, perturbation, trials, trials_reported, value, metric, evidence |
| contested    | claim, disputed_by, evidence                                                                                     |
| computations | title, code, inputs, value, evidence                                                                             |
| analyses     | title, question, answer, drew_on, evidence                                                                       |

Rows use stable IRIs beginning with https://data.aegean.ai/vla/\<team\>/. Derive IDs from the entity or result identity (e.g., paper, table, row and column), so reingest resolves to the same rows. Model/dataset reference columns hold row IRIs; conditions, model_class, role and dataset properties hold Term IRIs. Paper media is the original PDF's SHA-256. Each paper/model/dataset must have exactly one row; aliases belong on it. Normalized titles and aliases cannot collide across papers, models and datasets (project.entity). Use full paper titles rather than the model name. Benchmarks, suites and robots are ontology terms, not instance rows. Dataset collection uses datasets-family Term IRIs; each training link names an existing model, dataset, role term and evidence.

# 5. Evidence, grounding and auditable writes

## Evidence and annotation

Every claim needs evidence. A MediaRef identifies registered source media by its SHA-256 hash and pinpoints a supporting sentence or table value. PDF page numbering starts at 1; bbox = \[x0, y0, x1, y1\] in PDF points, with origin at the bottom left. Sentences store quote; table cells store value. Each bbox must correspond to a highlight in a registered annotated copy of the source PDF.

- Quote verbatim from the cited page and within one column. Split a sentence spanning pages into separate MediaRefs.

- Locate the evidence with kdb_tools.locate.find_quote or the value locator. If identical values occur multiple times, identify the correct table cell, not merely any occurrence.

- Highlight the exact evidence using kdb_tools.annotate.highlight; retain existing highlights, including seed annotations. Register the copy with annotates equal to the original PDF's hash, and re-register after adding highlights.

- A successful grounding check is necessary but not sufficient: a number may be on the page yet assigned to the wrong experiment. A TA checks semantic support manually.

## Actions and provenance

All page and lake changes must go through typed actions that validate inputs, apply changes transactionally, and log the affected rows/pages. Do not hand-edit generated pages, index.md or log.md, even for whitespace changes.

| **Action**    | **Responsibility**                                                      |
|---------------|-------------------------------------------------------------------------|
| RegisterMedia | Register source or annotated file; identical bytes cause no change.     |
| UpsertObjects | Insert or patch rows in one class. First dry_run, then submit the plan. |
| DeleteObjects | Remove rows with a reason; not a substitute for incremental updates.    |
| SeedOntology  | Write ontology pages and record their changes.                          |
| RevertAction  | Undo an earlier action when no later change affects its targets.        |

The starter supplies open_context for actions. UpsertObjects preserves omitted columns, clears explicitly null columns, and replaces list columns as a whole. Preserve the existing aliases and append new ones when updating an alias list. A stale plan fails instead of applying changes to a different state.

## Results, computations and disputes

A result records task, benchmark, suite, embodiment and any perturbation as robotics-family Term IRIs, plus a metric/value. Record reported trial count; if absent set trials_reported=false and leave trials empty. Compare like-for-like conditions only. A calculated value belongs in computations with executable short Python and inputs as \<row IRI\>#\<column\>. For contradictory evidence under the same conditions, keep both sources/rows and add contested with claim and disputed_by references in the form \<row IRI\>#evidence/\<zero-based index\>. A different perturbation or other experimental setup is not by itself a contradiction.

# 6. Ontology: meaning and relationships

An ontology gives terms stable identifiers and a hierarchy so the system can distinguish concepts with similar labels and find narrower concepts when answering broader questions. For example, LIBERO-90 may denote an evaluation suite or a pretraining dataset; context determines which meaning is intended.

## Seed ontologies

| **Ontology** | **Scope**                                                                                    |
|--------------|----------------------------------------------------------------------------------------------|
| robotics     | Robot tasks, benchmarks and suites, embodiments, LIBERO-Plus perturbations, VLA model class. |
| datasets     | General dataset collection methods and dataset roles.                                        |
| vla-datasets | Specialized robot-learning dataset terms; extends datasets.                                  |

Every term is a page under ontology/\<ontology\>/, with an IRI, pref_label, alt_labels, broader, and optional close_match or related_match to vocabularies such as SOMA. The body is a cited scope note. A specialized ontology refines its parent instead of redefining its terms. No cycles in broader or extends; labels must not collide within an ontology or its ancestors. Unrelated ontologies may share a label.

The ingest agent uses labels, aliases and source context to map task names, conditions and dataset roles to Term IRIs. Queries traverse broader relationships in the reverse direction to include narrower terms. For example, a pretraining query can include cross-embodiment pretraining. A pick-and-place query should include narrower tasks, while grouping results by benchmark, suite, embodiment, trials and perturbation before comparison.

## Initialization and extensions

Run kdb lake init first, then okf-seed kdb once, before the first ingest. The seed needs these ten PDFs (exact versions are in Section 10): CALVIN, LIBERO, RT-2, BridgeData V2, Open X-Embodiment, SIMPLER, Octo, π0, OpenVLA-OFT and LIBERO-Plus. LIBERO-Plus is present for the seed's term evidence, but no instance row may cite it before Submission 2.

When a missing term is genuinely needed, add a separate team ontology extending the appropriate seed. Define a YAML seed spec with domain, title, description, base_iri, extends, papers and terms. Each new term has a slug, kind, pref_label, alt_labels, broader and a scope note with the source paper/page/quote. Run okf-seed kdb --spec \<file\> followed by okf-validate. Seeding locates and highlights the cited text and writes pages using SeedOntology. The validator detects label collisions (project.term-label); revise and reseed rather than editing the generated pages. Well-chosen extensions earn stretch credit.

# 7. Ingest agent: the paper-to-knowledge pipeline

The required command ingest \<pdf\> adds one paper to an existing kDB and lake. The starter already registers a PDF and collects typed facts from three-page chunks. The student completes and improves this workflow.

## Expected flow

1\. Register the original PDF with RegisterMedia and associate its identity/version with a paper row.

2\. Parse page text and table structure with coordinates. Split two-column layouts and side-by-side tables so that values do not cross into unrelated rows.

3\. Feed manageable page chunks, relevant ontology terms and existing entity names to the Pydantic AI extraction agent. Return typed facts with copied quotes and table results with values and positions.

4\. Validate each quote or value against the PDF, using the provided locators and tools. Return failures to the model for correction. After limited retries, drop unsupported extractions rather than write them.

5\. Resolve canonical model and dataset entities, aliases, contextual ontology terms, training links, and result conditions. Resolve “ours” to the paper's actual model; create required model rows for models appearing only in result tables.

6\. Merge facts in Python (not by letting the model write the database directly). Generate stable row IDs, retain existing knowledge, and detect disputes without comparing dissimilar conditions.

7\. Construct evidence with exact bboxes; highlight evidence in the annotated PDF and register the copy. Write rows using dry_run and submit with UpsertObjects.

8\. Run validation and grounding on the affected rows. Do not report unsupported data as valid; inspect extraction/condition errors separately.

## Prompt and context management

The documentation warns that some servers silently truncate oversized prompts; one test reduced a 45,000-token prompt to 16,387 tokens on a 32K context setup. A paper converted to Markdown can reach 25,000–35,000 tokens. Estimate request size before sending, leave room for outputs, compare returned usage.prompt_tokens with the sent estimate, and investigate major gaps. Keep tool outputs capped and retrieve quotes only when needed. A listing of roughly 450 reference rows consumed around 26,000 tokens and disrupted subsequent requests.

Provided tools include find_quote, find_value, find_cell and check_pages (target a page or \<table\>/\<iri\>). The starter's guard and extraction validators already support part of this workflow. The student is responsible for the quality of extraction, row construction and context strategy.

# 8. Ask agent and command-line interface

The ask agent must answer using only the lake and kDB. It is read-only: it has no action tools. The starter provides list_classes, query_objects (plain equality and normally at most 50 rows) and get_object. Extend these with relevant search, ontology descendant lookup, benchmark/suite filtering or read-only SQL. The agent's structured output is a list of claims citing row IDs and specific evidence entries; validators reject nonexistent references.

| **Command**                      | **Expected behavior**                                                                                                          |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| kdb lake init --team \<team\>    | Initialize local lake, core/VLA schemas, action log and media; supplied.                                                       |
| kdb ingest \<pdf\>               | Ingest one paper, write grounded changes and register annotated PDF.                                                           |
| kdb ask "\<question\>"           | Answer with row and evidence citations for each sentence; print paper, page and quote/value. Refuse when the answer is absent. |
| kdb ask --show "\<question\>"    | Also display each cited highlighted region inline in compatible terminals, or output a PNG otherwise.                          |
| kdb history                      | List earlier questions, answers and citations.                                                                                 |
| kdb history --paper \<arxiv-id\> | Filter history to answers citing that paper.                                                                                   |
| kdb ask --save "\<question\>"    | Save as analyses with question, answer, drew_on rows and copied evidence; only save if validation and grounding pass.          |
| kdb export \<folder\>            | Export current lake snapshot, media and manifest for submission; supplied.                                                     |

## History and saved analyses

Store history in a SQLite file next to the kDB. Record the question, answer, every citation, model, token counts, Git commit, lake snapshot and full conversation. Commit/snapshot provenance allows earlier answers to be traced to the data version they used, even after later ingestion changes rows.

The --save CLI path may write an analyses row through the required actions after validation; the ask agent itself remains read-only. Failed validation or grounding must leave no saved analysis. Later queries may use successfully saved analyses. Each sentence must remain evidence-backed, including multi-paper answers.

## Example questions the system should support

- Show pick-and-place results, including narrower tasks, grouped by comparable experimental conditions.

- Which VLA models trained on the same datasets, and in which roles?

- What does the library actually establish about a result or reported condition? If it does not establish the answer, explicitly say so rather than infer an unsupported answer.

# 9. Incremental updates, assessment and submissions

## Incremental correctness

A new paper changes only affected rows. Do not delete/reinsert all rows or rebuild the whole kDB. Each action log must name inserted, updated or deleted rows with before/after values. Unaffected rows are not written at all. Pages change only for a new or modified term/type; all other pages stay byte-identical, including frontmatter and generated stamps.

To test: hold one paper out, build from the other 29, commit and note the lake snapshot. Ingest the held-out paper, commit again and note the new snapshot. Compare with the commands below (replace snapshot IDs with actual values).

uv run python -c "from kdb_tools.lake import Lake, load_config; print(Lake.open(load_config()).snapshot())"  
uv run okf-diff kdb --lake-config kdb.toml --snapshots 18 34  
uv run okf-diff kdb --git HEAD~1 HEAD

The snapshot diff includes tables, media and action IDs; unlogged changes yield diff.unlogged. The Git diff checks unchanged rewrites and generated stamps. Do not use LIBERO-Plus as a trial ingest in the first submission state.

## Deliverables and grading

| **Submission** | **Required content**                                                                                                                                                                                                                                   |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1              | Agent code and CLI, grounded kDB from the 30 papers, ontology, papers and annotations, matching lake export. No instance row cites LIBERO-Plus.                                                                                                        |
| 2              | Same kDB after one ingest of LIBERO-Plus, its annotated PDF, matching new export and a findings report (maximum two pages) listing all changed rows/pages and reasons, contradictions or superseded claims with both sources, and every contested row. |

- Grounding: both submissions pass okf-validate and okf-ground; manual sample checks semantic support. Unsupported claims score zero.

- Incremental update: grader compares actual changes and report with the reference ingest; checks contradictions, action logs and that unrelated data remains unchanged.

- Question answering: unpublished questions include multi-paper and unanswerable cases. Course agent runs against the final kDB using a fixed token budget, compared with PaperQA2 over raw PDFs on the same model. Aim for better correctness and fewer tokens.

- Live demo: answer multi-paper and unanswerable questions, show highlighted evidence, save a valid analysis and show history.

Export using uv run kdb export "\<shared Drive folder\>". This writes \<team\>-\<snapshot\> with catalog, data, media and manifest.json. Restore-test with uv run okf-restore "\<shared Drive folder\>/\<team\>-\<snapshot\>" restored. Export at the submitted Git commit; course uses GitHub push times, not file timestamps.

# 10. Required paper library

Download exactly the assigned arXiv versions. Save each as kdb/papers/\<arxiv-id\>.pdf (without the version suffix). Later versions may move evidence and fail grading. To download, use https://arxiv.org/pdf/ followed by the versioned ID below. Asterisk (\*) denotes the ten seed papers needed before okf-seed.

| **\#** | **Paper**            | **Versioned arXiv ID** | **\#** | **Paper**      | **Versioned arXiv ID** |
|--------|----------------------|------------------------|--------|----------------|------------------------|
| 1      | CALVIN \*            | 2112.03227v4           | 17     | TraceVLA       | 2412.10345v3           |
| 2      | RT-1                 | 2212.06817v2           | 18     | π0 \*          | 2410.24164v4           |
| 3      | ACT (ALOHA)          | 2304.13705v1           | 19     | CoT-VLA        | 2503.22020v1           |
| 4      | BridgeData V2 \*     | 2308.12952v3           | 20     | Dita           | 2503.19757v2           |
| 5      | Diffusion Policy     | 2303.04137v5           | 21     | FAST (π0-FAST) | 2501.09747v1           |
| 6      | LIBERO \*            | 2306.03310v2           | 22     | GR00T N1       | 2503.14734v2           |
| 7      | Open X-Embodiment \* | 2310.08864v9           | 23     | NORA           | 2504.19854v1           |
| 8      | RT-2 \*              | 2307.15818v1           | 24     | OpenVLA-OFT \* | 2502.19645v2           |
| 9      | CogACT               | 2411.19650v1           | 25     | RL4VLA         | 2505.19789v4           |
| 10     | DROID                | 2403.12945v2           | 26     | SimpleVLA-RL   | 2509.09674v1           |
| 11     | Octo \*              | 2405.12213v2           | 27     | SmolVLA        | 2506.01844v1           |
| 12     | OpenVLA              | 2406.09246v3           | 28     | SpatialVLA     | 2501.15830v5           |
| 13     | RDT-1B               | 2410.07864v2           | 29     | UniVLA         | 2505.06111v3           |
| 14     | RoboVLMs             | 2412.14058v4           | 30     | π0.5           | 2504.16054v1           |
| 15     | SIMPLER \*           | 2405.05941v1           | 31     | LIBERO-Plus \* | 2510.13626v3           |
| 16     | TinyVLA              | 2409.12514v5           |        |                |                        |

LIBERO-Plus is the reserved paper for Submission 2. Its PDF is present initially for ontology seeding, not general ingestion or first-submission instance evidence.

# 11. Reference checklist and official sources

## Mandatory implementation checklist

☐ Self-written Pydantic AI agent and portable approved model configuration.

☐ Grounded kDB with the correct 30-paper library and initialized ontologies.

☐ All extracted evidence has exact locations, registered media and matching PDF highlights.

☐ Canonical model/dataset entities and correct aliases; each training dataset has a trained_on link and role.

☐ Results linked to the appropriate ontology terms; trial counts reported or explicitly unknown.

☐ Correct condition-aware queries, computations when needed, and preservation of disputes.

☐ Single-paper incremental ingest, logged actions and untouched unrelated rows/pages.

☐ ask, --show, --save, history and paper-filtered history; grounded refusals and complete provenance.

☐ Both submission states validated, exported, restored for verification, committed and pushed.

☐ Submission 2 findings report and live demo readiness.

## Optional stretch objectives

Standing queries refreshed on each ingest; cited, non-duplicative ontology extensions; a short write-up explaining how the agent processes papers beyond the context window, what information it retains across chunks, and how it detects contradictions with older information outside its current context.

## Authoritative references

The following are the source pages supplied for this consolidated reference. Consult the official version for exact requirements or later revisions.

- Main assignment: https://aegean.ai/aiml-common/projects/nlp/physical-ai-kdb.md

- Inputs and paper library: https://aegean.ai/aiml-common/projects/nlp/physical-ai-kdb/inputs.md

- System and agents: https://aegean.ai/aiml-common/projects/nlp/physical-ai-kdb/system.md

- kDB format and examples: https://aegean.ai/aiml-common/projects/nlp/physical-ai-kdb/kdb-format.md

- Ontology, incremental updates and CLI: https://aegean.ai/aiml-common/projects/nlp/physical-ai-kdb/ontology.md

- Course repository: https://github.com/pantelis/physical-ai-kdb

- Starter README: https://github.com/pantelis/physical-ai-kdb/blob/main/starter/README.md

- Documentation index: https://aegean.ai/llms.txt

- OKF specification: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md

*Use this file as stable assignment context for future study sessions. Keep implementation progress and design decisions separately so that they are not confused with course requirements.*
