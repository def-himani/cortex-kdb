## ADR-006: Local Model Infrastructure

Status: Accepted

### Decision
- Model: Gemma 4 12B IT QAT
- Server: Ollama
- Custom model: cortex-gemma4:16k
- Context window: 16384 tokens
- Reasoning effort: none
- Endpoint: http://localhost:11434/v1

### Hardware
- MacBook Air M4
- 10-core CPU
- 10-core GPU
- 16 GB unified memory

### Reason
Avoid external API costs and rate limits while maintaining
an OpenAI-compatible model interface.

### Validation
- Local inference: Passed
- GPU offloading: 100%
- Context window: Verified
- API connectivity: Passed
- Pydantic AI integration: Passed
- Structured output: Passed
- Tool calling: Passed
- Token usage reporting: Passed

### Limitations
Full-library ingestion performance has not yet been evaluated.
Local inference is constrained by memory, processing speed,
and sustained computational workload.


## ADR-007: Resource-Conscious Development

Status: Accepted

### Decision
Build and validate application components using scripted
model responses and synthetic test data wherever possible.

Reserve real-model inference for targeted integration tests
and final paper ingestion.

### Reason
Avoid exhausting API quotas, unnecessary computational
workloads, and repeated processing of research papers.

### Development Policy
- Prefer offline unit tests.
- Avoid unnecessary external API integrations.
- Avoid repeatedly ingesting complete research papers.
- Do not introduce paid services without explicit approval.
- Complete and validate the pipeline before full ingestion.
- Preserve the completed database for submission and demo.

### Budget
Target additional service expenditure: $0.