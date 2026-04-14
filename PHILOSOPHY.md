# Design Philosophy

## The core premise

The entire architecture started from one observation: human memory is not a single flat context window. It is fragmented, hierarchical, and selective, and that selectivity is what makes it work. A brain that remembered everything with equal weight and equal accessibility would be unusable. Transformers, by contrast, do exactly that: one flat window, everything attended to simultaneously. That is the problem Active Memory was designed to fix.

## What was modeled and why

### Working memory as RAM

The Bucket maps directly to working memory. Humans hold roughly four to seven things in active attention at once. The Bucket enforces a similar constraint: a structured topic summary, a limited recent message stack, a set of retrieved associations ranked by relevance. The Active Agent only ever sees the Bucket. This is not a limitation. It is the design. Bounded context is what prevents overload.

### The hippocampus as the Curator

Humans do not consciously decide what to remember. Something else evaluates experiences and makes that call silently in the background. The Curator is that process. It runs after the response, invisible to the user, evaluating what just left the recent stack. The user never waits for it. It never participates in conversation. It only judges. This maps precisely to how episodic memory encoding works in humans: outside conscious awareness, triggered by the passage of an experience through the system.

### The reticular activating system as the Observer

The brain has a filter that decides what reaches conscious attention before reasoning ever begins. The Observer is that filter. It watches the conversation and maintains a structured topic tree: a hierarchy of named topics, each carrying its own summary, creation timestamp, last-updated timestamp, and a staleness annotation recording how many turns have passed since it was last touched. When new information arrives through eviction, the Observer merges it into existing topic nodes or creates new ones. It never duplicates. Topics that have not been updated in many turns naturally carry high staleness values, giving the model a temporal signal analogous to the way older human memories feel less immediate. The Observer does not reason about content. It manages the aperture through which content passes.

### Associative recall as the Retrieval layer

Human memory recall is not a database query. No one searches for memories by keyword. A smell, a word, a concept fires related memories automatically through association. The Retrieval layer replicates this using vector similarity: the meaning of the incoming message is compared against stored memories mathematically, and semantically related memories surface automatically without the Active Agent asking for them. The trigger is associative, not deliberate.

There is no hard cap on how many memories can surface. Everything above the similarity threshold is returned, ranked by relevance score and annotated with its tier (warm or cold) so the model can weight accordingly. This mirrors the way human recall does not impose a fixed limit on associations. What surfaces is determined by relevance, not by an arbitrary count.

### Forgetting as a feature

This was an explicit design principle. Humans stay cognitively sharp because they forget the right things. The topic tree forces forgetting through structure: when the Observer updates a topic, older detail is compressed into the node's summary rather than preserved verbatim. The tiered storage model means old memories do not remain equally accessible forever. Things sink to cold storage, then potentially get pruned. This is not data loss. It is the mechanism that keeps the system from degrading the way a flat context window does.

### Sleep consolidation as the Librarian

The brain reorganizes memory during sleep: merging redundant memories, strengthening frequently accessed ones, clearing what is no longer relevant. The Librarian is that process. It runs nightly on a schedule, never during live conversation. It promotes frequently retrieved memories from cold to warm storage, demotes stale ones, and prunes memories that have never been retrieved and are very old. The nightly schedule is intentional. Consolidation is a downtime process in both humans and in this system.

### Memory strengthening through repetition

This was the insight that most clearly separated Active Memory from a filing system. In human memory, a fact encountered repeatedly becomes more deeply embedded: faster to access, more reliably recalled. A fact encountered once and never revisited fades. The retrieval count metadata on every stored memory replicates this exactly. Every time a memory is surfaced its count increments. The Librarian uses that count to decide what to promote to warmer storage. Memories that matter rise. Memories that do not, sink or fade.

### Temporal context as a memory signal

Human memories carry a sense of when they happened. Not a precise timestamp, but a feeling of recency or distance. The topic tree encodes this through its staleness annotations: each topic node records the turn at which it was last updated, and the context string renders this as "updated N turns ago." A topic updated two turns ago feels current. A topic untouched for fifty turns feels distant. This gives the model a lightweight temporal signal without requiring it to parse raw timestamps or reason about absolute dates.

## What was explicitly decided against

### Storing reasoning traces

Capturing the model's chain of thought as a stored memory was considered and rejected for the same reason humans do not store reasoning. What matters is the decision and the context that produced it, not the reasoning process itself. Given the same inputs, the same reasoning reconstructs naturally. Storing reasoning is expensive and redundant. Storing decisions with their context is lean and sufficient.

### Continuous full attention

Having all components aware of the full conversation at all times was considered and rejected because constant full attention is not intelligence, it is pathological. In humans it manifests as sensory overload. In the system it would produce exactly the degradation Active Memory was designed to prevent. Each component sees only the narrow slice it needs to do its specific job.

### Substrate dependence

The memory system was deliberately designed to be model-agnostic. Any object that implements `chat(messages) -> str` works as a backend. This reflects a view that memory processes are independent of the substrate that performs reasoning. The same memory architecture should work whether the underlying model is a local 4B parameter model or a frontier API, just as human memory processes operate the same way regardless of individual differences in processing speed.

## The design test

Every component in the system was evaluated against one question: does this map to how biological memory actually works, or is it just engineering convenience?

Two components failed that test early. The Conductor, in its original form, was an if-statement dressed as an agent. The Reporter was a database call with a name. Both were evaluated against the biological analogue criterion, found to be pure engineering scaffolding, and simplified out of the architecture. What remained maps cleanly to identifiable cognitive processes, each with a biological analogue that informed its design.
