"""ThaiGABM-SEIR: a generative agent-based behavioural layer coupled to a
mechanistic stratified SEIR model, parameterised for the Thai context.

Layering (see ThaiGABM_SEIR_project_guideline.md):

    Thai evidence  ->  synthetic population (personas + households)
                   ->  behavioural model (LLM agents / rules)
                   ->  behavioural aggregator (per-stratum, per-setting)
                   ->  mechanistic stratified SEIR
                   ->  information environment  --(feedback)-->  agents

The LLM never touches epidemiology: it only produces *behaviour*.
"""

__all__ = ["config", "evidence", "strata", "persona", "actions", "aggregator",
           "seir", "information", "runner", "validation", "experiments"]
